.. _Bootstrap:


====================
Bootstrap statistics
====================

We have published a paper on systematically addressing uncertainty in atomistic machine learning, in which we focused on a basic bootstrap ensemble method:


    Peterson, Christensen, and Khorshidi, "Addressing uncertainty in atomistic machine learning", *PCCP* 19:10978-10985, 2017. |uncertainty_paper|


.. |uncertainty_paper| raw:: html

   <a href="http://dx.doi.org/10.1039/C7CP00375G" target="_blank">DOI:10.1039/C7CP00375G</a>

A helper module to create bootstrap calculators, which are capable of giving not just a mean model prediction, but uncertainty intervals, is described here.
Note that you should use uncertainty intervals with caution, and, as we describe in the above paper, the "correct" interpretation of seeing large uncertainty bounds for a particular atomic configuration is that a new electronic structure calculation is required (at that configuration), and *not* that the true median will lie within those bounds.

Training
--------

The below script shows a simple example of creating a bootstrap ensemble of 10 calculators for a small sample training set.
(But you probably want an ensemble size much larger than 10 for reasonable statistics!)

.. code-block:: python

    from amp.utilities import Logger
    from amp.stats.bootstrap import BootStrap


    def generate_data(count, filename='training.traj'):
        """Generates test or training data with a simple MD simulation."""
        import os
        from ase import Atoms, Atom, units
        import ase.io
        from ase.calculators.emt import EMT
        from ase.build import fcc110
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md import VelocityVerlet
        from ase.constraints import FixAtoms
        if os.path.exists(filename):
            return
        traj = ase.io.Trajectory(filename, 'w')
        atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
        atoms.extend(Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                            Atom('Cu', atoms[7].position + (0., 0., 5.))]))
        atoms.set_constraint(FixAtoms(indices=[0, 2]))
        atoms.calc = EMT()
        atoms.get_potential_energy()
        traj.write(atoms)
        MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
        dyn = VelocityVerlet(atoms, dt=1. * units.fs)
        for step in range(count - 1):
            dyn.run(50)
            traj.write(atoms)


    generate_data(5, 'training.traj')

    calc_text = """
    from amp import Amp
    from amp.descriptor.gaussian import Gaussian
    from amp.model.neuralnetwork import NeuralNetwork
    from amp.model import LossFunction

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(),
               dblabel='../amp-db',
               envcommand='loadamp')
    calc.model.lossfunction = LossFunction(force_coefficient=0.,
        convergence={'force_rmse': None})
    """

    start_command = 'python run.py'

    calc = BootStrap(log=Logger('bootstrap.log'))
    calc.train(images='training.traj', n=10, calc_text=calc_text,
               start_command=start_command, label='bootstrap')

Run the above script once and wait for it to finish (probably <1 minute).
You will see lots of directories created with the ensemble calculators.
Run the same script *again*, and it will clean up / archive these directories into a compressed (.tar.gz) file, and create a calculator parameters file called 'bootstrap.ensemble', which you can load with `Bootstrap(load='bootstrap.ensemble')`, as described later.

First, some notes on the above. The individual calculators are created with the `calc_text` variable in the above script; you can modify things like neural network size or convergence criteria in this text block.

In the above, the optional `start_command` is the command to start the job, which defaults to "python run.py".
Here, it runs each calculator's training sequentially; that is, after one finishes it starts the next.
If your machine has >10 cores, or you don't mind the training processes all competing for resources, you can have them all run in parallel by placing an ampersand (in \*nix systems) at the end of this line, that is "python run.py &".

Most likely, you want to run this on a high-performance computing cluster that uses a queuing system.
In this case, `start_command` is your queuing command, for our SLURM system this is just

.. code-block:: python

    start_command = 'sbatch run.py'

If you need to supply headerlines to your queuing system, you can do them with something like the below.

.. code-block:: python

    headerlines = """#SBATCH --time=00:30:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=8
    #SBATCH --partition=batch
    """

    ...

    calc.train(images='training.traj', n=10, train_line=train_line,
               calc_text=calc_text, headerlines=headerlines,
               start_command=start_command, label='bootstrap')


In a similar way, you can also supply a custom `train_line` if necessary; see the module's autodocumentation for details.

Loading and using
-----------------

The bootstrap ensemble can be loaded via the calculator's load keyword.
The below script shows an example of loading the calculator, and using it to predict the energies and the spread of the ensemble for the training images.

.. code-block:: python

		  import ase.io
		  from amp.stats.bootstrap import BootStrap


		  calc = BootStrap(load='bootstrap.ensemble')

		  traj = ase.io.Trajectory('training.traj')

		  for image in traj:
				energies = calc.get_potential_energy(image,
                                                 output=(0.05, 0.5, 0.95))
				print(energies)
				energy = image.get_potential_energy()
				print(energy)

Note that the call to `calc.get_potential_energy` returns *three* energy predictions, at the 5th, 50th (median), and 95th percentile, as specified with the tuple (0.05, 0.5, 0.95).
When you run this, you should see that the median prediction matches the true energy (from `image.get_potential_energy`) quite well, while the spread in the data is due to the sparsity of data;  as described in our paper above, this ensemble technique punishes regions of the potential energy surface with infrequent data.

Hands-free training
-------------------
In typical use, calling the :py:meth:`~amp.stats.bootstrap.BootStrap.train` method of the :py:class:`~amp.stats.bootstrap.BootStrap` class  will spawn many independent training jobs.
Subsequent calls to `train` will help you manage those jobs: checking which have converged, checking which failed to converge (and re-submitting them), checking which timed out (and re-submitting them), and, if all converged, creating a bundled calculator.
It can be most efficient to submit a (single-core) job that repeatedly calls this command for you and acts as a job manager until all the training jobs are complete.
This can be achieved by taking advantage of the `results` dictionary returned by train, as in the below example script which uses SLURM environment commands. 

.. code-block:: python

    #!/usr/bin/env python
    #SBATCH --time=50:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --partition=batch

    import time
    from amp.stats.bootstrap import BootStrap
    from amp.utilities import Logger

    calc_text = """
    from amp import Amp
    from amp.model.neuralnetwork import NeuralNetwork
    from amp.descriptor.gaussian import Gaussian
    from amp.model import LossFunction


    calc = Amp(model=NeuralNetwork(),
               descriptor=Gaussian(),
               dblabel='../amp-db')
    calc.model.lossfunction = LossFunction(convergence={'force_rmse': 0.02,
                                                        'force_maxresid': 0.03})
    """

    headerlines = """#SBATCH --time=05:30:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=8
    #SBATCH --partition=batch
    """

    start_command = 'sbatch run.py'

    calc = BootStrap(log=Logger('bootstrap.log'))

    complete = False
    count = 0
    while not complete:
        results =  calc.train(images='training.traj',
                              n=50,
                              calc_text=calc_text,
                              start_command=start_command,
                              label='bootstrap',
                              headerlines=headerlines,
                              expired=360.)
        calc.log('train loop: ' + str(count))
        count += 1
        complete = results['complete']
        time.sleep(120.)
