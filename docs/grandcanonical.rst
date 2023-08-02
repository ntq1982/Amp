.. _Grandcanonical:


=======================================
Electronically grand-canonical learning
=======================================

Electronic structure at controlled potential
--------------------------------------------

Many electrochemical simulations are now operating in the electronically grand-canonical ensemble; that is, at constant voltage (work function).
In these electronic structure calculations, the user specifies a desired work function, and the software varies the number of electrons in the simulation in order to find this work function.
This enables the user to do things like search for reaction barriers at a specified potential, rather than have it vary over the course of an elementary reaction.

For example, our group has produced the |sjm_gpaw| (SJM) within the electronic structure code GPAW, described here:


    Kastlunger, Lindgren, Peterson. "Controlled-potential simulation of elementary electrochemical reactions: proton discharge on metal surfaces." *The Journal of Physical Chemistry C* 122:12771-12781, 2018. |sjm_paper|

We describe the thermodynamics of this ensemble here:

    Lindgren, Kastlunger, Peterson. "Electrochemistry from the atomic scale, in the electronically grand-canonical ensemble." *The Journal of Chemical Physics* 157:180902, 2022. |gc_paper|

.. |sjm_gpaw| raw:: html

    <a href="https://wiki.fysik.dtu.dk/gpaw/documentation/sjm/sjm.html" target="_blank">Solvated Jellium Method</a>


.. |sjm_paper| raw:: html

    <a href="http://dx.doi.org/10.1021/acs.jpcc.8b02465" target="_blank">DOI:10.1021/acs.jpcc.8b02465</a>

.. |gc_paper| raw:: html

    <a href="http://dx.doi.org/10.1063/5.0123656" target="_blank">DOI:10.1063/5.0123656</a>


Atomistic learning at fixed potential: the dual-learning scheme
---------------------------------------------------------------

We have developed a new scheme that allows for atomistic machine-learning calculations to operate in the electronically grand-canonical ensemble; that is, at user-specified potentials.
A manuscript is under peer review that describes and demonstrates this approach in detail; we will put a full link below when it becomes available:

    Xi Chen, Muammar El Khatib, Per Lindgren, Adam Willard, Andrew J. Medford, Andrew A. Peterson. "Atomistic learning in the electronically grand-canonical ensemble." *In review*, 2023.

This scheme employs two parallel neural networks for each atom.
The first neural network, takes as input the atomic positions and the simulation potential (or work function), and outputs the per-atom charges.
The sum of these (that is, the net system charge) is trained against the net system charge (or "excess electrons") used in the original grand-canonical simulation.
The second neural network, inspired by Goedecker's work, takes as input the atomic positions and predicts environment-dependent electronegativities for each atom.
Finally, this information is combined in a second-order charge--electronegativity expansion to predict the system energy, which is transformed into its grand-canonical form for comparison with the DFT results.

In this way, a trained model takes from the user the atomic positions and desired potential, and outputs both the grand-canonical energy and the required excess electrons, identical to a good grand-canonical electronic structure routine such as SJM.

Use
---

To train in the grand-canonical ensemble, one needs the output of electronic structure calculations in the same ensemble.
That is, each atomic image in the trajectory should not only have the atomic positions and the (grand-canonical) energy, but also the excess electrons as well as the resulting work function.
These latter quantities show up in ASE trajectories under ``atoms.calc.results``, and are placed their automatically if you are using the SJM calculator in GPAW.
If you are using a different grand-canonical code, you may have to hack your trajectory files to have the correct information.

To use this, we require a special version of ASE that does not restrict what information is saved in ``atoms.calc.results``.
We hope that this will be merged into the official version of ASE soon, but in the meantime we offer instructions to install the correct version of ASE for this purpose in the Installation notes.

An example trajectory that has all of the correct information can be found in the Amp distribution, under `tests/chargeNN_test/trainingset-charge.traj`.
Training a model to this trajectory is as simple as:

.. code-block:: python

    from ase.io import Trajectory

    from amp import Amp
    from amp.descriptor.gaussian import Gaussian
    from amp.model.chargeneuralnetwork import ChargeNeuralNetwork

    traj = Trajectory('trainingset-charge.traj')
    calc = Amp(descriptor=Gaussian(),
               model=ChargeNeuralNetwork(slab_metal='Au'),
               label='gcml')
    calc.train(images=traj)


You can open and use the trained calculator as normal:

.. code-block:: python

    from amp import Amp
    calc = Amp.load('gcml.amp')

Please look at the docstrings for ``ChargeNeuralNetwork`` for full details on all the parameters avaiable to tune the training process.
