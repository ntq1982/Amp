.. _Nearsightedforcetraining:


==========================
Nearsighted force training
==========================

We have published a paper on the nearsighted force-training (NFT) approach, in which we used an ensemble-based atomic uncertainty metric to systematically generate small structures to address uncertain local chemical environments:


    Zeng, Chen, and Peterson, "A nearsighted force-training approach to systematically generate training data for the machine learning of large atomic structures ". *JCP* 156, 064104 (2022). |nft_paper|


.. |nft_paper| raw:: html

    <a href="https://doi.org/10.1063/5.0079314" target="_blank">DOI:10.1063/5.0079314</a>

We introduce a module to train an ensemle of bootstrap calculators in an active learning scheme, which aims to address uncertain local chemical environments in a large structure iteratively.
We first train bootstrap calculators on an initial training set comprising of simple bulk structures.
Next, we quantify atomic uncertainties on a large target structure, as the standard deviation of force predictions of the bootstrap calculators multiplied by a constant coefficient.
We extract atomic "chunks" centered on the most uncertain atoms, and evaluated those "chunks" by single point calculations.
We then extend the training set by the calculated "chunks", and we retrain the bootstrap calculators until a certain stopping criterion is satisfied.
For the retraining with atomic "chunks", it is crucial that only the forces on central atoms are trained, which is the reason why this approach is termed as "nearsighted force training".

Automatic protocol
------------------

The example script at below shows how to train bootstrap calculators based on the nearsighted force-training automatic protocol.

.. code-block:: python

    from amp.nft.activelearner import NFT
    from amp.utilities import Logger

    calc_text = """
    from amp import Amp
    from amp.model import LossFunction
    from amp.descriptor.gaussian import Gaussian
    from amp.model.neuralnetwork import NeuralNetwork

    hl = [5, 5]
    calc = Amp(model=NeuralNetwork(hiddenlayers=hl),
               descriptor=Gaussian(),
               dblabel='../amp-data')
    calc.model.lossfunction = LossFunction(convergence={'energy_rmse': 0.001,
                                                         'force_rmse':0.005,
                                                         'force_maxresid': 0.02})
    """
    al = NFT(stop_delta=0.02, max_iterations=20, steps_not_improved=2,
             threshold=-0.9)

    traj = 'initial_images.traj'
    target_image = 'pt260.traj'
    start_command = 'python run.py'

    al.run(images=traj, target_image=target_image, n=10,
           calc_text=calc_text, start_command=start_command,
           parent_calc=EMT(), cutoff=6.5)

Once the active learning is stopped, the bootstrap calculators giving the best results will be saved as `best.[label].ensemble`.
The intermediate results will be saved inside the training folder in a folder named "saved-info", which includes the trajectory and indices of selected atomic chunks , and atomic uncertainties of the target structure at each NFT iteration
Indices and atomic uncertainties are saved in the **ndarray** format.

The active learning will be terminated if either condition at below is met---those conditions are supplied as parameters for the :py:class:`~amp.nft.activelearner.NFT` class.

- `stop_delta` controls the convergence structure uncertainty (maximum atomic uncertainty in the target structure).
- `max_iterations` controls the maximum allowed number of NFT iterations.
- `steps_not_improved` defines the number of consecutive NFT iterations to stop the NFT procedure if the structure uncertainty has not been improved.

The `threshold` controls the number of atomic "chunks" extracted from the target structure to be evaluated in single-point calculations.
For example, `threshold=-0.9` indicates that "chunks" with the top 10\% atomic uncertainties will be calculated in electronic structures.

Calling :py:meth:`~amp.nft.activelearner.NFT.run` method will spawn many independent training jobs, here `n=10` jobs.
Details of each job is given in the `calc_text`, and the jobs are submitted with the `start_command`.
For details about those two parameters, we refer the readers to the documentation regarding the `Bootstrap statistics <https://amp.readthedocs.io/en/latest/bootstrap.html>`__.
The initial bootstrap calculators are trained on `initial_images.traj`, and the uncertainty evaluation is targeted on `pt260.traj`.
The `parent_calc` is the electronic structure method to perform single-point calculations on atomic "chunks".
The `cutoff` controls the range of atoms to be included in an atomic "chunk".
