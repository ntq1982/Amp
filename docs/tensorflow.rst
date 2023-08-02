.. _TensorFlow:

==================================
TensorFlow
==================================

Google has released an open-source version of its machine-learning software named Tensorflow, which can allow for efficient backpropagation of neural networks and utilization of GPUs for extra speed.

We have incorporated an experimental module that uses a tensorflow back-end, which may provide an acceleration particularly through access to GPU systems.
As of this writing, the tensorflow code is in flux (with version 1.0 anticipated shortly).


Dependencies
---------------------------------

This package requires google's TensorFlow 0.11.0. You can install it as shown
below for Linux::

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
    pip install -U --upgrade $TF_BINARY_URL

or macOS::

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py2-none-any.whl
    pip install -U --upgrade $TF_BINARY_URL

If you want more information, please see `tensorflow's website <https://www.tensorflow.org/versions/r0.11/get_started/os_setup#pip_installation>`_ for instructions
for installation on your system.

Example
---------------------------------

.. code-block:: python

 #!/usr/bin/env python
 """Simple test of the Amp calculator, using Gaussian descriptors and neural
 network model. Randomly generates data with the EMT potential in MD
 simulations."""

 from ase.calculators.emt import EMT
 from ase.build import fcc110
 from ase import Atoms, Atom
 from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
 from ase import units
 from ase.md import VelocityVerlet
 from ase.constraints import FixAtoms

 from amp import Amp
 from amp.descriptor.gaussian import Gaussian
 from amp.model.tflow import NeuralNetwork


 def generate_data(count):
     """Generates test or training data with a simple MD simulation."""
     atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
     adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                        Atom('Cu', atoms[7].position + (0., 0., 5.))])
     atoms.extend(adsorbate)
     atoms.set_constraint(FixAtoms(indices=[0, 2]))
     atoms.calc = EMT()
     MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
     dyn = VelocityVerlet(atoms, dt=1. * units.fs)
     newatoms = atoms.copy()
     newatoms.calc = EMT()
     newatoms.get_potential_energy()
     images = [newatoms]
     for step in range(count - 1):
         dyn.run(50)
         newatoms = atoms.copy()
         newatoms.calc = EMT()
         newatoms.get_potential_energy()
         images.append(newatoms)
     return images


 def train_test():
     label = 'train_test/calc'
     train_images = generate_data(2)
     convergence = {
             'energy_rmse': 0.02,
             'force_rmse': 0.02
             }

     calc = Amp(descriptor=Gaussian(),
                model=NeuralNetwork(hiddenlayers=(3, 3), convergenceCriteria=convergence),
                label=label,
                cores=1)

     calc.train(images=train_images,)
     for image in train_images:
         print "energy =", calc.get_potential_energy(image)
         print "forces =", calc.get_forces(image)


 if __name__ == '__main__':
     train_test()

Known issues
---------------------------------
- `tflow` module does not work for versions different from 0.11.0.

About
---------------------------------

This module was contributed by Zachary Ulissi (Department of Chemical Engineering, Stanford University, zulissi@gmail.com) with help, testing, and discussions from Andrew Doyle (Stanford) and the Amp development team.
