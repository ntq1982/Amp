#!/usr/bin/env python
"""Simple test of the Amp calculator, using Gaussian descriptors and neural
network model. Randomly generates data with the EMT potential in MD
simulations."""

import sys
from ase.calculators.emt import EMT
from ase.build import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian


def check_perform():
    """Determines whether or not to perform the test.
    This should only perform the test if the python version is 2.x
    and tensorflow is installed. If returns False (meaning don't
    peform test), also supplies the reason."""
    if sys.version_info >= (3,):
        return False, 'amp.model.tflow not supported in python3.'
    try:
        import tensorflow
    except ImportError:
        return False, 'Tensorflow not installed.'
    return True, ''


def generate_data(count):
    """Generates test or training data with a simple MD simulation."""
    atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperautre_K=300.)
    dyn = VelocityVerlet(atoms, timestep=1. * units.fs)
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
    """Gaussian/tflow train test."""
    perform, reason = check_perform()
    if not perform:
        print('Skipping this test because {}.'.format(reason))
        return

    from amp.model.tflow import NeuralNetwork
    label = 'train_test/calc'
    train_images = generate_data(2)
    convergence = {
            'energy_rmse': 0.02,
            'force_rmse': 0.02
            }

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(hiddenlayers=(3, 3),
                                   convergenceCriteria=convergence),
               label=label,
               cores=1)

    calc.train(images=train_images,)
    for image in train_images:
        print("energy =", calc.get_potential_energy(image))
        print("forces =", calc.get_forces(image))


if __name__ == '__main__':
    train_test()
