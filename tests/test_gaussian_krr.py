#!/usr/bin/env python
"""Simple test of the Amp calculator, using Gaussian descriptors and Kernel
Ridge model. Randomly generates data with the EMT potential in MD
simulations."""

from ase.calculators.emt import EMT
from ase.build import fcc110
from ase import Atoms, Atom
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.kernelridge import KernelRidge


def generate_data(count):
    """Generates test or training data with a simple MD simulation."""
    atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.calc = EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.)
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
    """Gaussian/KRR train test."""
    label = 'train_test/calc'
    train_images = generate_data(2)
    traj = Trajectory('trainingset.traj', mode='w')

    for image in train_images:
        traj.write(image)

    calc = Amp(descriptor=Gaussian(),
               model=KernelRidge(forcetraining=True,
                                 trainingimages='trainingset.traj'),
               label=label,
               cores=1)

    calc.train(images=train_images,)
    for image in train_images:
        print("energy = %s" % str(calc.get_potential_energy(image)))
        print("forces = %s" % str(calc.get_forces(image)))

    # Test that we can re-load this calculator and call it again.
    del calc
    calc2 = Amp.load(label + '.amp')
    for image in train_images:
        print("energy = %s" % str(calc2.get_potential_energy(image)))
        print("forces = %s" % str(calc2.get_forces(image)))


if __name__ == '__main__':
    train_test()
