#!/usr/bin/env python
"""Simple test of the Amp calculator, using Gaussian descriptors and ChargeNN
model. Trajectory from SJM prepared."""

from pathlib import Path

from ase.io import Trajectory

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.chargeneuralnetwork import ChargeNeuralNetwork

THIS_DIR = Path(__file__).parent


def train_test():
    """Gaussian/ChargeNN train test."""
    label = 'charge_train_test/calc'
    traj = Trajectory(THIS_DIR / 'trainingset-charge-short.traj')

    calc = Amp(descriptor=Gaussian(),
               model=ChargeNeuralNetwork(slab_metal='Au'),
               label=label,
               )

    calc.train(images=traj,)
    for image in traj:
        print("energy = %s" % str(calc.get_potential_energy(image)))
        print("charges = %s" % str(-sum(calc.get_charges(image))))

    # Test that we can re-load this calculator and call it again.
    del calc
    calc2 = Amp.load(label + '.amp')
    for image in traj:
        print("energy = %s" % str(calc2.get_potential_energy(image)))
        print("charges = %s" % str(-sum(calc2.get_charges(image))))


if __name__ == '__main__':
    train_test()
