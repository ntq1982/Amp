"""Simple test of the Amp calculator, using Gaussian descriptors and neural
network model. Randomly generates data with the EMT potential in MD
simulations."""

import matplotlib
# The 'Agg' command must be *before* all other matplotlib imports for
# headless operation.
matplotlib.use('Agg')

import os
from ase import Atoms, Atom, units
import ase.io
from ase.calculators.emt import EMT
from ase.build import fcc110
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.descriptor.analysis import FingerprintPlot


def generate_data(count, filename='training.traj'):
    """Generates test or training data with a simple MD simulation."""
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
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.)
    dyn = VelocityVerlet(atoms, timestep=1. * units.fs)
    for step in range(count - 1):
        dyn.run(50)
        traj.write(atoms)


def test():
    "FingerprintPlot test."""
    generate_data(2, filename='fpplot-training.traj')

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(),
               label='fpplot-test'
               )
    calc.model.lossfunction = LossFunction(convergence={'energy_rmse': 1.00,
                                                        'force_rmse': 1.00})
    calc.train(images='fpplot-training.traj')

    images = ase.io.Trajectory('fpplot-training.traj')
    fpplot = FingerprintPlot(calc)
    fpplot(images)
    fpplot(images, overlay=images[0])
    fpplot(images, overlay=[images[1][2], images[0][-1]])

if __name__ == '__main__':
    test()
