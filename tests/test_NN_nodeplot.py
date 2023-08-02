import matplotlib
matplotlib.use('Agg')  # For headless operation.

from ase.calculators.emt import EMT
from ase.build import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.model.neuralnetwork import NodePlot
from amp.utilities import hash_images


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


def train_data(images, setup_only=False):
    label = 'nodeplot_test/calc'
    train_images = images

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(hiddenlayers=(5, 5)),
               label=label,
               cores=1)
    loss = LossFunction(convergence={'energy_rmse': 0.02,
                                     'force_rmse': 0.02})
    calc.model.lossfunction = loss

    if not setup_only:
        calc.train(images=train_images, )
        for image in train_images:
            print("energy =", calc.get_potential_energy(image))
            print("forces =", calc.get_forces(image))
    else:
        images = hash_images(train_images)
        calc.descriptor.calculate_fingerprints(images=images,
                                               log=calc._log,
                                               parallel={'cores': 1},
                                               calculate_derivatives=False)
        calc.model.fit(trainingimages=images,
                       descriptor=calc.descriptor,
                       log=calc._log,
                       parallel={'cores': 1},
                       only_setup=True)
        return calc


def test_nodeplot():
    """Nodeplot creation test."""
    images = generate_data(2)
    calc = train_data(images, setup_only=True)
    nodeplot = NodePlot(calc)
    nodeplot.plot(images, filename='nodeplottest.pdf')


if __name__ == "__main__":
    test_nodeplot()
