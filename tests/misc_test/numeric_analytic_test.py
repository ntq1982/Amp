#!/usr/bin/env python
"""This test randomly generates data with the EMT potential in MD simulations,
and then checks for consistency between analytical and numerical forces,
as well as dloss_dparameters."""

from ase.calculators.emt import EMT
from ase.build import fcc110

from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.regression import Regressor


def generate_data():
    """Generates atomic images for tests."""
    atoms = fcc110('Pt', (2, 2, 1), vacuum=7.)
    atoms[0].symbol = 'Cu'
    del atoms[3]
    atoms.calc = EMT()
    atoms.get_potential_energy()
    atoms.get_forces()
    newatoms = atoms.copy()
    newatoms.calc = EMT()
    newatoms[0].position += (0.27, -0.11, 0.3)
    newatoms[1].position += (0.12, 0.03, -0.22)
    newatoms.get_potential_energy()
    newatoms.get_forces()
    return [atoms, newatoms]


def test():
    """Gaussian/Neural numeric-analytic consistency."""
    images = generate_data()
    regressor = Regressor(optimizer='BFGS')

    _G = make_symmetry_functions(type='G2', etas=[0.05, 5.],
                                 elements=['Cu', 'Pt'])
    _G += make_symmetry_functions(type='G4', etas=[0.005],
                                  zetas=[1., 4.], gammas=[1.],
                                  elements=['Cu', 'Pt'])
    Gs = {'Cu': _G, 'Pt': _G}
    calc = Amp(descriptor=Gaussian(Gs=Gs),
               model=NeuralNetwork(hiddenlayers=(2, 1),
                                   regressor=regressor,
                                   randomseed=42,
                                   ),
               cores=1)

    step = 0
    for d in [None, 0.00001]:
        for fortran in [True, False]:
            for cores in [1, 2]:
                step += 1
                label = \
                    'numeric_analytic_test/analytic-%s-%i' % (fortran, cores) \
                    if d is None \
                    else 'numeric_analytic_test/numeric-%s-%i' \
                    % (fortran, cores)
                print(label)

                loss = LossFunction(convergence={'energy_rmse': 10 ** 10,
                                                 'force_rmse': 10 ** 10},
                                    d=d)
                calc.set_label(label)
                calc.dblabel = 'numeric_analytic_test/analytic-True-1'
                calc.model.lossfunction = loss
                calc.descriptor.fortran = fortran
                calc.model.fortran = fortran
                calc.cores = cores

                calc.train(images=images,)

                if step == 1:
                    ref_energies = []
                    ref_forces = []
                    for image in images:
                        ref_energies += [calc.get_potential_energy(image)]
                        ref_forces += [calc.get_forces(image)]
                        ref_dloss_dparameters = \
                            calc.model.lossfunction.dloss_dparameters
                else:
                    energies = []
                    forces = []
                    for image in images:
                        energies += [calc.get_potential_energy(image)]
                        forces += [calc.get_forces(image)]
                        dloss_dparameters = \
                            calc.model.lossfunction.dloss_dparameters

                    for image_no in range(2):

                        diff = abs(energies[image_no] - ref_energies[image_no])
                        assert (diff < 10.**(-13.)), \
                            'The calculated value of energy of image %i is ' \
                            'wrong!' % (image_no + 1)

                        for atom_no in range(len(images[0])):
                            for i in range(3):
                                diff = abs(forces[image_no][atom_no][i] -
                                           ref_forces[image_no][atom_no][i])
                                assert (diff < 10.**(-10.)), \
                                    'The calculated %i force of atom %i of ' \
                                    'image %i is wrong!' \
                                    % (i, atom_no, image_no + 1)
                        # Checks analytical and numerical dloss_dparameters
                        for _ in range(len(ref_dloss_dparameters)):
                            diff = abs(dloss_dparameters[_] -
                                       ref_dloss_dparameters[_])
                            assert(diff < 10 ** (-10.)), \
                                'The calculated value of loss function ' \
                                'derivative is wrong!'
    # Checks analytical and numerical forces
    forces = []
    for image in images:
        image.calc = calc
        forces += [calc.calculate_numerical_forces(image, d=d)]
    for atom_no in range(len(images[0])):
        for i in range(3):
            diff = abs(forces[image_no][atom_no][i] -
                       ref_forces[image_no][atom_no][i])
            print('{:3d} {:1d} {:7.1e}'.format(atom_no, i, diff))
            assert (diff < 10.**(-6.)), \
                'The calculated %i force of atom %i of ' \
                'image %i is wrong! (Diff = %f)' \
                % (i, atom_no, image_no + 1, diff)


if __name__ == '__main__':
    test()
