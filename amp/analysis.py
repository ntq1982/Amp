#!/usr/bin/env python

import os
import numpy as np
from ase.io import Trajectory

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

from . import Amp
from .utilities import now, hash_images, make_filename, hash_with_potential


rcParams.update({'figure.autolayout': True})


def plot_sensitivity(calc,
                     images,
                     d=0.0001,
                     label='sensitivity',
                     dblabel=None,
                     plotfile=None,
                     overwrite=False,
                     energy_coefficient=1.0,
                     force_coefficient=0.04,
                     charge_coefficient=10.0,
                     ):
    """Returns the plot of loss function in terms of perturbed parameters.

    Takes the load file and images. Any other keyword taken by the Amp
    calculator can be fed to this class also.

    Parameters
    ----------
    calc : Amp object or str
        Either an existing instantiated Amp calculator or a path for loading an
        existing ".amp" file. In the latter case, should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    d : float
        The amount of perturbation in each parameter.
    label : str
        Default prefix/location used for all files.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    plotfile : Object
        File for the plot.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    charge_coefficient : float
        Coefficient of charge loss in the total loss function.
    """

    from amp.model import LossFunction

    if isinstance(calc, str):
        calc = Amp.load(file=calc)

    if plotfile is None:
        plotfile = make_filename(label, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.' % plotfile)

    calc.dblabel = label if dblabel is None else dblabel

    if force_coefficient == 0.:
        calculate_derivatives = False
    else:
        calculate_derivatives = True

    calc._log('\nAmp sensitivity analysis started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':
        images = hash_with_potential(images)
    else:
        images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=calculate_derivatives)

    vector = calc.model.vector.copy()

    lossfunction = LossFunction(energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                charge_coefficient=charge_coefficient,
                                parallel=calc._parallel,
                                )
    calc.model.lossfunction = lossfunction

    # Set up local loss function.
    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':

        _ = calc.model.calculate_charge_fp_append(
            images,
            calc.model.parameters.slab_metal,
            calc.model.parameters.surface_correction,
            calc.model.parameters.etas)
        charge_fp_append, charge_fpprime_append = _
        calc.model.lossfunction.attach_model(
            calc.model,
            log=calc._log,
            fingerprints=calc.descriptor.fingerprints,
            fingerprintprimes=calc.descriptor.fingerprintprimes,
            charge_fp_appends=charge_fp_append,
            charge_fpprime_appends=charge_fpprime_append,
            images=images)
    else:
        calc.model.lossfunction.attach_model(
            calc.model,
            log=calc._log,
            fingerprints=calc.descriptor.fingerprints,
            fingerprintprimes=calc.descriptor.fingerprintprimes,
            images=images)

    originalloss = calc.model.lossfunction.get_loss(
        vector, lossprime=False)['loss']

    calc._log('\n Perturbing parameters...', tic='perturb')

    allparameters = []
    alllosses = []
    num_parameters = len(vector)

    for count in range(num_parameters):
        calc._log('parameter %i out of %i' % (count + 1, num_parameters))
        parameters = []
        losses = []
        # parameter is perturbed -d and loss function calculated.
        vector[count] -= d
        parameters.append(vector[count])
        perturbedloss = calc.model.lossfunction.get_loss(
            vector, lossprime=False)['loss']
        losses.append(perturbedloss)

        vector[count] += d
        parameters.append(vector[count])
        losses.append(originalloss)
        # parameter is perturbed +d and loss function calculated.
        vector[count] += d
        parameters.append(vector[count])
        perturbedloss = calc.model.lossfunction.get_loss(
            vector, lossprime=False)['loss']
        losses.append(perturbedloss)

        allparameters.append(parameters)
        alllosses.append(losses)
        # returning back to the original value.
        vector[count] -= d

    calc._log('...parameters perturbed and loss functions calculated',
              toc='perturb')

    calc._log('Plotting loss function vs perturbed parameters...',
              tic='plot')

    with PdfPages(plotfile) as pdf:
        count = 0
        for parameter in vector:
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            ax.plot(allparameters[count],
                    alllosses[count],
                    marker='o', linestyle='--', color='b',)

            xmin = allparameters[count][0] - \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            xmax = allparameters[count][-1] + \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            ymin = min(alllosses[count]) - \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ymax = max(alllosses[count]) + \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

            ax.set_xlabel('parameter no %i' % count)
            ax.set_ylabel('loss function')
            pdf.savefig(fig)
            pyplot.close(fig)
            count += 1

    calc._log(' ...loss functions plotted.', toc='plot')


def plot_parity_and_error(calc,
                          images,
                          label_parity='parity',
                          label_error='error',
                          dblabel=None,
                          xtic_angle=45.,
                          plot_forces=True,
                          plot_charges=False,
                          plotfile_parity=None,
                          plotfile_error=None,
                          color='b.',
                          overwrite=False,
                          returndata=False,
                          ):
    """Makes a parity plot and an error plot of Amp energies and forces versus
    real energies and forces.

    Parameters
    ----------
    calc : Amp object or str
        Either an existing instantiated Amp calculator or a path for loading an
        existing ".amp" file. In the latter case, should be fed like
        'load="filename.amp"'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    label_parity : str
        Default prefix/location used for the parity plot.
    label_error : str
        Default prefix/location used for the error plot.
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    xtic_angle : float
        Set the xtics angles. Default is 45.
    plot_forces : bool
        Determines whether or not forces should be plotted as well.
    plot_charges : bool
        Determines whether or not charges should be plotted as well.
    plotfile_parity : Object
        File for the parity plot.
    plotfile_error : Object
        File for the error plot.
    color : str
        Plot color.
    overwrite : bool
        If a plot or an script containing values found overwrite it.
    returndata : bool
        Whether to return a reference to the figures and their data or not.
    """

    if isinstance(calc, str):
        calc = Amp.load(file=calc, dblabel=dblabel)

    if plotfile_parity is None:
        plotfile_parity = make_filename(label_parity, '-plot.pdf')
    if plotfile_error is None:
        plotfile_error = make_filename(label_error, '-plot.pdf')

    if (not overwrite) and os.path.exists(plotfile_parity):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile_parity)
    if (not overwrite) and os.path.exists(plotfile_error):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile_error)

    if plot_forces is True:
        calculate_derivatives = True
    else:
        calculate_derivatives = False

    calc._log('\nAmp parity and error plots started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':
        images = hash_with_potential(images)
    else:
        images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=calculate_derivatives)

    calc._log('Calculating potential energies...', tic='pot-energy')
    energy_data = {}
    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':
        calc._log('Calculating charges...', tic='total-charge')
        charge_data = {}
        _ = calc.model.calculate_charge_fp_append(
            images,
            calc.model.parameters.slab_metal,
            calc.model.parameters.surface_correction,
            calc.model.parameters.etas)
        charge_fp_append, charge_fpprime_append = _
    for hash, image in images.items():
        no_of_atoms = len(image)

        model_name = calc.model.__class__.__name__

        if model_name == 'ChargeNeuralNetwork':
            energy_args = dict(
                    fingerprints=calc.descriptor.fingerprints[hash],
                    wf=images[hash].calc.results['electrode_potential'],
                    qfp_append=charge_fp_append[hash])

            amp_energy, amp_charge = calc.model.calculate_gc_energy(
                    **energy_args)
            amp_charge *= -1.
            actual_energy = image.get_potential_energy(apply_constraint=False)
            act_energy_per_atom = actual_energy / no_of_atoms
            energy_error = abs(amp_energy - actual_energy) / no_of_atoms
            energy_data[hash] = [actual_energy, amp_energy,
                                 act_energy_per_atom, energy_error]

            try:
                actual_charge = image.calc.parameters.sj['excess_electrons']
            except:
                actual_charge = image.calc.results['ne']
            act_charge_per_atom = actual_charge / no_of_atoms
            charge_error = abs(amp_charge - actual_charge) / no_of_atoms
            charge_data[hash] = [actual_charge, amp_charge,
                                 act_charge_per_atom, charge_error]
        else:
            energy_args = dict(
                    fingerprints=calc.descriptor.fingerprints[hash],
                    )

            if model_name == 'KernelRidge':
                if calc.model.trainingimages is not None:
                    trainingimages = hash_images(
                        Trajectory(calc.model.trainingimages))
                    energy_args['trainingimages'] = trainingimages
                    calc.descriptor.calculate_fingerprints(
                            images=trainingimages,
                            parallel=calc._parallel,
                            log=calc._log,
                            calculate_derivatives=calculate_derivatives
                            )
                    fp_trainingimages = calc.descriptor.fingerprints
                    energy_args['fp_trainingimages'] = fp_trainingimages
                    energy_args['hash'] = hash

            amp_energy = calc.model.calculate_energy(**energy_args)
            actual_energy = image.get_potential_energy(apply_constraint=False)
            act_energy_per_atom = actual_energy / no_of_atoms
            energy_error = abs(amp_energy - actual_energy) / no_of_atoms
            energy_data[hash] = [actual_energy, amp_energy,
                                 act_energy_per_atom, energy_error]

    calc._log('...potential energies calculated.', toc='pot-energy')
    # calculating minimum and maximum energies
    min_act_energy = min([energy_data[hash][0]
                          for hash, image in images.items()])
    max_act_energy = max([energy_data[hash][0]
                          for hash, image in images.items()])
    min_act_energy_per_atom = min([energy_data[hash][2]
                                   for hash, image in images.items()])
    max_act_energy_per_atom = max([energy_data[hash][2]
                                   for hash, image in images.items()])

    # calculating energy per atom rmse
    energy_square_error = 0.
    for hash, image in images.items():
        energy_square_error += energy_data[hash][3] ** 2.
    energy_per_atom_rmse = np.sqrt(energy_square_error / len(images))

    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':
        # calculating charge per atom rmse if using training-charge scheme
        calc._log('...charge calculated.', toc='total-charge')
        min_act_charge = min([charge_data[hash][0]
                              for hash, image in images.items()])
        max_act_charge = max([charge_data[hash][0]
                              for hash, image in images.items()])
        min_act_charge_per_atom = min([charge_data[hash][2]
                                       for hash, image in images.items()])
        max_act_charge_per_atom = max([charge_data[hash][2]
                                       for hash, image in images.items()])

        charge_square_error = 0.
        for hash, image in images.items():
            charge_square_error += charge_data[hash][3] ** 2.
        charge_per_atom_rmse = np.sqrt(charge_square_error / len(images))

    if plot_forces is True:
        calc._log('Calculating forces...', tic='forces')
        force_data = {}
        for hash, image in images.items():

            if model_name == 'ChargeNeuralNetwork':

                forces_args = dict(
                    fingerprints=calc.descriptor.fingerprints[hash],
                    fingerprintprimes=calc.descriptor.fingerprintprimes[hash],
                    wf=images[hash].calc.results['electrode_potential'],
                    qfp_append=charge_fp_append[hash],
                    qfpprime_append=charge_fpprime_append[hash]
                    )
                amp_forces = calc.model.calculate_gc_forces(**forces_args)
            else:
                forces_args = dict(
                    fingerprints=calc.descriptor.fingerprints[hash],
                    fingerprintprimes=calc.descriptor.fingerprintprimes[hash]
                    )

                if model_name == 'KernelRidge':
                    if calc.model.trainingimages is not None:
                        trainingimages = \
                            hash_images(Trajectory(calc.model.trainingimages))
                        calc.descriptor.calculate_fingerprints(
                            images=trainingimages,
                            calculate_derivatives=True
                            )
                        t_descriptor = calc.descriptor
                        forces_args['trainingimages'] = trainingimages
                        forces_args['t_descriptor'] = t_descriptor

                amp_forces = calc.model.calculate_forces(**forces_args)
            actual_forces = image.get_forces(apply_constraint=False)
            force_data[hash] = [actual_forces, amp_forces,
                                abs(np.array(amp_forces) -
                                    np.array(actual_forces))]
        calc._log('...forces calculated.', toc='forces')

        min_act_force = min([force_data[hash][0][index][k]
                             for hash, image in images.items()
                             for index in range(len(image))
                             for k in range(3)])

        max_act_force = max([force_data[hash][0][index][k]
                             for hash, image in images.items()
                             for index in range(len(image))
                             for k in range(3)])

        # calculating force rmse
        force_square_error = 0.
        for hash, image in images.items():
            no_of_atoms = len(image)
            for index in range(no_of_atoms):
                for k in range(3):
                    force_square_error += \
                        ((1.0 / 3.0) * force_data[hash][2][index][k] ** 2.) / \
                        no_of_atoms
        force_rmse = np.sqrt(force_square_error / len(images))

    # make parity plot
    if plot_forces is False and plot_charges is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    elif plot_forces is True and plot_charges is True:
        fig = pyplot.figure(figsize=(5., 15.))
        ax = fig.add_subplot(311)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    calc._log('Plotting energy parities...', tic='energy-plot')
    ax.plot(list(zip(*np.vstack(energy_data.values())))[0],
            list(zip(*np.vstack(energy_data.values())))[1], color)
    # draw line
    ax.plot([min_act_energy, max_act_energy],
            [min_act_energy, max_act_energy],
            'r-',
            lw=0.3,)
    ax.set_xlabel("ab initio energy, eV")
    ax.set_ylabel("Amp energy, eV")
    ax.set_title("Energies")
    ax.tick_params(axis='x', rotation=xtic_angle)
    calc._log('...energies plotted.', toc='energy-plot')

    if plot_forces is True:
        if plot_charges:
            ax = fig.add_subplot(313)
        else:
            ax = fig.add_subplot(212)
        calc._log('Plotting forces...', tic='force-plot')
        ax.plot(np.hstack(force_data.values())[0].flatten(),
                np.hstack(force_data.values())[1].flatten(), color)
        # draw line
        ax.plot([min_act_force, max_act_force],
                [min_act_force, max_act_force],
                'r-',
                lw=0.3,)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("Amp force, eV/Ang")
        ax.set_title("Forces")
        ax.tick_params(axis='x', rotation=xtic_angle)
        calc._log('...forces plotted.', toc='force-plot')

    if plot_charges is True:
        if plot_forces:
            ax = fig.add_subplot(312)
        else:
            ax = fig.add_subplot(212)
        calc._log('Plotting charges...', tic='charge-plot')
        ax.plot(list(zip(*np.vstack(charge_data.values())))[0],
                list(zip(*np.vstack(charge_data.values())))[1], color)
        # draw line
        ax.plot([min_act_charge, max_act_charge],
                [min_act_charge, max_act_charge],
                'r-',
                lw=0.3,)
        ax.set_xlabel(r"ab initio $N_e$")
        ax.set_ylabel(r"Amp $N_e$")
        ax.set_title("Number of Excess electrons")
        ax.tick_params(axis='x', rotation=xtic_angle)
        calc._log('...charges plotted.', toc='charge-plot')

    fig.tight_layout()
    fig.savefig(plotfile_parity)
    pyplot.close(fig)

    # make error plot

    if plot_forces is False and plot_charges is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    elif plot_forces is True and plot_charges is True:
        fig = pyplot.figure(figsize=(5., 15.))
        ax = fig.add_subplot(311)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    calc._log('Plotting energy errors...', tic='energy-plot')
    ax.plot(list(zip(*np.vstack(energy_data.values())))[2],
            list(zip(*np.vstack(energy_data.values())))[3], color)
    # draw horizontal line for rmse
    ax.plot([min_act_energy_per_atom, max_act_energy_per_atom],
            [energy_per_atom_rmse, energy_per_atom_rmse],
            color='black', linestyle='dashed', lw=1,)
    ax.text(max_act_energy_per_atom,
            energy_per_atom_rmse,
            'energy rmse = %6.5f' % energy_per_atom_rmse,
            ha='right',
            va='bottom',
            color='black')
    ax.set_xlabel("ab initio energy (eV) per atom")
    ax.set_ylabel("$|$ab initio energy - Amp energy$|$ / number of atoms")
    ax.set_title("Energies")
    ax.tick_params(axis='x', rotation=xtic_angle)
    calc._log('...energy errors plotted.', toc='energy-plot')

    if plot_forces is True:
        if plot_charges:
            ax = fig.add_subplot(313)
        else:
            ax = fig.add_subplot(212)

        calc._log('Plotting force errors...', tic='force-plot')
        ax.plot(np.hstack(force_data.values())[0].flatten(),
                np.hstack(force_data.values())[2].flatten(), color)
        # draw horizontal line for rmse
        ax.plot([min_act_force, max_act_force],
                [force_rmse, force_rmse],
                color='black',
                linestyle='dashed',
                lw=1,)
        ax.text(max_act_force,
                force_rmse,
                'force rmse = %5.4f' % force_rmse,
                ha='right',
                va='bottom',
                color='black',)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("$|$ab initio force - Amp force$|$")
        ax.set_title("Forces")
        ax.tick_params(axis='x', rotation=xtic_angle)
        calc._log('...force errors plotted.', toc='force-plot')

    if plot_charges is True:
        if plot_forces:
            ax = fig.add_subplot(312)
        else:
            ax = fig.add_subplot(212)
        calc._log('Plotting charge errors...', tic='charge-plot')
        ax.plot(list(zip(*np.vstack(charge_data.values())))[2],
                list(zip(*np.vstack(charge_data.values())))[3], color)
        # draw horizontal line for rmse
        ax.plot([min_act_charge_per_atom, max_act_charge_per_atom],
                [charge_per_atom_rmse, charge_per_atom_rmse],
                color='black', linestyle='dashed', lw=1,)
        ax.text(max_act_charge_per_atom,
                charge_per_atom_rmse,
                'charge rmse = %6.5f' % charge_per_atom_rmse,
                ha='right',
                va='bottom',
                color='black')
        ax.set_xlabel(r"ab initio $N_e$")
        ax.set_ylabel(r"$|$ab initio $N_e$ - Amp $N_e |$")
        ax.set_title("Number of Excess electrons")
        ax.tick_params(axis='x', rotation=xtic_angle)
        calc._log('...charge errors plotted.', toc='charge-plot')
    fig.tight_layout()
    fig.savefig(plotfile_error)
    pyplot.close(fig)

    if returndata:
        if plot_forces is False:
            if plot_charges:
                return energy_data, charge_data
            else:
                return energy_data
        else:
            if plot_charges:
                return energy_data, charge_data, force_data
            else:
                return energy_data, force_data


def calc_rmse(calc_paths,
              images,
              cores=None,
              dblabel=None,
              energy_coefficient=1.0,
              force_coefficient=0.04,
              charge_coefficient=10.0,
              ):
    """Calculates energy and force RMSEs for a set of Amp calculators. All
    calculators must have the same descriptors and models.

    Parameters
    ----------
    calc_paths : list
        List of paths for loading existing ".amp" files.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This can also be the path to an ASE trajectory (.traj)
        or database (.db) file.  Energies can be obtained from any reference,
        e.g. DFT calculations.
    cores : int
        Can specify cores to use for parallel processing; if None, will
        determine from environment
    dblabel : str
        Optional separate prefix/location of database files, including
        fingerprints, fingerprint primes, and neighborlists, to avoid
        calculating them. If not supplied, just uses the value from label.
    energy_coefficient : float
        Coefficient of energy loss in the total loss function.
    force_coefficient : float
        Coefficient of force loss in the total loss function.
    charge_coefficient : float
        Coefficient of charge loss in the total loss function.
    """

    from amp.model import LossFunction

    calcs = []
    calc = Amp.load(file=calc_paths[0], cores=cores, dblabel=dblabel)
    calcs.append(calc)
    for i in range(1, len(calc_paths)):
        calcs.append(Amp.load(file=calc_paths[i], cores=cores,
                              dblabel=dblabel, logging=False))
        calc._log('Loaded file: %s' % calc_paths[i])

    if charge_coefficient == 0.:
        if force_coefficient == 0.:
            calculate_derivatives = False
            convergence = {'energy_rmse': 0.001}
        else:
            calculate_derivatives = True
            convergence = {'energy_rmse': 0.001, 'force_rmse': 0.01}
    else:
        if force_coefficient == 0.:
            calculate_derivatives = False
            convergence = {'energy_rmse': 0.001, 'charge_rmse': 0.001}
        else:
            calculate_derivatives = True
            convergence = {'energy_rmse': 0.001, 'charge_rmse': 0.001,
                           'force_rmse': 0.01}

    # Setting the convergence is a kludgy way to keep LossFunction.__init__()
    # from resetting the force_coefficient to 0

    calc._log('\nAmp calc_rmse started. ' + now() + '\n')
    calc._log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    calc._log('Model: %s' % calc.model.__class__.__name__)

    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':
        images = hash_with_potential(images)
    else:
        images = hash_images(images)

    calc._log('\nDescriptor\n==========')
    calc.descriptor.calculate_fingerprints(
        images=images,
        parallel=calc._parallel,
        log=calc._log,
        calculate_derivatives=calculate_derivatives)

    lossfunction = LossFunction(energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                charge_coefficient=charge_coefficient,
                                parallel=calc._parallel,
                                raise_ConvergenceOccurred=False,
                                convergence=convergence,
                                )
    calc.model.lossfunction = lossfunction

    # Set up local loss function.
    if calc.model.__class__.__name__ == 'ChargeNeuralNetwork':
        _ = calc.model.calculate_charge_fp_append(
            images,
            calc.model.parameters.slab_metal,
            calc.model.parameters.surface_correction,
            calc.model.parameters.etas)
        charge_fp_append, charge_fpprime_append = _
        if force_coefficient == 0.:
            calc.model.lossfunction.attach_model(
                calc.model,
                log=calc._log,
                fingerprints=calc.descriptor.fingerprints,
                charge_fp_appends=charge_fp_append,
                charge_fpprime_appends=charge_fpprime_append,
                images=images)
        else:
            calc.model.lossfunction.attach_model(
                calc.model,
                log=calc._log,
                fingerprints=calc.descriptor.fingerprints,
                fingerprintprimes=calc.descriptor.fingerprintprimes,
                charge_fp_appends=charge_fp_append,
                charge_fpprime_appends=charge_fpprime_append,
                images=images)

    else:
        if force_coefficient == 0.:
            calc.model.lossfunction.attach_model(
                calc.model,
                log=calc._log,
                fingerprints=calc.descriptor.fingerprints,
                images=images)
        else:
            calc.model.lossfunction.attach_model(
                calc.model,
                log=calc._log,
                fingerprints=calc.descriptor.fingerprints,
                fingerprintprimes=calc.descriptor.fingerprintprimes,
                images=images)

    steps = []
    loss = []
    energy_loss = []
    force_loss = []
    charge_loss = []
    energy_maxresid = []
    force_maxresid = []
    charge_maxresid = []
    energy_rmse = []
    force_rmse = []
    charge_rmse = []
    for i in range(len(calc_paths)):
        steps.append(int(os.path.basename(calc_paths[i])[:-4]))
        vector = calcs[i].model.vector.copy()
        results = calc.model.lossfunction.get_loss(
                      vector,
                      lossprime=calculate_derivatives)
        loss.append(results['loss'])
        energy_loss.append(results['energy_loss'])
        force_loss.append(results['force_loss'])
        charge_loss.append(results['charge_loss'])
        energy_maxresid.append(results['energy_maxresid'])
        force_maxresid.append(results['force_maxresid'])
        charge_maxresid.append(results['charge_maxresid'])
        energy_rmse.append(np.sqrt(energy_loss[i] / len(images)))
        if charge_coefficient == 0.:
            if force_coefficient == 0.:
                calc._log('%5i %19s %12.4e   %10.4e   %10.4e' %
                          (steps[i], now(), loss[i], energy_rmse[i],
                           energy_maxresid[i]))
            else:
                force_rmse.append(np.sqrt(force_loss[i] / len(images)))
                calc._log('%5i %19s %12.4e   %10.4e  '
                          ' %10.4e   %10.4e   %10.4e' %
                          (steps[i], now(), loss[i], energy_rmse[i],
                           energy_maxresid[i], force_rmse[i],
                           force_maxresid[i]))
        else:
            charge_rmse.append(np.sqrt(charge_loss[i] / len(images)))
            if force_coefficient == 0.:
                calc._log(
                    '%5i %19s %12.4e   %10.4e   %10.4e   %10.4e   %10.4e' %
                    (steps[i], now(), loss[i], energy_rmse[i],
                     energy_maxresid[i], charge_rmse[i], charge_maxresid[i]))
            else:
                force_rmse.append(np.sqrt(force_loss[i] / len(images)))
                calc._log('%5i %19s %12.4e   %10.4e  '
                          ' %10.4e   %10.4e   %10.4e  '
                          ' %10.4e   %10.4e' %
                          (steps[i], now(), loss[i], energy_rmse[i],
                           energy_maxresid[i], charge_rmse[i],
                           charge_maxresid[i], force_rmse[i],
                           force_maxresid[i]))

    data = {}
    data['steps'] = steps
    data['loss'] = loss
    data['energy_loss'] = energy_loss
    data['force_loss'] = force_loss
    data['charge_loss'] = charge_loss
    data['energy_maxresid'] = energy_maxresid
    data['force_maxresid'] = force_maxresid
    data['charge_maxresid'] = charge_maxresid
    data['energy_rmse'] = energy_rmse
    if force_coefficient != 0.:
        data['force_rmse'] = force_rmse
    if charge_coefficient != 0.:
        data['charge_rmse'] = charge_rmse
    return data


def read_trainlog(logfile, verbose=True, multiple=0):
    """Reads the log file from the training process, returning the relevant
    parameters.

    Parameters
    ----------
    logfile : str
        Name or path to the log file.

    verbose : bool
        Write out logfile during analysis.

    multiple : int or True
        If multiple training sessions are recorded in the same log file,
        return session number <multiple> (counting from 0). If set to True,
        returns all sessions as list.
    """
    data = {}

    with open(logfile, 'r') as f:
        lines = f.read().splitlines()

    # Get number of training sets.
    multiple_starts = []
    for index, line in enumerate(lines):
        if line.startswith('Amp training started.'):
            multiple_starts.append(index)
    if multiple is True:
        datalist = []
        for index in range(len(multiple_starts)):
            datalist.append(read_trainlog(logfile, verbose,
                                          multiple=index))
        return datalist
    else:
        lines = lines[multiple_starts[multiple]:]

    def print_(text):
        if verbose:
            print(text)

    # Get number of images.
    for line in lines:
        if 'unique images after hashing' in line:
            no_images = int(line.split()[0])
            break
    data['no_images'] = no_images

    # Find where convergence data starts.
    startline = None
    for index, line in enumerate(lines):
        if 'Loss function convergence criteria:' in line:
            startline = index
            data['convergence'] = {}
            d = data['convergence']
            break
    else:
        return data

    # Get convergence parameters.
    ready = [False] * 10
    for index, line in enumerate(lines[startline:]):
        if 'energy_rmse:' in line:
            ready[0] = True
            d['energy_rmse'] = float(line.split(':')[-1])
        elif 'force_rmse:' in line:
            ready[1] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_rmse'] = None
                trainforces = False
            else:
                d['force_rmse'] = float(line.split(':')[-1])
                trainforces = True
            print_('train forces: %s' % trainforces)
        elif 'charge_rmse:' in line:
            ready[7] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['charge_rmse'] = None
                traincharges = False
            else:
                d['charge_rmse'] = float(line.split(':')[-1])
                traincharges = True
            print_('train charges: %s' % traincharges)
        elif 'force_coefficient:' in line:
            ready[2] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_coefficient'] = 0.
            else:
                d['force_coefficient'] = float(_)
        elif 'charge_coefficient:' in line:
            ready[8] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['charge_coefficient'] = 0.
            else:
                d['charge_coefficient'] = float(_)
        elif 'energy_coefficient:' in line:
            ready[3] = True
            d['energy_coefficient'] = float(line.split(':')[-1])
        elif 'energy_maxresid:' in line:
            ready[5] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['energy_maxresid'] = None
            else:
                d['energy_maxresid'] = float(_)
        elif 'force_maxresid:' in line:
            ready[6] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['force_maxresid'] = None
            else:
                d['force_maxresid'] = float(_)
        elif 'charge_maxresid:' in line:
            ready[9] = True
            _ = line.split(':')[-1].strip()
            if _ == 'None':
                d['charge_maxresid'] = None
            else:
                d['charge_maxresid'] = float(_)
        elif 'Step' in line and 'Time' in line:
            ready[4] = True
            startline += index + 2
        if ready == [True] * 7:
            break

    for _ in d.items():
        print_('{}: {}'.format(_[0], _[1]))
    E = d['energy_rmse']**2 * no_images
    if trainforces:
        F = d['force_rmse']**2 * no_images
    else:
        F = 0.
    if traincharges:
        Q = d['charge_rmse']**2 * no_images
    else:
        Q = 0.
    costfxngoal = d['energy_coefficient'] * E + d['force_coefficient'] * F + \
        d['charge_coefficient'] * Q
    d['costfxngoal'] = costfxngoal

    # Extract data (emrs and fmrs are max residuals).
    steps = []
    es = []
    qs = []
    fs = []
    emrs = []
    qmrs = []
    fmrs = []
    costfxns = []
    costfxnEs = []
    costfxnQs = []
    costfxnFs = []
    index = startline
    d['converged'] = None
    while index < len(lines):
        line = lines[index]
        if 'Saving checkpoint data.' in line:
            index += 1
            continue
        elif 'Overwriting file' in line:
            index += 1
            continue
        elif 'optimization completed successfully.' in line:  # old version
            d['converged'] = True
            break
        elif '...optimization successful.' in line:
            d['converged'] = True
            break
        elif 'could not find parameters for the' in line:
            break
        elif '...optimization unsuccessful.' in line:
            d['converged'] = False
            break
        elif len(line) == 0:
            # Job apparently timed out.
            break
        print_(line)
        if traincharges:
            if trainforces:
                (step, time, costfxn, e, _, q, _,
                 emr, _, qmr, _, f, _, fmr, _) = line.split()
                fs.append(float(f))
                fmrs.append(float(fmr))
                F = float(f)**2 * no_images
                costfxnFs.append(d['force_coefficient'] * F / float(costfxn))
            else:
                step, time, costfxn, e, _, q, _, emr, _, qmr, _ = line.split()
            qs.append(float(q))
            qmrs.append(float(qmr))
            Q = float(q) ** 2. * no_images
            costfxnQs.append(d['charge_coefficient'] * Q / float(costfxn))
        else:
            if trainforces:
                step, time, costfxn, e, _, emr, _, f, _, fmr, _ = line.split()
                fs.append(float(f))
                fmrs.append(float(fmr))
                F = float(f)**2 * no_images
                costfxnFs.append(d['force_coefficient'] * F / float(costfxn))
            else:
                step, time, costfxn, e, _, emr, _ = line.split()
        steps.append(int(step))
        es.append(float(e))
        emrs.append(float(emr))
        costfxns.append(float(costfxn))
        E = float(e)**2 * no_images
        costfxnEs.append(d['energy_coefficient'] * E / float(costfxn))
        index += 1
    d['steps'] = steps
    d['es'] = es
    d['qs'] = qs
    d['fs'] = fs
    d['emrs'] = emrs
    d['qmrs'] = qmrs
    d['fmrs'] = fmrs
    d['costfxns'] = costfxns
    d['costfxnEs'] = costfxnEs
    d['costfxnFs'] = costfxnFs
    d['costfxnQs'] = costfxnQs

    return data


def plot_convergence(data, plotfile='convergence.pdf'):
    """Makes a plot of the convergence of the cost function and its energy
    and force components.

    Parameters
    ----------
    data : dict
        Convergence data dictionary as returned by read_trainlog.
    plotfile : str or None
        Name or path to the plot file. If None, instead returns reference to
        the created figure.
    """

    # Find if multiple runs contained in data set.
    d = data['convergence']
    steps = range(len(d['steps']))
    breaks = []
    for index, step in enumerate(d['steps'][1:]):
        if step < d['steps'][index]:
            breaks.append(index)

    # Make plots.
    fig = pyplot.figure(figsize=(6., 8.))
    # Margins, vertical gap, and top-to-bottom ratio of figure.
    lm, rm, bm, tm, vg, tb = 0.12, 0.05, 0.08, 0.03, 0.08, 4.
    bottomaxheight = (1. - bm - tm - vg) / (tb + 1.)

    ax = fig.add_axes((lm, bm + bottomaxheight + vg,
                       1. - lm - rm, tb * bottomaxheight))
    ax.semilogy(steps, d['es'], 'b', lw=2, label='energy rmse')
    ax.semilogy(steps, d['emrs'], 'b:', lw=2, label='energy maxresid')
    if d['force_rmse']:
        ax.semilogy(steps, d['fs'], 'g', lw=2, label='force rmse')
        ax.semilogy(steps, d['fmrs'], 'g:', lw=2, label='force maxresid')
    if d['charge_rmse']:
        ax.semilogy(steps, d['qs'], 'r', lw=2, label='charge rmse')
        ax.semilogy(steps, d['qmrs'], 'r:', lw=2, label='charge maxresid')
    ax.semilogy(steps, d['costfxns'], color='0.5', lw=2,
                label='loss function')
    # Targets.
    if d['energy_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['energy_rmse']] * 2,
                    color='b', linestyle='-', alpha=0.5)
    if d['energy_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['energy_maxresid']] * 2,
                    color='b', linestyle=':', alpha=0.5)
    if d['charge_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['charge_rmse']] * 2,
                    color='r', linestyle='-', alpha=0.5)
    if d['charge_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['charge_maxresid']] * 2,
                    color='r', linestyle=':', alpha=0.5)
    if d['force_rmse']:
        ax.semilogy([steps[0], steps[-1]], [d['force_rmse']] * 2,
                    color='g', linestyle='-', alpha=0.5)
    if d['force_maxresid']:
        ax.semilogy([steps[0], steps[-1]], [d['force_maxresid']] * 2,
                    color='g', linestyle=':', alpha=0.5)
    ax.set_ylabel('error')
    ax.legend(loc='best', fontsize=9.)
    if len(breaks) > 0:
        ylim = ax.get_ylim()
        for b in breaks:
            ax.plot([b] * 2, ylim, '--k')

    if d['charge_rmse']:
        if d['force_rmse']:
            # Loss function component plot.
            axf = fig.add_axes((lm, bm, 1. - lm - rm, bottomaxheight))
            axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                             color='blue')
            axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                             y2=np.array(d['costfxnEs']) +
                             np.array(d['costfxnQs']),
                             color='red')
            axf.fill_between(x=np.array(steps), y1=np.array(d['costfxnEs']) +
                             np.array(d['costfxnQs']),
                             y2=np.array(d['costfxnEs']) +
                             np.array(d['costfxnQs']) +
                             np.array(d['costfxnFs']),
                             color='green')
            axf.set_ylabel('loss function component')
            axf.set_xlabel('loss function call')
            axf.set_ylim(0, 1)
        else:
            ax.set_xlabel('loss function call')
    else:
        if d['force_rmse']:
            # Loss function component plot.
            axf = fig.add_axes((lm, bm, 1. - lm - rm, bottomaxheight))
            axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                             color='blue')
            axf.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                             y2=np.array(d['costfxnEs']) +
                             np.array(d['costfxnFs']),
                             color='green')
            axf.set_ylabel('loss function component')
            axf.set_xlabel('loss function call')
            axf.set_ylim(0, 1)
        else:
            ax.set_xlabel('loss function call')

    if plotfile is None:
        return fig
    fig.savefig(plotfile)
    pyplot.close(fig)
