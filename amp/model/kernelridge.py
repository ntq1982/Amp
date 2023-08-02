#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
#
# Author: Muammar El Khatib <muammarelkhatib@brown.edu>

import threading
import time
import sys
import os
import numpy as np
from collections import OrderedDict
from scipy.linalg import cholesky

from ase.calculators.calculator import Parameters

from ..utilities import (make_filename, hash_images, Logger,
                         ConvergenceOccurred, make_sublists, now,
                         setup_parallel)

try:
    from .. import fmodules
except ImportError:
    fmodules = None


class Model(object):
    """Class that includes common methods between different models."""

    @property
    def log(self):
        """Method to set or get a logger. Should be an instance of
        amp.utilities.Logger.

        Parameters
        ----------
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        """
        if hasattr(self, '_log'):
            return self._log
        if hasattr(self.parent, 'log'):
            return self.parent.log
        return Logger(None)

    @log.setter
    def log(self, log):
        self._log = log

    def tostring(self):
        """Returns an evaluable representation of the calculator that can
        be used to re-establish the calculator."""
        # Make sure numpy prints out enough data.
        np.set_printoptions(precision=30, threshold=999999999)
        return self.parameters.tostring()

    def calculate_energy(self, fingerprints, hash=None, trainingimages=None,
                         fp_trainingimages=None):
        """Calculates the model-predicted energy for an image, based on its
        fingerprint.

        Parameters
        ----------
        fingerprints : dict or list
            Dictionary with images hashes as keys and the corresponding
            fingerprints as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            self.atomic_energies = []
            energy = 0.0

            if not isinstance(fingerprints, list):
                fingerprints = fingerprints[hash]

            if self.cholesky is False and self.nnpartition is None:
                for index, (symbol, afp) in enumerate(fingerprints):
                    arguments = dict(afp=afp, index=index, symbol=symbol)

                    if hash is not None:
                        arguments['hash'] = hash
                        arguments['fp_trainingimages'] = fp_trainingimages
                        arguments['kernel'] = self.parameters.kernel
                        arguments['sigma'] = self.parameters.sigma
                        arguments['trainingimages'] = trainingimages

                    atom_energy = self.calculate_atomic_energy(**arguments)
                    self.atomic_energies.append(atom_energy)
                    energy += atom_energy
            elif self.cholesky is True and self.nnpartition is not None:
                for index, (symbol, afp) in enumerate(fingerprints):
                    arguments = dict(symbol=symbol, afp=afp, hash=hash,
                                     fp_trainingimages=fp_trainingimages,
                                     kernel=self.parameters.kernel,
                                     trainingimages=trainingimages,
                                     sigma=self.parameters.sigma,
                                     fingerprints=fingerprints)

                    atom_energy = self.energy_from_cholesky(**arguments)
                    self.atomic_energies.append(atom_energy)
                    energy += atom_energy
            else:
                for index, (symbol, afp) in enumerate(fingerprints):
                    arguments = dict(symbol=symbol, afp=afp, hash=hash,
                                     fp_trainingimages=fp_trainingimages,
                                     kernel=self.parameters.kernel,
                                     trainingimages=trainingimages,
                                     sigma=self.parameters.sigma,
                                     fingerprints=fingerprints)

                    preprocessing = self.parameters.preprocessing
                    if preprocessing:
                        arguments['preprocessing'] = preprocessing

                    atom_energy = self.energy_from_cholesky(**arguments)
                    self.atomic_energies.append(atom_energy)
                    energy += atom_energy

        return energy

    def calculate_forces(self, fingerprints, fingerprintprimes, hash=None,
                         trainingimages=None, t_descriptor=None):
        """Calculates the model-predicted forces for an image, based on
        derivatives of fingerprints.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images' hashes as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprint derivatives as values.
        hash : str
            Image unique hash.
        trainingimages : dict
            Dictionary with training images.
        t_descriptor : object
            Object with training fingerprints and fingerprintprimes.

        Returns
        -------
        forces : dict
            Atomic forces.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            forces = np.zeros((len(selfindices), 3))

            for selfindex, (symbol, afp) in enumerate(fingerprints):
                for component in range(3):
                    arguments = dict(
                            index=selfindex,
                            symbol=symbol,
                            component=component,
                            hash=hash,
                            t_descriptor=t_descriptor,
                            sigma=self.parameters.sigma,
                            trainingimages=trainingimages,
                            fingerprintprimes=fingerprintprimes
                            )

                    preprocessing = self.parameters.preprocessing

                    if preprocessing:
                        arguments['preprocessing'] = preprocessing

                    if self.cholesky is False:
                        dforce = self.calculate_force(**arguments)
                    else:
                        dforce = self.forces_from_cholesky(**arguments)

                    forces[selfindex][component] += dforce
        return forces

    def calculate_dEnergy_dParameters(self, fingerprints):
        """Calculates a list of floats corresponding to the derivative of
        model-predicted energy of an image with respect to model parameters.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprints as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            denergy_dparameters = None
            for index, (symbol, afp) in enumerate(fingerprints):
                temp = self.calculate_dAtomicEnergy_dParameters(afp=afp,
                                                                index=index,
                                                                symbol=symbol)
                if denergy_dparameters is None:
                    denergy_dparameters = temp
                else:
                    denergy_dparameters += temp
        return denergy_dparameters

    def calculate_numerical_dEnergy_dParameters(self, fingerprints, d=0.00001):
        """Evaluates dEnergy_dParameters using finite difference.

        This will trigger two calls to calculate_energy(), with each parameter
        perturbed plus/minus d.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprints as values.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            vector = self.vector
            denergy_dparameters = []
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                eplus = self.calculate_energy(fingerprints)
                vector[_] -= 2 * d
                self.vector = vector
                eminus = self.calculate_energy(fingerprints)
                denergy_dparameters += [(eplus - eminus) / (2 * d)]
                vector[_] += d
                self.vector = vector
            denergy_dparameters = np.array(denergy_dparameters)
        return denergy_dparameters

    def calculate_dForces_dParameters(self, fingerprints, fingerprintprimes):
        """Calculates an array of floats corresponding to the derivative of
        model-predicted atomic forces of an image with respect to model
        parameters.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprint derivatives as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): None
                                   for selfindex in selfindices
                                   for i in range(3)}
            for key in fingerprintprimes.keys():
                selfindex, selfsymbol, nindex, nsymbol, i = key
                derafp = fingerprintprimes[key]
                afp = fingerprints[nindex][1]
                temp = self.calculate_dForce_dParameters(afp=afp,
                                                         derafp=derafp,
                                                         direction=i,
                                                         nindex=nindex,
                                                         nsymbol=nsymbol,)
                if dforces_dparameters[(selfindex, i)] is None:
                    dforces_dparameters[(selfindex, i)] = temp
                else:
                    dforces_dparameters[(selfindex, i)] += temp
        return dforces_dparameters

    def calculate_numerical_dForces_dParameters(self, fingerprints,
                                                fingerprintprimes, d=0.00001):
        """Evaluates dForces_dParameters using finite difference. This will
        trigger two calls to calculate_forces(), with each parameter perturbed
        plus/minus d.

        Parameters
        ---------
        fingerprints : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprint derivatives as values.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): []
                                   for selfindex in selfindices
                                   for i in range(3)}
            vector = self.vector
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                fplus = self.calculate_forces(fingerprints, fingerprintprimes)
                vector[_] -= 2 * d
                self.vector = vector
                fminus = self.calculate_forces(fingerprints, fingerprintprimes)
                for selfindex in selfindices:
                    for i in range(3):
                        dforces_dparameters[(selfindex, i)] += \
                            [(fplus[selfindex][i] - fminus[selfindex][i]) / (
                                2 * d)]
                vector[_] += d
                self.vector = vector
            for selfindex in selfindices:
                for i in range(3):
                    dforces_dparameters[(selfindex, i)] = \
                        np.array(dforces_dparameters[(selfindex, i)])
        return dforces_dparameters


class LossFunction:
    """Basic loss function, which can be used by the model.get_loss
    method which is required in standard model classes.

    This version is pure python and thus will be slow compared to a
    fortran/parallel implementation.

    If parallel is None, it will pull it from the model itself. Only use
    this keyword to override the model's specification.

    Also has parallelization methods built in.

    See self.default_parameters for the default values of parameters
    specified as None.

    Parameters
    ----------
    energy_coefficient : float
        Coefficient of the energy contribution in the loss function.
    force_coefficient : float
        Coefficient of the force contribution in the loss function.
        Can set to None as shortcut to turn off force training.
    convergence : dict
        Dictionary of keys and values defining convergence.  Keys are
        'energy_rmse', 'energy_maxresid', 'force_rmse', and 'force_maxresid'.
        If 'force_rmse' and 'force_maxresid' are both set to None, force
        training is turned off and force_coefficient is set to None.
    parallel : dict
        Parallel configuration dictionary. Will pull from model itself if
        not specified.
    overfit : float
        Multiplier of the weights norm penalty term in the loss function.
    raise_ConvergenceOccurred : bool
        If True will raise convergence notice.
    log_losses : bool
        If True will log the loss function value in the log file else will not.
    d : None or float
        If d is None, both loss function and its gradient are calculated
        analytically. If d is a float, then gradient of the loss function is
        calculated by perturbing each parameter plus/minus d.
    """

    default_parameters = {'convergence': {'energy_rmse': 0.001,
                                          'energy_maxresid': None,
                                          'force_rmse': None,
                                          'force_maxresid': None, }
                          }

    def __init__(self, energy_coefficient=1.0, force_coefficient=0.04,
                 convergence=None, parallel=None, overfit=0.,
                 raise_ConvergenceOccurred=True, log_losses=True, d=None):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        # 'dict' creates a copy; otherwise mutable in class.
        c = p['convergence'] = dict(self.default_parameters['convergence'])
        if convergence is not None:
            for key, value in convergence.items():
                p['convergence'][key] = value
        p['energy_coefficient'] = energy_coefficient
        p['force_coefficient'] = force_coefficient
        p['overfit'] = overfit
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self.log_losses = log_losses
        self.d = d
        self._initialized = False
        self._data_sent = False
        self._parallel = parallel
        self._step = 0
        if (c['force_rmse'] is None) and (c['force_maxresid'] is None):
            p['force_coefficient'] = None
        if p['force_coefficient'] is None:
            c['force_rmse'] = None
            c['force_maxresid'] = None

    def attach_model(self, model, fingerprints=None,
                     fingerprintprimes=None, images=None):
        """Attach the model to be used to the loss function.

        fingerprints and training images need not be supplied if they are
        already attached to the model via model.trainingparameters.

        Parameters
        ----------
        model : object
            Class representing the regression model.
        fingerprints : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashes as keys and the corresponding
            fingerprint derivatives as values.
        images : list or str
            List of ASE atoms objects with positions, symbols, energies, and
            forces in ASE format. This is the training set of data. This can
            also be the path to an ASE trajectory (.traj) or database (.db)
            file. Energies can be obtained from any reference, e.g. DFT
            calculations.
        """
        self._model = model
        self.fingerprints = fingerprints
        self.fingerprintprimes = fingerprintprimes
        self.images = images

    def _initialize(self, args):
        """Procedures to be run on the first call only, such as establishing
        SSH sessions, etc."""
        if self._initialized is True:
            return

        if self._parallel is None:
            self._parallel = self._model._parallel
        log = self._model.log

        if self.fingerprints is None:
            self.fingerprints = \
                self._model.trainingparameters.descriptor.fingerprints

        # May also make sense to decide whether or not to calculate
        # fingerprintprimes based on the value of train_forces.
        if ((self.parameters.force_coefficient is not None) and
                (self.fingerprintprimes is None)):
            self.fingerprintprimes = \
                self._model.trainingparameters.descriptor.fingerprintprimes
        if self.images is None:
            self.images = self._model.trainingparameters.trainingimages

        if self._parallel['cores'] != 1:
            # Initialize workers and send them parameters.

            python = sys.executable
            workercommand = '%s -m %s' % (python, self.__module__)
            self._sessions = setup_parallel(self._parallel, workercommand, log,
                                            setup_publisher=True)
            n_pids = self._sessions['n_pids']
            workerkeys = make_sublists(self.images.keys(), n_pids)
            server = self._sessions['master']
            setup_complete = np.array([False] * n_pids)
            while not setup_complete.all():
                message = server.recv_pyobj()
                if message['subject'] == 'purpose':
                    server.send_string('calculate_loss_function')
                elif message['subject'] == 'setup complete':
                    server.send_pyobj('thank you')
                    setup_complete[int(message['id'])] = True
                elif message['subject'] == 'request':
                    request = message['data']  # Variable name.
                    if request == 'images':
                        subimages = {k: self.images[k] for k in
                                     workerkeys[int(message['id'])]}
                        server.send_pyobj(subimages)
                    elif request == 'fortran':
                        server.send_pyobj(self._model.fortran)
                    elif request == 'modelstring':
                        server.send_pyobj(self._model.tostring())
                    elif request == 'lossfunctionstring':
                        server.send_pyobj(self.parameters.tostring())
                    elif request == 'fingerprints':
                        server.send_pyobj({k: self.fingerprints[k] for k in
                                           workerkeys[int(message['id'])]})
                    elif request == 'fingerprintprimes':
                        if self.fingerprintprimes is not None:
                            server.send_pyobj({k: self.fingerprintprimes[k]
                                               for k in
                                               workerkeys[int(message['id'])]})
                        else:
                            server.send_pyobj(None)
                    elif request == 'args':
                        server.send_pyobj(args)
                    elif request == 'publisher':
                        server.send_pyobj(self._sessions['publisher_socket'])
                    else:
                        raise NotImplementedError('Unknown request: {}'
                                                  .format(request))
            subscribers_working = np.array([False] * n_pids)

            def thread_function():
                """Broadcast from the background."""
                thread = threading.current_thread()
                while True:
                    if thread.abort is True:
                        break
                    self._sessions['publisher'].send_pyobj('test message')
                    time.sleep(0.1)

            thread = threading.Thread(target=thread_function)
            thread.abort = False  # to cleanly exit the thread
            thread.start()
            while not subscribers_working.all():
                message = server.recv_pyobj()
                server.send_pyobj('meaningless reply')
                if message['subject'] == 'subscriber working':
                    subscribers_working[int(message['id'])] = True
            thread.abort = True
            self._sessions['publisher'].send_pyobj('done')

        if self.log_losses:
            p = self.parameters
            convergence = p['convergence']
            log(' Loss function convergence criteria:')
            log('  energy_rmse: ' + str(convergence['energy_rmse']))
            log('  energy_maxresid: ' + str(convergence['energy_maxresid']))
            log('  force_rmse: ' + str(convergence['force_rmse']))
            log('  force_maxresid: ' + str(convergence['force_maxresid']))
            log(' Loss function set-up:')
            log('  energy_coefficient: ' + str(p.energy_coefficient))
            log('  force_coefficient: ' + str(p.force_coefficient))
            log('  overfit: ' + str(p.overfit))
            log('\n')
            if p.force_coefficient is None:
                header = '%5s %19s %12s %12s %12s'
                log(header %
                    ('', '', '', '', 'Energy'))
                log(header %
                    ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid'))
                log(header %
                    ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12))
            else:
                header = '%5s %19s %12s %12s %12s %12s %12s'
                log(header %
                    ('', '', '', '', 'Energy',
                     '', 'Force'))
                log(header %
                    ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid',
                     'ForceRMSE', 'MaxResid'))
                log(header %
                    ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12,
                     '=' * 12, '=' * 12))

        self._initialized = True

    def _send_data_to_fortran(self,):
        """Procedures to be run in fortran mode for a single requested core
        only. Also just on the first call for sending data to fortran modules.
        """
        if self._data_sent is True:
            return

        num_images = len(self.images)
        p = self.parameters
        energy_coefficient = p.energy_coefficient
        overfit = p.overfit
        if p.force_coefficient is None:
            train_forces = False
            force_coefficient = 0.
        else:
            train_forces = True
            force_coefficient = p.force_coefficient
        mode = self._model.parameters.mode
        if mode == 'atom-centered':
            num_atoms = None
        elif mode == 'image-centered':
            raise NotImplementedError('Image-centered mode is not coded yet.')

        (actual_energies, actual_forces, elements, atomic_positions,
         num_images_atoms, atomic_numbers, raveled_fingerprints, num_neighbors,
         raveled_neighborlists, raveled_fingerprintprimes) = (None,) * 10

        value = ravel_data(train_forces,
                           mode,
                           self.images,
                           self.fingerprints,
                           self.fingerprintprimes,)

        if mode == 'image-centered':
            if not train_forces:
                (actual_energies, atomic_positions) = value
            else:
                (actual_energies, actual_forces, atomic_positions) = value
        else:
            if not train_forces:
                (actual_energies, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints) = value
            else:
                (actual_energies, actual_forces, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints, num_neighbors,
                 raveled_neighborlists, raveled_fingerprintprimes) = value

        send_data_to_fortran(fmodules,
                             energy_coefficient,
                             force_coefficient,
                             overfit,
                             train_forces,
                             num_atoms,
                             num_images,
                             actual_energies,
                             actual_forces,
                             atomic_positions,
                             num_images_atoms,
                             atomic_numbers,
                             raveled_fingerprints,
                             num_neighbors,
                             raveled_neighborlists,
                             raveled_fingerprintprimes,
                             self._model,
                             self.d)
        self._data_sent = True

    def _cleanup(self):
        """Closes SSH sessions."""
        self._initialized = False
        if not hasattr(self, '_sessions'):
            return
        server = self._sessions['master']
        # Need to properly close socket connection (python3).
        server.close()

        for _ in self._sessions['connections']:
            if hasattr(_, 'logout'):
                _.logout()
        del self._sessions['connections']

    def get_loss(self, parametervector, energy_weights, energy_kernel,
                 forces_weights=None, forces_kernel=None, lossprime=False):
        """Returns the current value of the loss function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.
        energy_weights : list
            List of energy regression coefficients.
        energy_kernel : dict
            Dictionary of energy kernel matrix per-atom type.
        forces_weights : list
            List of forces regression coefficients.
        forces_kernel : dict
            Dictionary of forces kernel matrix per-atom type.
        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """

        self._initialize(args={'lossprime': lossprime, 'd': self.d})

        if self._parallel['cores'] == 1:
            if self._model.fortran:
                self._model.vector = parametervector
                self._send_data_to_fortran()
                (loss, dloss_dparameters, energy_loss, force_loss,
                 energy_maxresid, force_maxresid) = \
                    fmodules.calculate_loss(parameters=parametervector,
                                            num_parameters=len(
                                                parametervector),
                                            lossprime=lossprime)
            else:
                loss, dloss_dparameters, energy_loss, force_loss, \
                    energy_maxresid, force_maxresid = \
                    self.calculate_loss(parametervector,
                                        energy_weights,
                                        energy_kernel,
                                        forces_weights,
                                        forces_kernel,
                                        lossprime=lossprime)
        else:
            server = self._sessions['master']
            n_pids = self._sessions['n_pids']

            results = self.process_parallels(parametervector,
                                             server,
                                             n_pids)
            loss = results['loss']
            dloss_dparameters = results['dloss_dparameters']
            energy_loss = results['energy_loss']
            force_loss = results['force_loss']
            energy_maxresid = results['energy_maxresid']
            force_maxresid = results['force_maxresid']

        self.loss, self.energy_loss, self.force_loss, \
            self.energy_maxresid, self.force_maxresid = \
            loss, energy_loss, force_loss, energy_maxresid, force_maxresid

        if lossprime:
            self.dloss_dparameters = dloss_dparameters

        if self.raise_ConvergenceOccurred:
            # Only during calculation of loss function (and not lossprime)
            # convergence is checked and values are printed out in the log
            # file.
            if lossprime is False:
                self._model.vector = parametervector
                converged = self.check_convergence(loss,
                                                   energy_loss,
                                                   force_loss,
                                                   energy_maxresid,
                                                   force_maxresid)
                if converged:
                    self._cleanup()
                    raise ConvergenceOccurred()

        return {'loss': self.loss,
                'dloss_dparameters': (self.dloss_dparameters
                                      if lossprime is True
                                      else dloss_dparameters),
                'energy_loss': self.energy_loss,
                'force_loss': self.force_loss,
                'energy_maxresid': self.energy_maxresid,
                'force_maxresid': self.force_maxresid, }

    def calculate_loss(self, parametervector, energy_weights, energy_kernel,
                       forces_weights, forces_kernel, lossprime):
        """Method that calculates the loss, derivative of the loss with respect
        to parameters (if requested), and max_residual.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.
        energy_weights : list
            List of energy regression coefficients.
        energy_kernel : dict
            Dictionary of energy kernel matrix per-atom type.
        forces_weights : list
            List of forces regression coefficients.
        forces_kernel : dict
            Dictionary of forces kernel matrix per-atom type.
        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """
        self._model.vector = parametervector
        p = self.parameters
        energyloss = 0.
        forceloss = 0.
        energy_maxresid = 0.
        force_maxresid = 0.
        dloss_dparameters = np.array([0.] * len(parametervector))
        model = self._model

        force_resid = 0.

        for hash in self.images.keys():
            image = self.images[hash]
            no_of_atoms = len(image)
            amp_energy = model.calculate_energy(
                    self.fingerprints[hash],
                    hash)
            actual_energy = image.get_potential_energy(
                    apply_constraint=False)
            residual_per_atom = abs(
                    amp_energy - actual_energy
                    ) / no_of_atoms

            if residual_per_atom > energy_maxresid:
                energy_maxresid = residual_per_atom
            energyloss += residual_per_atom ** 2

            if p.force_coefficient is not None:
                descriptor = self._model.trainingparameters.descriptor
                amp_forces = \
                    model.calculate_forces(
                            self.fingerprints[hash],
                            self.fingerprintprimes[hash],
                            hash=hash,
                            t_descriptor=descriptor
                            )

                actual_forces = image.get_forces(apply_constraint=False)
                for index in range(no_of_atoms):
                    temp_f = np.linalg.norm(
                            amp_forces[index] - actual_forces[index],
                            ord=1
                            )
                    force_resid += temp_f

                force_resid = force_resid / no_of_atoms

                if force_resid > force_maxresid:
                    force_maxresid = force_resid

                forceloss += (1. / 3.) * force_resid ** 2

        loss = energyloss * p.energy_coefficient

        if p.force_coefficient is not None:
            loss += p.force_coefficient * forceloss

        # if model.lamda coefficient is more than zero, overfit
        # contribution to loss and dloss_dparameters is also added.

        if model.lamda > 0.:
            # Based on https://stats.stackexchange.com/a/70127/160746
            overfitloss = 0.
            for key in energy_kernel.keys():
                _weights = energy_weights[key]
                kernel = energy_kernel[key]
                overfitloss += _weights.T.dot(kernel.dot(_weights))

                if p.force_coefficient is not None:
                    for symbol in forces_kernel.keys():
                        for component in range(3):
                            _kernel = forces_kernel[symbol][component]
                            _vector = forces_weights[symbol][component]
                            overfitloss += _vector.T.dot(_kernel.dot(_vector))

            overfitloss *= model.lamda
            loss += overfitloss

        return loss, dloss_dparameters, energyloss, forceloss, \
            energy_maxresid, force_maxresid

    # All incoming requests will be dictionaries with three keys.
    # d['id']: process id number, assigned when process created above.
    # d['subject']: what the message is asking for / telling you.
    # d['data']: optional data passed from worker.

    def process_parallels(self, vector, server, n_pids):
        """

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        server : object
            Master session of parallel processing.
        processes: list of objects
            Worker sessions for parallel processing.
        """
        # FIXME/ap: We don't need to pass in most of the arguments.
        # They are stored already.
        results = {'loss': 0.,
                   'dloss_dparameters': [0.] * len(vector),
                   'energy_loss': 0.,
                   'force_loss': 0.,
                   'energy_maxresid': 0.,
                   'force_maxresid': 0.}

        publisher = self._sessions['publisher']

        # Broadcast parameters for this call.
        publisher.send_pyobj(vector)

        # Receive the result.
        finished = np.array([False] * self._sessions['n_pids'])
        while not finished.all():
            message = server.recv_pyobj()
            server.send_pyobj('thank you')

            assert message['subject'] == 'result'
            result = message['data']

            results['loss'] += result['loss']
            results['dloss_dparameters'] += result['dloss_dparameters']
            results['energy_loss'] += result['energy_loss']
            results['force_loss'] += result['force_loss']
            if result['energy_maxresid'] > results['energy_maxresid']:
                results['energy_maxresid'] = result['energy_maxresid']
            if result['force_maxresid'] > results['force_maxresid']:
                results['force_maxresid'] = result['force_maxresid']
            finished[int(message['id'])] = True

        return results

    def check_convergence(self, loss, energy_loss, force_loss,
                          energy_maxresid, force_maxresid):
        """Check convergence

        Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer.

        Parameters
        ----------
        loss : float
            Value of the loss function.
        energy_loss : float
            Value of the energy contribution of the loss function.
        force_loss : float
            Value of the force contribution of the loss function.
        energy_maxresid : float
            Maximum energy residual.
        force_maxresid : float
            Maximum force residual.
        """
        p = self.parameters
        images = self._model.trainingparameters.trainingimages
        energy_rmse_converged = True
        log = self._model.log
        if p.convergence['energy_rmse'] is not None:
            energy_rmse = np.sqrt(energy_loss / len(images))
            if energy_rmse > p.convergence['energy_rmse']:
                energy_rmse_converged = False
        energy_maxresid_converged = True
        if p.convergence['energy_maxresid'] is not None:
            if energy_maxresid > p.convergence['energy_maxresid']:
                energy_maxresid_converged = False
        if p.force_coefficient is not None:
            force_rmse_converged = True
            if p.convergence['force_rmse'] is not None:
                force_rmse = np.sqrt(force_loss / len(images))
                if force_rmse > p.convergence['force_rmse']:
                    force_rmse_converged = False
            force_maxresid_converged = True
            if p.convergence['force_maxresid'] is not None:
                if force_maxresid > p.convergence['force_maxresid']:
                    force_maxresid_converged = False

            if self.log_losses:
                log('%5i %19s %12.4e %10.4e %1s'
                    ' %10.4e %1s %10.4e %1s %10.4e %1s' %
                    (self._step, now(), loss, energy_rmse,
                     'C' if energy_rmse_converged else '-',
                     energy_maxresid,
                     'C' if energy_maxresid_converged else '-',
                     force_rmse,
                     'C' if force_rmse_converged else '-',
                     force_maxresid,
                     'C' if force_maxresid_converged else '-'))
            return energy_rmse_converged and energy_maxresid_converged and \
                force_rmse_converged and force_maxresid_converged
        else:
            if self.log_losses:
                log('%5i %19s %12.4e %10.4e %1s %10.4e %1s' %
                    (self._step, now(), loss, energy_rmse,
                     'C' if energy_rmse_converged else '-',
                     energy_maxresid,
                     'C' if energy_maxresid_converged else '-'))
            return energy_rmse_converged and energy_maxresid_converged


class KernelRidge(Model):
    """Class implementing Kernelized Ridge Regression in Amp

    Parameters
    ----------
    sigma : float, list, or dict
        Length scale of the Gaussian in the case of RBF, exponential, and
        laplacian kernels. Default is 1. (float) and it computes isotropic
        kernels. Pass a list if you would like to compute anisotropic kernels,
        or a dictionary if you want sigmas for each model.

        Example:

            >>> sigma={'energy': {'H': value, 'O': value},
                       'forces': {'H': {0: value, 1: value, 2: value},
                              'O': {0: value, 1: value, 2: value}}}

        `value` can be a float or a list.
    kernel : str
        Choose the kernel. Available kernels are: 'linear', 'rbf', 'laplacian',
        and 'exponential'. Default is 'rbf'.
    lamda : float, or dictionary
        Strength of the regularization. If you pass a dictionary then force and
        energy will have different regularization:

            >>> lamda = {'energy': value, 'forces': value}

        Dictionaries are only used when performing Cholesky factorization.
    weights : dict
        Dictionary of weights.
    regressor : object
        Regressor class to be used.
    mode : str
        Atom- or image-centered mode.
    trainingimages : str
        Path to Trajectory file containing the images in the training set. This
        is useful for predicting new structures.
    version : str
        Version.
    fortran : bool
        Use fortran code.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training set.  Specify None for
        no checkpoints. Default is None.
    lossfunction : object
        Loss function object.
    cholesky : bool
        Whether or not we are using Cholesky decomposition to determine the
        weights. This method returns an unique set of regression coefficients.
    weights_independent : bool
        Whether or not the weights are going to be split for energy and forces.
    randomize_weights : bool
        If set to True, weights are randomly started when minimizing the L2
        loss function.
    forcetraining : bool
        Turn force training true.
    nnpartition : str
        Use per-atom energy partition from an Amp neural network calculator.
        You have to set the path to .amp file. Useful for energy training with
        Cholesky factorization. Default is set to None.
    preprocessing : bool
        Preprocess training data.
    sum_rule : bool
        Whether or not we sum of fingerprintprime elements over a given axis.
        This applies np.sum(fingerprint_list, axis=0).

    Notes
    -----
        In the case of training total energies, we need to apply either an
        atomic decomposition Ansatz (ADA) during training or an energy
        partition scheme to the training set. ADA can be achieved based on
        Int. J.  Quantum Chem., vol. 115, no.  16, pp.  1051-1057, Aug. 2015".
        For an explanation of what they do, see the Master thesis by Sonja
        Mathias.

        http://wissrech.ins.uni-bonn.de/teaching/master/masterthesis_mathias_revised.pdf

        ADA is the default way of training total energies in this KernelRidge
        class.

        An energy partition scheme for  total energies can be obtained from an
        artificial neural network or methods such as the interacting quantum
        atoms theory (IQA). I implemented the nnpartition mode for which users
        can provide the path to a NN calculator and we take the energies
        per-atom from the function .calculate_atomic_energy(). The strategy
        would be to use train the NN with a very tight convergence criterion
        (1e-6 RSME).  Then, calling .calculate_atomic_energy() would give you
        the atomic energies for such set.

        For forces is a different history because we do know the derivative of
        the energy with respect to atom positions (a per-atom quantity).  So we
        rely on the method in the algorithm shown in Rupp, M. (2015).  Machine
        learning for quantum mechanics in a nutshell. International Journal of
        Quantum Chemistry, 115(16), 1058-1073.
    """
    def __init__(self, sigma=1., kernel='rbf', lamda=1e-5, weights=None,
                 regressor=None, mode=None, trainingimages=None, version=None,
                 fortran=False, checkpoints=None, lossfunction=None,
                 cholesky=True, weights_independent=True,
                 randomize_weights=False, forcetraining=False,
                 preprocessing=False, nnpartition=None, sum_rule=True):

        np.set_printoptions(precision=30, threshold=999999999)

        # Version check, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use KernelRidge'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               'newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        p = self.parameters = Parameters()
        p.importname = '.model.kernelridge.KernelRidge'
        p.version = version
        p.weights = weights
        p.weights_independent = self.weights_independent = weights_independent
        p.mode = mode
        p.kernel = self.kernel = kernel
        p.sigma = self.sigma = sigma
        p.lamda = self.lamda = lamda
        p.sum_rule = self.sum_rule = sum_rule
        p.cholesky = self.cholesky = cholesky
        p.nnpartition = self.nnpartition = nnpartition
        p.trainingimages = self.trainingimages = trainingimages
        p.preprocessing = self.preprocessing = preprocessing

        self.randomize_weights = randomize_weights
        self.regressor = regressor
        self.parent = None  # Can hold a reference to main Amp instance.
        self.fortran = fortran
        self.checkpoints = checkpoints
        self.lossfunction = lossfunction
        self.properties = ['energy']
        if forcetraining:
            self.properties.append('forces')

        self.kernel_e = OrderedDict()  # Kernel dictionary for energies
        self.kernel_f = OrderedDict()  # Kernel dictionary for forces

        if self.lossfunction is None:
            self.lossfunction = LossFunction()
            if forcetraining is True and cholesky is True:
                self.lossfunction.parameters['force_coefficient'] = True

    def fit(self, trainingimages, descriptor, log, parallel, only_setup=False):
        """Fit kernel ridge model

        This function is capable to fit KernelRidge using either a L2 loss
        function or matrix factorization in the case when the cholesky keyword
        argument is set to True.

        Parameters
        ----------
        trainingimages : dict
            Hashed dictionary of training images.
        descriptor : object
            Class with local chemical environments of atoms.
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        parallel: dict
            Parallel configuration dictionary. Takes the same form as in
            amp.Amp.
        """

        # Set all parameters and report to logfile.
        self._parallel = parallel
        self._log = log

        if self.regressor is None and self.cholesky is False:
            from ..regression import Regressor
            # lossprime is not yet implemented when optimizing the loss
            # function.
            self.regressor = Regressor(lossprime=False)

        p = self.parameters
        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor

        if self.preprocessing is True:
            log('Preprocessing data...', tic='preprocessing')
            preprocessed_afp = self.preprocess_features(
                    tp.trainingimages,
                    tp.descriptor,
                    forcetraining=self.forcetraining)
            log('...preprocessing finished in ', toc='preprocessing')
            tp.fingerprints = preprocessed_afp[0]
        else:
            tp.fingerprints = tp.descriptor.fingerprints

        if p.mode is None:
            p.mode = descriptor.parameters.mode
        else:
            assert p.mode == descriptor.parameters.mode
        log('Regression in %s mode.' % p.mode)

        if len(list(self.kernel_e.keys())) == 0:

            if isinstance(p.sigma, float) or isinstance(p.sigma, int):
                log('Calculating isotropic %s kernel with same sigma for all '
                    'atoms and properties...' % self.kernel, tic='kernel')
                sigma = self.get_sigma(p.sigma, user_input='float',
                                       forcetraining=self.forcetraining)

            elif isinstance(p.sigma, list):
                log('Calculating anisotropic %s kernel with same sigma for '
                    'all atoms and properties...' % self.kernel, tic='kernel')
                sigma = self.get_sigma(p.sigma, user_input='list',
                                       forcetraining=self.forcetraining)

            elif isinstance(p.sigma, dict):
                log('Calculating %s kernels with specified sigmas for '
                    'each property...' % self.kernel, tic='kernel')
                sigma = self.get_sigma(p.sigma, user_input='dict',
                                       forcetraining=self.forcetraining)

            p.sigma = self.sigma = sigma
            log('Kernel parameters:')
            log('    lamda: %s' % self.lamda)
            log('    sigma: %s' % self.sigma)
            kij_args = dict(trainingimages=tp.trainingimages,
                            fp_trainingimages=tp.fingerprints,)

            if self.fortran is True:
                kij_args['only_features'] = True

            self.get_energy_kernel(**kij_args)

            if self.forcetraining is True:
                if self.preprocessing:
                    tp.descriptor = preprocessed_afp[1]

                kijf_args = dict(trainingimages=tp.trainingimages,
                                 t_descriptor=tp.descriptor)

                if self.fortran is True:
                    kijf_args['only_features'] = True
                self.get_forces_kernel(**kijf_args)

                log('...kernel matrices computed in', toc='kernel')
            else:
                log('...kernel matrix computed in', toc='kernel')

        # These weights are used for the case of the l2 loss function
        # minimization
        if p.weights is None:
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                weights = OrderedDict()

                if self.cholesky is False:
                    force_rmse = self.lossfunction.parameters['convergence']['force_rmse']

                    if force_rmse is not None:
                        self.properties.append('forces')

                for prop in self.properties:
                    weights[prop] = OrderedDict()

                    if self.cholesky is False:
                        log('Initializing weights.')
                        for hash in tp.trainingimages.keys():
                            imagefingerprints = tp.fingerprints[hash]
                            for symbol, fingerprint in imagefingerprints:
                                if (symbol not in weights and
                                        prop == 'energy'):
                                    size = \
                                        len(self.reference_features_e[symbol])
                                    if self.randomize_weights:
                                        weights[prop][symbol] = \
                                                np.random.uniform(
                                                low=-1.0,
                                                high=1.0,
                                                size=(size))
                                    else:
                                        weights[prop][symbol] = np.ones(size)
                                elif (symbol not in weights and
                                        prop == 'forces'):
                                    if p.weights_independent is True:
                                        size = \
                                            len(self.ref_features_f[symbol][0])
                                        if self.randomize_weights:
                                            weights[prop][symbol] = \
                                                    np.random.uniform(
                                                    low=-1.0,
                                                    high=1.0,
                                                    size=(3, size)
                                                    )
                                        else:
                                            weights[prop][symbol] = np.ones(
                                                                    (3, size))
                                    else:
                                        weights[prop][symbol] = \
                                                np.ones(size)
                p.weights = weights
        else:
            log('Initial weights already present.')

        if only_setup:
            return

        if self.cholesky is False:
            result = self.regressor.regress(model=self, log=log)
            return result  # True / False
        else:
            try:
                if self.nnpartition is None:
                    log('Starting Cholesky decomposition of energy kernel '
                        'matrix obtained from the atomic energy decomposition '
                        'Ansatz...', tic='energy')

                    size = len(self.reference_features_e)
                    K = self.kij.reshape(size, size)
                    K = self.LT.dot(K).dot(self.LT.T)
                    I_e = np.identity(K.shape[0])

                    if isinstance(self.lamda, dict):
                        lamda = self.lamda['energy']
                    else:
                        lamda = self.lamda

                    cholesky_U = cholesky((K + lamda * I_e))
                    betas = np.linalg.solve(cholesky_U.T, self.energy_targets)
                    _weights = np.linalg.solve(cholesky_U, betas)

                    log('Shape of energy kernel matrix is {}.'
                        .format(K.shape))

                    weights = [w * g for index, w in enumerate(_weights) for
                               g in self.fingerprint_map[index]]

                    log('... Cholesky decomposition finished in ',
                        toc='energy')

                    p.weights['energy'] = weights
                else:
                    log('Starting Cholesky decomposition of energy kernel'
                        ' matrix... ',
                        tic='cholesky_energy_kernel')

                    symbols = []
                    log('Shape of energy kernel matrix for each element:')
                    for symbol in self.kernel_e_loss.keys():
                        size = self.kernel_e_loss[symbol].shape
                        I_e = np.identity(size[0])
                        kernel = self.kernel_e_loss[symbol]
                        if symbol not in symbols:
                            log('    {}: {}' .format(symbol, size))
                            symbols.append(symbol)

                        if isinstance(self.lamda, dict):
                            lamda = self.lamda['energy']
                        else:
                            lamda = self.lamda

                        cholesky_U = cholesky((kernel + lamda * I_e))

                        betas = np.linalg.solve(cholesky_U.T,
                                                self.energy_targets[symbol])
                        weights = np.linalg.solve(cholesky_U, betas)
                        p.weights['energy'][symbol] = weights

                    log('... Cholesky decompositions finished in ',
                        toc='cholesky_energy_kernel')

                if self.forcetraining is True:
                    log('Starting Cholesky decomposition of force kernel '
                        'matrix...', tic='cholesky_force_kernel')

                    log('Shape of force kernel matrix for each element:')

                    for symbol in self.kernel_f_cholesky.keys():
                        p.weights['forces'][symbol] = []
                        symbols = []

                        for i in range(3):
                            K_f = np.array(self.kernel_f_cholesky[symbol][i])
                            size = K_f.shape
                            if symbol not in symbols:
                                log('    {}: {}' .format(symbol, size))
                                symbols.append(symbol)
                            I_f = np.identity(size[0])

                            if isinstance(self.lamda, dict):
                                lamda = self.lamda['forces']
                            else:
                                lamda = self.lamda

                            cholesky_U = cholesky((K_f + lamda * I_f))
                            betas = np.linalg.solve(
                                       cholesky_U.T,
                                       self.force_targets[symbol][i]
                                       )
                            weights = np.linalg.solve(cholesky_U, betas)
                            p.weights['forces'][symbol].append(weights)
                    log('... Cholesky decompositions finished in ',
                        toc='cholesky_force_kernel')
                return True
            except np.linalg.linalg.LinAlgError:
                log('The kernel matrix seems to be singular. Add more\n'
                    'noise to its diagonal elements by increasing the '
                    'penalization term.')
                return False
            except:
                return False

    def get_sigma(self, sigma, user_input, forcetraining=False):
        """Function to build sigma

        Parameters
        ----------
        sigma : float, list or dict.
            This is the raw user input for sigma.
        user_input : type
            Checks the type of user input for sigma.
        forcetraining : bool
            Whether or not force training is set to true.

        Returns
        -------
        _sigma : dict
            Universal sigma dictionary for KernelRidge in Amp.
        """

        tp = self.trainingparameters
        trainingimages = tp['trainingimages']
        fingerprints = tp['fingerprints']
        _sigma = {}

        if user_input == 'float' or user_input == 'list':
            for hash in trainingimages.keys():
                # We create 'energy' key
                _sigma['energy'] = {}

                for symbol, afp in fingerprints[hash]:
                    if symbol not in _sigma['energy'].keys():
                        _sigma['energy'][symbol] = sigma

                    if forcetraining:
                        if 'forces' not in _sigma.keys():
                            _sigma['forces'] = {}

                        if symbol not in _sigma['forces'].keys():
                            _sigma['forces'][symbol] = {}

                        for component in range(3):
                            _sigma['forces'][symbol][component] = sigma
            return _sigma

        elif user_input == 'dict':
            # When user_input is a dict, we check that the structure is ok

            symbols = []
            for hash in trainingimages.keys():
                for symbol, afp in fingerprints[hash]:
                    if symbol not in symbols:
                        symbols.append(symbol)

            for prop in self.properties:
                check_property = sigma[prop]
                try:
                    if len(check_property.keys()) != len(symbols):
                        self._log('Property {} has not the correct '
                                  'number of atom symbols...' .format(prop))
                        self._log("Structure of the sigma dictionary must be "
                                  "sigma = {'energy': {symbol: (value or "
                                  "list)},"
                                  "'forces': {symbol: {0: (value or list), "
                                  "1: (value or list), 2: (value or list)}}}")
                        raise('Incorrect number of atoms in sigma '
                              'dictionary... \n '
                              'Check the output file for more information.')

                    if prop == 'forces':
                        for symbol in check_property.keys():
                            try:
                                check_property[symbol].keys()
                            except AttributeError:
                                self._log('Forces in sigma dictionary need at '
                                          'least one '
                                          'key component...')
                                self._log("Structure of the sigma dictionary "
                                          "must be sigma = {'energy': {symbol:"
                                          " (value or list)}, 'forces': "
                                          "{symbol: {0: (value or list), "
                                          "1: (value or list), 2: (value or "
                                          "list)}}}")
                                raise('Forces in sigma dictionary need at '
                                      'least one component key... Check the '
                                      'output file for more information.')
                except AttributeError:
                    self._log('Adjusting sigma dictionary...')

                    for hash in trainingimages.keys():
                        # We create 'energy' key
                        _sigma[prop] = {}

                        for symbol, afp in fingerprints[hash]:
                            if symbol not in _sigma[prop].keys():
                                _sigma[prop][symbol] = sigma['energy']

                            if forcetraining:
                                if 'forces' not in _sigma.keys():
                                    _sigma['forces'] = {}

                                if symbol not in _sigma['forces'].keys():
                                    _sigma['forces'][symbol] = {}

                                for component in range(3):
                                    _sigma['forces'][symbol][component] = \
                                                            sigma['forces']
                    return _sigma
                else:
                    return sigma
        return _sigma

    def preprocess_features(self, trainingimages, descriptor, afp=None,
                            fprime=None, forcetraining=False,
                            component=None):
        """Preprocess fingerprints

        Parameters
        ----------
        descriptor : object
            Object containing fingerprints.
        trainingimages : object
            Training images in ASE format.
        afp : list
            Atomic fingerprint as a list. Useful for energy_from_cholesky().
        fprime : list
            Derivative of atomic fingerprint as a list. Useful for
            forces_from_cholesky().
        forcetraining : bool
            Whether or not the forces are going to be preprocessed.
        component : int
            X, Y or Z component of the atomic forces.

        Notes
        -----
            Training set features have to be scaled for using dual metrics as
            kernels. Scikit-learn offers good preprocessing tools. Once
            a scaler is fit with training set features and used for scaling
            them, we used that fitted scaler to transform unseen feature
            vectors, too.  For some more information read:

                https://stackoverflow.com/q/49509575/1995261
                https://stackoverflow.com/q/41506134/1995261
        """
        from sklearn.preprocessing import StandardScaler
        hashes = list(hash_images(trainingimages).keys())

        try:
            fp = descriptor.fingerprints
        except AttributeError:
            fp = descriptor

        energy_fingerprints = []
        symbols = []
        fingerprints = OrderedDict()
        fingerprintprimes = OrderedDict()

        if fprime is None:
            for hash in hashes:
                _symbols = []
                for symbol, fingerprint in fp[hash]:
                    _symbols.append(symbol)
                    energy_fingerprints.append(fingerprint)
                symbols.append(_symbols)

            # Making a numpy array
            energy_fingerprints = np.array(energy_fingerprints)

            try:
                # We verify that KernelRidge has the self.energy_scaler
                # attribute. This would ensure that it is computed when needed
                # only.
                self.energy_scaler
                self.energy_scaled_fp
            except AttributeError:
                self.energy_scaler = StandardScaler().fit(energy_fingerprints)

                self.energy_scaled_fp = self.energy_scaler.transform(
                        energy_fingerprints)

            if isinstance(afp, list):
                afp = self.energy_scaler.transform(
                      np.array(afp).reshape(1, -1))
                return afp

            inc = 0
            for index, hash in enumerate(hashes):
                fingerprints[hash] = OrderedDict()
                append_this = []
                for symbol in symbols[index]:
                    append_this.append((symbol, self.energy_scaled_fp[inc]))
                    inc += 1
                fingerprints[hash] = append_this

        if afp is None:
            if forcetraining:
                try:
                    fprimes = descriptor.fingerprintprimes
                # Needed for predictions
                except AttributeError:
                    fprimes = descriptor

                forces_fingerprints_x = []
                forces_fingerprints_y = []
                forces_fingerprints_z = []

                try:
                    self.force_scaler_x
                    self.force_scaler_y
                    self.force_scaler_z
                    self.forces_scaled_fp_x
                    self.forces_scaled_fp_y
                    self.forces_scaled_fp_z
                except AttributeError:
                    for hash in hashes:
                        fingerprintprimes[hash] = OrderedDict()
                        for key in sorted(fprimes[hash].keys()):
                            if key[-1] == 0:
                                forces_fingerprints_x.append(
                                        fprimes[hash][key])
                            elif key[-1] == 1:
                                forces_fingerprints_y.append(
                                        fprimes[hash][key])
                            elif key[-1] == 2:
                                forces_fingerprints_z.append(
                                        fprimes[hash][key])

                    forces_fingerprints_x = np.array(forces_fingerprints_x)
                    forces_fingerprints_y = np.array(forces_fingerprints_y)
                    forces_fingerprints_z = np.array(forces_fingerprints_z)

                    self.force_scaler_x = StandardScaler().fit(
                            forces_fingerprints_x)
                    self.force_scaler_y = StandardScaler().fit(
                            forces_fingerprints_y)
                    self.force_scaler_z = StandardScaler().fit(
                            forces_fingerprints_z)
                    self.forces_scaled_fp_x = self.force_scaler_x.transform(
                            forces_fingerprints_x)
                    self.forces_scaled_fp_y = self.force_scaler_y.transform(
                            forces_fingerprints_y)
                    self.forces_scaled_fp_z = self.force_scaler_z.transform(
                            forces_fingerprints_z)

                if isinstance(fprime, list):
                    if component == 0:
                        fprime = self.force_scaler_x.transform(fprime)
                    elif component == 1:
                        fprime = self.force_scaler_y.transform(fprime)
                    elif component == 2:
                        fprime = self.force_scaler_z.transform(fprime)

                    return fprime

                for hash in hashes:
                    ix, iy, iz = 0, 0, 0
                    for key in sorted(fprimes[hash].keys()):
                        if key[-1] == 0:
                            fingerprintprimes[hash][key] = \
                                    self.forces_scaled_fp_x[ix]
                            ix += 1
                        elif key[-1] == 1:
                            fingerprintprimes[hash][key] = \
                                    self.forces_scaled_fp_y[iy]
                            iy += 1
                        elif key[-1] == 2:
                            fingerprintprimes[hash][key] = \
                                    self.forces_scaled_fp_z[iz]
                            iz += 1

        return fingerprints, fingerprintprimes

    def get_energy_kernel(self, trainingimages=None, fp_trainingimages=None,
                          only_features=False):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing information about the images. Note
            that you have to hash the images before passing them to this
            method.
        fp_trainingimages : object
            Fingerprints calculated using the trainingimages.
        only_features : bool
            If set to True, only the self.reference_features_e are built.

        Returns
        -------
        kernel_e : dictionary
            The kernel in a dictionary where keys are images' hashes.
        """
        # This creates a list containing all features in all images on the
        # training set.
        if self.cholesky and self.nnpartition is None:
            self.kij = []
            self.energy_targets = []
            # Matrix needed to use the atomic decomposition Ansatz
            self.LT = []
            self.fingerprint_map = []
            self.reference_features_e = []
        else:
            self.energy_targets = OrderedDict()
            self.reference_features_e = OrderedDict()
            self.kernel_e_loss = OrderedDict()

        hashes = list(hash_images(trainingimages).keys())

        for hash in hashes:
            if self.cholesky is False or self.nnpartition is not None:
                for symbol, afp in fp_trainingimages[hash]:
                    if symbol not in self.reference_features_e.keys():
                        self.reference_features_e[symbol] = []
                    afp = np.asarray(afp)
                    self.reference_features_e[symbol].append(afp)
            else:
                afp_in_hash = fp_trainingimages[hash]
                f_map = []

                for symbol, afp in afp_in_hash:
                    f_map.append(1)
                    self.reference_features_e.append((symbol, np.asarray(afp)))
                self.fingerprint_map.append(f_map)

        if only_features is False:
            if self.nnpartition is not None:
                # Load the neural network calculator just once
                from .. import Amp
                nn_calc = Amp.load(self.nnpartition)

            for index, hash in enumerate(hashes):
                self.kernel_e[hash] = OrderedDict()
                kernel = []

                if self.cholesky is False and self.nnpartition is None:
                    # This is the case when using L2 loss function.
                    for index, (symbol, afp) in enumerate(
                            fp_trainingimages[hash]):

                        if symbol not in self.kernel_e_loss.keys():
                            self.kernel_e_loss[symbol] = []

                        if isinstance(self.sigma, dict):
                            sigma = self.sigma['energy'][symbol]
                        else:
                            sigma = self.sigma

                        _kernel = self.kernel_matrix(
                                np.asarray(afp),
                                self.reference_features_e[symbol],
                                kernel=self.kernel,
                                sigma=sigma
                                )
                        self.kernel_e[hash][(index, symbol)] = _kernel
                        self.kernel_e_loss[symbol].append(_kernel)
                        kernel.append(_kernel)

                elif self.cholesky is True and self.nnpartition is not None:
                    """
                    When using the per-atom energy partition from the neural
                    network, self.energy_targets is a dictionary and has to be
                    populated in here using the atomic_energies from the NN.
                    """
                    for index, (symbol, afp) in enumerate(
                            fp_trainingimages[hash]):

                        if symbol not in self.kernel_e_loss.keys():
                            # This should guarantee that order is respected.
                            self.kernel_e_loss[symbol] = []
                            self.energy_targets[symbol] = []

                        if isinstance(self.sigma, dict):
                            sigma = self.sigma['energy']
                        else:
                            sigma = self.sigma

                        _kernel = self.kernel_matrix(
                                np.asarray(afp),
                                self.reference_features_e[symbol],
                                kernel=self.kernel,
                                sigma=sigma
                                )
                        self.kernel_e[hash][(index, symbol)] = _kernel
                        self.kernel_e_loss[symbol].append(_kernel)
                        kernel.append(_kernel)
                        atomic_energy = nn_calc.model.calculate_atomic_energy(
                                        afp,
                                        index,
                                        symbol)
                        self.energy_targets[symbol].append(np.array(
                                                           atomic_energy))
                    """ For debugging purposes only
                        total_energy += atomic_energy
                    print('DFT energy:', energy)
                    print('ANN energy:', total_energy)
                    """
                else:
                    """
                    This is the case when using the atomic decomposition
                    Ansatz.
                    """
                    # We append targets
                    energy = trainingimages[hash].get_potential_energy()
                    self.energy_targets.append(energy)

                    # We build L.T matrix
                    _LT = []

                    for i, group in enumerate(self.fingerprint_map):
                        if i == index:
                            for _ in group:
                                _LT.append(1.)
                        else:
                            for _ in group:
                                _LT.append(0.)
                    self.LT.append(_LT)

                    # Building the kernel matrix
                    for index, (symbol, afp) in enumerate(
                            fp_trainingimages[hash]):

                        sigma = self.sigma['energy'][symbol]

                        _kernel = self.kernel_matrix(
                                np.asarray(afp),
                                self.reference_features_e,
                                feature_symbol=symbol,
                                kernel=self.kernel,
                                sigma=sigma
                                )
                        self.kij.append(_kernel)

            if self.cholesky and self.nnpartition is None:
                self.kij = np.asarray(self.kij)
                self.LT = np.asarray(self.LT)

            elif self.cholesky is False or self.nnpartition is not None:
                for key in self.kernel_e_loss.keys():
                    _s = len(self.kernel_e_loss[key][0])
                    arr = self.kernel_e_loss[key]
                    self.kernel_e_loss[key] = np.array(arr).reshape(_s, _s)

    def get_forces_kernel(self, trainingimages=None, t_descriptor=None,
                          only_features=False):
        """Method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing the training set images. Note that
            images have to be hashed before passing them to this method.
        t_descriptor : object
            Descriptor object containing the fingerprintprimes from the
            training set.
        only_features : bool
            If set to True, only the self.force_features are built.

        Returns
        -------
        kernel_f : dictionary
            Dictionary containing images hashes and kernels per-atom.
        """

        hashes = list(hash_images(trainingimages).keys())

        # For non processed features
        try:
            fingerprintprimes = t_descriptor.fingerprintprimes
        # For processed features
        except AttributeError:
            fingerprintprimes = t_descriptor

        self.force_features = OrderedDict()
        self.ref_features_f = OrderedDict()

        for hash in hashes:
            self.force_features[hash] = OrderedDict()
            image = trainingimages[hash]

            # We iterate once over the whole fingerprint object per hash
            # in order to build the right dictionaries for applying the
            # kernel functions.
            for key, dfp in sorted(fingerprintprimes[hash].items()):
                _key = (key[0], key[1])
                selfsymbol = key[1]
                component = key[-1]

                if _key not in self.force_features[hash].keys():
                    self.force_features[hash][_key] = OrderedDict()

                if component not in self.force_features[hash][_key].keys():
                    self.force_features[hash][_key][component] = []

                self.force_features[hash][_key][component].append(dfp)

            # We iterate over the self.force_features dictionary to build
            # the references per atom.
            for k, dfp in sorted(self.force_features[hash].items()):
                selfsymbol = k[1]

                if selfsymbol not in self.ref_features_f.keys():
                    self.ref_features_f[selfsymbol] = OrderedDict()

                for component, dfp in self.force_features[hash][k].items():
                    if self.sum_rule:
                        dfp = np.sum(np.array(dfp), axis=0)
                        self.force_features[hash][k][component] = dfp

                    if component not in self.ref_features_f[selfsymbol].keys():
                        self.ref_features_f[selfsymbol][component] = []

                    self.ref_features_f[selfsymbol][component].append(dfp)

        if only_features is False:
            # if self.cholesky is True:
            self.force_targets = OrderedDict()
            self.kernel_f_cholesky = OrderedDict()

            for hash in hashes:
                image = trainingimages[hash]
                self.kernel_f[hash] = OrderedDict()

                # if self.cholesky is True:
                actual_forces = image.get_forces(apply_constraint=False)

                for atom in image:
                    selfsymbol = atom.symbol
                    selfindex = atom.index
                    self.kernel_f[hash][
                            (selfindex, selfsymbol)] = OrderedDict()

                    if selfsymbol not in self.kernel_f_cholesky.keys():
                        self.kernel_f_cholesky[selfsymbol] = OrderedDict()
                        self.force_targets[selfsymbol] = OrderedDict()

                    for component in range(3):
                        keys = self.kernel_f_cholesky[selfsymbol].keys()
                        if component not in keys:
                            self.kernel_f_cholesky[selfsymbol][component] = []
                            self.force_targets[selfsymbol][component] = []

                        afp = self.force_features[hash][
                                (selfindex, selfsymbol)][component]

                        sigma = self.sigma['forces'][selfsymbol][component]

                        _kernel = self.kernel_matrix(
                                afp,
                                self.ref_features_f[selfsymbol][component],
                                kernel=self.kernel,
                                sigma=sigma
                                )
                        self.kernel_f[hash][
                                (selfindex, selfsymbol)][
                                        component] = _kernel
                        # if self.cholesky is True:
                        target = actual_forces[selfindex][component]

                        self.kernel_f_cholesky[selfsymbol][component].append(
                                _kernel)
                        self.force_targets[selfsymbol][component].append(
                                target)

            if self.cholesky is False:
                for symbol in self.kernel_f_cholesky.keys():
                    for component in self.kernel_f_cholesky[symbol].keys():
                        _s = len(self.kernel_f_cholesky[symbol][component])
                        arr = np.array(
                                self.kernel_f_cholesky[symbol][component]
                                ).reshape(_s, _s)
                        self.kernel_f_cholesky[symbol][component] = arr
            return self.kernel_f

    @property
    def forcetraining(self):
        """Returns True if forcetraining is turned on (as determined by
        examining the convergence criteria in the loss function), else
        returns False.
        """
        if self.lossfunction.parameters['force_coefficient'] is None:
            forcetraining = False
        elif self.lossfunction.parameters['force_coefficient'] > 0.:
            forcetraining = True
        return forcetraining

    @property
    def vector(self):
        """Access to get or set the model parameters (weights for each kernel)
        as a single vector, useful in particular for regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters

        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(
                    p.weights,
                    weights_independent=self.weights_independent
                    )
        return self.ravel.to_vector(weights=p.weights)

    @vector.setter
    def vector(self, vector):
        p = self.parameters

        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights)
        weights = self.ravel.to_dicts(vector)
        p['weights'] = weights

    def get_loss(self, vector, lossprime):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        p = self.parameters
        if self.lossfunction._step == 0:
            filename = make_filename(self.parent.label,
                                     '-initial-parameters.amp')
            if not os.path.exists(filename):
                # If it exists, must be resuming from checkpoints.
                filename = self.parent.save(filename)

        force_rmse = \
            self.lossfunction.parameters['convergence']['force_rmse']

        if force_rmse is None:
            K_e = self.kernel_e_loss
            result = self.lossfunction.get_loss(vector, p.weights['energy'],
                                                K_e,
                                                lossprime=lossprime)['loss']
        else:
            K_e = self.kernel_e_loss
            K_f = self.kernel_f_cholesky
            result = self.lossfunction.get_loss(vector, p.weights['energy'],
                                                K_e, p.weights['forces'], K_f,
                                                lossprime=lossprime)['loss']
        if self.checkpoints:
            if self.lossfunction._step % self.checkpoints == 0:
                self._log('Saving checkpoint data.')
                if self.checkpoints < 0:
                    path = os.path.join(self.parent.label + '-checkpoints')
                    if not os.path.exists(path):
                        os.mkdir(path)
                    filename = os.path.join(path,
                                            '{}.amp'
                                            .format(int(
                                                self.lossfunction._step)))
                else:
                    filename = make_filename(self.parent.label,
                                             '-checkpoint.amp')
                self.parent.save(filename, overwrite=True)
        if hasattr(self, 'observer'):
            self.observer(self, vector, loss)

        self.lossfunction._step += 1

        if lossprime:
            return result['loss'], result['dloss_dparameters']
        else:
            return result
            # return result['loss']

    def get_lossprime(self, vector):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.  Returns one
        output, the value of the derivative of the loss function with respect
        to model parameters.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        return self.lossfunction.get_loss(vector,
                                          lossprime=True)['dloss_dparameters']

    @property
    def lossfunction(self):
        """Allows the user to set a custom loss function.

        For example,
        >>> from amp.model import LossFunction
        >>> lossfxn = LossFunction(energy_tol=0.0001)
        >>> calc.model.lossfunction = lossfxn

        Parameters
        ----------
        lossfunction : object
            Loss function object, if at all desired by the user.
        """
        return self._lossfunction

    @lossfunction.setter
    def lossfunction(self, lossfunction):
        if hasattr(lossfunction, 'attach_model'):
            lossfunction.attach_model(self)  # Allows access to methods.
        self._lossfunction = lossfunction

    def calculate_atomic_energy(self, afp, index, symbol, hash=None,
                                fp_trainingimages=None, trainingimages=None,
                                kernel=None, sigma=None):
        """
        Given input to the KernelRidge model, output (which corresponds to
        energy) is calculated about the specified atom. The sum of these for
        all atoms is the total energy (in atom-centered mode).

        Parameters
        ----------
        index: int
            Index of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        symbol : str
            Symbol of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        hash : str
            hash of desired image to compute
        kernel : str
            The kernel to be computed in the case that Amp.load is used.
        sigma : float, list, or dict
            Length scale of the Gaussian in the case of RBF, exponential, and
            laplacian kernels. Default is 1. (float) and it computes isotropic
            kernels. Pass a list if you would like to compute anisotropic
            kernels, or a dictionary if you want sigmas for each model.

        Returns
        -------
        atomic_amp_energy : float
            Atomic energy on atom with index=index.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weights = self.parameters.weights

        if len(list(self.kernel_e.keys())) == 0 or hash not in self.kernel_e:
            kij_args = dict(
                    trainingimages=trainingimages,
                    fp_trainingimages=fp_trainingimages,
                    only_features=True
                    )

            # This is needed for both setting the size of parameters to
            # optimize and also to return the kernel for energies
            self.get_energy_kernel(**kij_args)

            if isinstance(self.sigma, dict):
                sigma = self.sigma['energy'][symbol]
            else:
                sigma = self.sigma

            kernel = self.kernel_matrix(
                            np.asarray(afp),
                            self.reference_features_e[symbol],
                            kernel=self.kernel,
                            sigma=sigma
                            )
            atomic_amp_energy = kernel.dot(weights['energy'][symbol])
        else:
            atomic_amp_energy = self.kernel_e[hash][
                        ((index, symbol))].dot(weights['energy'][symbol])
        return atomic_amp_energy

    def energy_from_cholesky(self, symbol=None, afp=None, hash=None,
                             fp_trainingimages=None, trainingimages=None,
                             kernel=None, sigma=None, fingerprints=None,
                             preprocessing=None):
        """
        Given input to the KernelRidge model, output (which corresponds to
        energy) is calculated about the specified atom. The sum of these for
        all atoms is the total energy (in atom-centered mode).

        Parameters
        ---------
        symbol : str
            Atom symbol.
        hash : str
            hash of desired image to compute
        kernel : str
            The kernel to be computed in the case that Amp.load is used.
        sigma : float, list, or dict
            Length scale of the Gaussian in the case of RBF, exponential, and
            laplacian kernels. Default is 1. (float) and it computes isotropic
            kernels. Pass a list if you would like to compute anisotropic
            kernels, or a dictionary if you want sigmas for each model.

        Returns
        -------
        atomic_amp_energy : float
            Atomic energy on atom with index=index.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weights = self.parameters.weights

        if (len(list(self.kernel_e.keys())) == 0 or
           hash not in self.kernel_e or
           len(list(self.kernel_e.values())[0]) == 0):
            try:
                self.reference_features_e
            except AttributeError:

                if preprocessing:
                    _fp_trainingimages = self.preprocess_features(
                            trainingimages,
                            fp_trainingimages)
                    _fp_trainingimages = _fp_trainingimages[0]

                else:
                    _fp_trainingimages = fp_trainingimages

                kij_args = dict(trainingimages=trainingimages,
                                fp_trainingimages=_fp_trainingimages,
                                only_features=True)

                # This is needed for both setting the size of parameters to
                # optimize and also to return the kernel for energies
                self.get_energy_kernel(**kij_args)

            if self.nnpartition is None:
                sigma = self.sigma['energy'][symbol]

                if preprocessing:
                    afp = self.preprocess_features(trainingimages,
                                                   fp_trainingimages,
                                                   afp=afp)

                kernel = self.kernel_matrix(afp,
                                            self.reference_features_e,
                                            feature_symbol=symbol,
                                            kernel=kernel,
                                            sigma=sigma)

                amp_energy = kernel.dot(weights['energy'])

            else:
                sigma = self.sigma['energy'][symbol]
                kernel = self.kernel_matrix(afp,
                                            self.reference_features_e[symbol],
                                            kernel=kernel,
                                            sigma=sigma)

                amp_energy = kernel.dot(weights['energy'][symbol])
        else:
            amp_energy = self.kernel_e[hash].dot(weights['energy'])
        return amp_energy

    def calculate_force(self, index, symbol, component, fingerprintprimes=None,
                        trainingimages=None, t_descriptor=None, sigma=None,
                        hash=None):
        """Given derivative of input to KernelRidge, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        index : integer
            Index of central atom for which the atomic force will be computed.
        symbol : str
            Symbol of central atom for which the atomic force will be computed.
        component : int
            Direction of the force.
        fingerprintprimes : list
            List of fingerprint primes.
        trainingimages : list
            Object or list containing the training set. This is needed when
            performing predictions of unseen data.
        descriptor : object
            Object containing the information about fingerprints.
        hash : str
            Unique key for the image of interest.
        sigma : float, list, or dict
            Length scale of the Gaussian in the case of RBF, exponential, and
            laplacian kernels. Default is 1. (float) and it computes isotropic
            kernels. Pass a list if you would like to compute anisotropic
            kernels, or a dictionary if you want sigmas for each model.

        Returns
        -------
        force : float
            Atomic force on Atom with index=index and symbol=symbol.
        """
        weights = self.parameters.weights
        key = index, symbol

        if len(list(self.kernel_f.keys())) == 0 or hash not in self.kernel_f:
            self.get_forces_kernel(
                    trainingimages=trainingimages,
                    t_descriptor=t_descriptor,
                    only_features=True
                    )

            fprime = 0
            for afp in fingerprintprimes:
                if (index == afp[0] and symbol == afp[1] and
                        component == afp[-1]):
                    fprime += np.array(fingerprintprimes[afp])

            features = self.ref_features_f[symbol][component]

            if isinstance(self.sigma, dict):
                sigma = self.sigma['forces'][symbol][component]
            else:
                sigma = self.sigma

            kernel = self.kernel_matrix(fprime, features, kernel=self.kernel,
                                        sigma=sigma)
            if (self.weights_independent is True and self.cholesky is False):
                force = kernel.dot(weights['forces'][symbol][component])

            elif (self.weights_independent is False and
                    self.cholesky is False):
                force = kernel.dot(weights['forces'][symbol])
        else:
            if (self.weights_independent is True and self.cholesky is False):
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol][component]
                        )
            elif (self.weights_independent is False and self.cholesky is
                    False):
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol]
                        )
        force *= -1.
        return force

    def forces_from_cholesky(self, index, symbol, component,
                             fingerprintprimes=None, trainingimages=None,
                             t_descriptor=None, sigma=None, hash=None,
                             preprocessing=False):
        """Given derivative of input to KernelRidge, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        index : integer
            Index of central atom for which the atomic force will be computed.
        symbol : str
            Symbol of central atom for which the atomic force will be computed.
        component : int
            Direction of the force.
        fingerprintprimes : list
            List of fingerprint primes.
        trainingimages : list
            Object or list containing the training set. This is needed when
            performing predictions of unseen data.
        descriptor : object
            Object containing the information about fingerprints.
        hash : str
            Unique key for the image of interest.
        sigma : float, list, or dict
            Length scale of the Gaussian in the case of RBF, exponential, and
            laplacian kernels. Default is 1. (float) and it computes isotropic
            kernels. Pass a list if you would like to compute anisotropic
            kernels, or a dictionary if you want sigmas for each model.
        preprocessing : bool
            Whether or not the features were preprocessed.

        Returns
        -------
        force : float
            Atomic force on Atom with index=index and symbol=symbol.
        """
        weights = self.parameters.weights
        key = index, symbol

        if len(list(self.kernel_f.keys())) == 0 or hash not in self.kernel_f:
            try:
                self.ref_features_f
            except AttributeError:
                if preprocessing:
                    _fp_trainingimages = self.preprocess_features(
                            trainingimages,
                            t_descriptor,
                            forcetraining=True)
                    t_descriptor = _fp_trainingimages[1]

                self.get_forces_kernel(
                        trainingimages=trainingimages,
                        t_descriptor=t_descriptor,
                        only_features=True)

            fprime = []

            for afp in fingerprintprimes:
                if (index == afp[0] and symbol == afp[1] and
                        component == afp[-1]):
                    fprime.append(np.array(fingerprintprimes[afp]))

            if preprocessing:
                fprime = self.preprocess_features(trainingimages,
                                                  t_descriptor,
                                                  forcetraining=True,
                                                  fprime=fprime,
                                                  component=component)

            if self.sum_rule:
                fprime = np.sum(np.array(fprime), axis=0)

            features = self.ref_features_f[symbol][component]

            sigma = self.sigma['forces'][symbol][component]

            kernel = self.kernel_matrix(
                            fprime,
                            features,
                            kernel=self.kernel,
                            sigma=sigma
                            )

            if (self.weights_independent is True and self.cholesky is True):
                force = kernel.dot(weights['forces'][symbol][component])
        else:
            try:
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][component])
            except KeyError:
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol][component])
        return force

    def kernel_matrix(self, feature, features, feature_symbol=None,
                      kernel='rbf', sigma=1.):
        """This method takes as arguments a feature vector and a string that refers
        to the kernel type used.

        Parameters
        ----------
        feature : list or numpy array
            Single feature.
        features : list or numpy array
            Column vector containing the fingerprints of all atoms in the
            training set.
        feature_symbol : str
            Symbol of chemical element for central atom.
        kernel : str
            Select the kernel to be used. Supported kernels are: 'linear',
            rbf', 'exponential, and 'laplacian'.
        sigma : float, or list.
            Gaussian width. If passed as a list or np.darray, kernel can become
            anisotropic.

        Returns
        -------
        K : array
            The kernel matrix.

        Notes
        -----
        Kernels may differ a lot between them. The kernel_matrix method in this
        class contains algorithms to build the desired matrix. The computation
        of the kernel is done by auxiliary functions that are located at the
        end of the KernelRidge class.
        """
        feature = np.asarray(feature)
        K = []

        call = {'exponential': exponential, 'laplacian': laplacian,
                'rbf': rbf}
        nonlinear_kernels = ['rbf', 'laplacian', 'exponential']

        if kernel == 'linear':
            try:
                features = np.asarray(features)
                for afp in features:
                    K.append(linear(feature, afp))
            except ValueError:
                for symbol, afp in features:
                    afp = np.asarray(afp)
                    K.append(linear(feature, afp,
                             i_symbol=feature_symbol, j_symbol=symbol))

        # All kernels in this control flow share the same structure
        elif kernel in nonlinear_kernels:
            try:
                features = np.asarray(features)
                for afp in features:
                    K.append(call[kernel](feature, afp, sigma=sigma))
            except ValueError:
                for symbol, afp in features:
                    afp = np.asarray(afp)
                    K.append(call[kernel](feature, afp,
                             i_symbol=feature_symbol, j_symbol=symbol,
                             sigma=sigma))

        else:
            raise NotImplementedError('This kernel needs to be coded.')

        return np.asarray(K)


class Raveler(object):
    """Raveler class inspired by neuralnetwork.py

    Takes a weights dictionary created by KernelRidge class and convert it into
    vector and back to dictionaries. This is needed for doing the optimization
    of the loss function.

    Parameters
    ----------
    weights : dict
        Dictionary containing weights per-atom.
    size : int
        Number of elements in the dictionary.
    weights_independent : bool
        Different weights for each atom when training forces.
    """
    def __init__(self, weights, weights_independent=False):
        self.count = 0
        self.weights_keys = []
        self.properties_keys = []
        self.weights_independent = weights_independent
        self.sizes = OrderedDict()

        for prop in weights.keys():
            self.properties_keys.append(prop)
            for key in weights[prop].keys():
                if prop == 'energy':
                    self.weights_keys.append(key)
                    add = len(weights[prop][key])
                    if key not in self.sizes.keys():
                        self.sizes[key] = add
                    self.count += add
                elif prop == 'forces':
                    if self.weights_independent is True:
                        for component in range(3):
                            self.count += len(weights[prop][key][component])
                    else:
                        self.count += len(weights[prop][key])

    def to_vector(self, weights):
        """Convert weights dictionaries to one dimensional vectors.

        Parameters
        ----------
        weights : dict
            Dictionary of weights.

        Returns
        -------
        vector : ndarray
            One-dimensional weight vector to be used by the optimizer.
        """
        vector = []
        for prop in weights.keys():
            if prop == 'energy':
                for key in weights[prop].keys():
                    for element in weights[prop][key]:
                        vector.append(element)
            elif prop == 'forces':
                if self.weights_independent is True:
                    for component in range(3):
                        for key in weights[prop].keys():
                            for element in weights[prop][key][component]:
                                vector.append(element)
                else:
                    for key in weights[prop].keys():
                        vector.append(weights[prop][key])

        vector = np.ravel(vector)
        return vector

    def to_dicts(self, vector):
        """Convert vector of weights back into weights dictionaries.

        Parameters
        ----------
        vector : ndarray
            One-dimensional weight vector.

        Returns
        -------
        weights : dict
            Dictionary of weights.
        """

        assert len(vector) == self.count
        first = 0
        last = 0
        weights = OrderedDict()

        for prop in self.properties_keys:
            weights[prop] = OrderedDict()
            if prop == 'energy':
                for k in self.weights_keys:
                    if k not in weights[prop].keys():
                        step = self.sizes[k]
                        last += step
                        weights[prop][k] = vector[first:last]
                        first += step
            elif prop == 'forces':
                for k in self.weights_keys:
                    if (k not in weights[prop].keys() and
                            self.weights_independent is True):
                        weights[prop][k] = np.zeros((3, self.sizes[k]))
                        for component in range(3):
                            step = self.sizes[k]
                            last += step
                            weights[prop][k][
                                    component] = vector[first:last]
                            first += step
                    elif (k not in weights[prop].keys() and
                            self.weights_independent is False):
                        step = self.sizes[k]
                        last += step
                        weights[prop][k] = vector[first:last]
                        first += step
        return weights


"""
Auxiliary functions to compute different kernels
"""


def linear(feature_i, feature_j, i_symbol=None, j_symbol=None):
    """ Compute a linear kernel

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.

    Returns
    -------
    linear :float
        Linear kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        linear = np.dot(feature_i, feature_j)
        return linear


def rbf(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.
    sigma : float, or list.
        Gaussian width. If passed as a list or np.darray, kernel can become
        anisotropic.

    Returns
    -------
    rbf :float
        RBF kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert(len(sigma) == len(feature_i) and
                   len(sigma) == len(feature_j)), "Length of sigma does not " \
                                                  "match atomic fingerprint " \
                                                  "length."
            sigma = np.array(sigma)
            anisotropic_rbf = np.exp(-(np.sum(np.divide(np.square(
                              np.subtract(feature_i, feature_j)),
                                          (2. * np.square(sigma))))))
            return anisotropic_rbf
        else:
            rbf = np.exp(-(np.linalg.norm(feature_i - feature_j) ** 2.) /
                         (2. * sigma ** 2.))
            return rbf


def exponential(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.):
    """ Compute the exponential kernel

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.
    sigma : float, or list.
        Gaussian width.

    Returns
    -------
    exponential : float
        Exponential kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert(len(sigma) == len(feature_i) and
                   len(sigma) == len(feature_j)), "Length of sigma does not " \
                                                  "match atomic fingerprint " \
                                                  "length."
            sigma = np.array(sigma)
            anisotropic_exp = np.exp(-(np.sqrt(np.sum(np.square(
                          np.divide(np.subtract(feature_i, feature_j),
                                               (2. * np.square(sigma))))))))
            return anisotropic_exp
        else:
            exponential = np.exp(-(np.linalg.norm(feature_i - feature_j)) /
                                 (2. * sigma ** 2))
            return exponential


def laplacian(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.):
    """ Compute the laplacian kernel

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.
    sigma : float
        Gaussian width.

    Returns
    -------
    laplacian : float
        Laplacian kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert(len(sigma) == len(feature_i) and
                   len(sigma) == len(feature_j)), "Length of sigma does not " \
                                                  "match atomic fingerprint " \
                                                  "length."
            sigma = np.array(sigma)

            sum_ij = np.sum(np.square(np.divide(np.subtract(feature_i,
                                                            feature_j),
                                                sigma)))

            anisotropic_lap = np.exp(-(np.sqrt(sum_ij)))
            return anisotropic_lap
        else:
            laplacian = np.exp(-(np.linalg.norm(feature_i - feature_j)) /
                               sigma)
        return laplacian


def ravel_data(train_forces, mode, images, fingerprints, fingerprintprimes):
    """
    Reshapes data of images into lists.

    Parameters
    ----------
    train_forces : bool
        Determining whether forces are also trained or not.
    mode : str
        Can be either 'atom-centered' or 'image-centered'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This is the training set of data. This can also be the
        path to an ASE trajectory (.traj) or database (.db) file. Energies can
        be obtained from any reference, e.g. DFT calculations.
    fingerprints : dict
        Dictionary with images hashes as keys and the corresponding
        fingerprints as values.
    fingerprintprimes : dict
        Dictionary with images hashes as keys and the corresponding fingerprint
        derivatives as values.
    """
    from ase.data import atomic_numbers

    actual_energies = [image.get_potential_energy(apply_constraint=False)
                       for image in images.values()]

    if mode == 'atom-centered':
        num_images_atoms = [len(image) for image in images.values()]
        atomic_numbers = [atomic_numbers[atom.symbol]
                          for image in images.values() for atom in image]

        def ravel_fingerprints(images,
                               fingerprints):
            """
            Reshape fingerprints of images into a list.
            """
            raveled_fingerprints = []
            elements = []
            for hash, image in images.items():
                for index in range(len(image)):
                    elements += [fingerprints[hash][index][0]]
                    raveled_fingerprints += [fingerprints[hash][index][1]]
            elements = sorted(set(elements))
            return elements, raveled_fingerprints

        elements, raveled_fingerprints = ravel_fingerprints(images,
                                                            fingerprints)
    else:
        atomic_positions = [image.positions.ravel()
                            for image in images.values()]

    if train_forces is True:

        actual_forces = \
            [image.get_forces(apply_constraint=False)[index]
             for image in images.values() for index in range(len(image))]

        if mode == 'atom-centered':

            def ravel_neighborlists_and_fingerprintprimes(images,
                                                          fingerprintprimes):
                """
                Reshape neighborlists and fingerprintprimes of images into a
                list and a matrix, respectively.
                """
                # Only neighboring atoms of type II (within the main cell)
                # need to be sent to fortran for force training.
                # All keys in fingerprintprimes are for type II neighborhoods.
                # Also note that each atom is considered as neighbor of
                # itself in fingerprintprimes.
                num_neighbors = []
                raveled_neighborlists = []
                raveled_fingerprintprimes = []
                for hash, image in images.items():
                    for atom in image:
                        selfindex = atom.index
                        selfsymbol = atom.symbol
                        selfneighborindices = []
                        selfneighborsymbols = []
                        for key, derafp in fingerprintprimes[hash].items():
                            # key = (selfindex, selfsymbol, nindex, nsymbol, i)
                            # i runs from 0 to 2. neighbor indices and symbols
                            # should be added just once.
                            if key[0] == selfindex and key[4] == 0:
                                selfneighborindices += [key[2]]
                                selfneighborsymbols += [key[3]]

                        neighborcount = 0
                        for nindex, nsymbol in zip(selfneighborindices,
                                                   selfneighborsymbols):
                            raveled_neighborlists += [nindex]
                            neighborcount += 1
                            for i in range(3):
                                fpprime = fingerprintprimes[hash][(selfindex,
                                                                   selfsymbol,
                                                                   nindex,
                                                                   nsymbol,
                                                                   i)]
                                raveled_fingerprintprimes += [fpprime]
                        num_neighbors += [neighborcount]

                return (num_neighbors,
                        raveled_neighborlists,
                        raveled_fingerprintprimes)

            (num_neighbors,
             raveled_neighborlists,
             raveled_fingerprintprimes) = \
                ravel_neighborlists_and_fingerprintprimes(images,
                                                          fingerprintprimes)
    if mode == 'image-centered':
        if not train_forces:
            return (actual_energies, atomic_positions)
        else:
            return (actual_energies, actual_forces, atomic_positions)
    else:
        if not train_forces:
            return (actual_energies, elements, num_images_atoms,
                    atomic_numbers, raveled_fingerprints)
        else:
            return (actual_energies, actual_forces, elements, num_images_atoms,
                    atomic_numbers, raveled_fingerprints, num_neighbors,
                    raveled_neighborlists, raveled_fingerprintprimes)


def send_data_to_fortran(_fmodules,
                         energy_coefficient,
                         force_coefficient,
                         overfit,
                         train_forces,
                         num_atoms,
                         num_images,
                         actual_energies,
                         actual_forces,
                         atomic_positions,
                         num_images_atoms,
                         atomic_numbers,
                         raveled_fingerprints,
                         num_neighbors,
                         raveled_neighborlists,
                         raveled_fingerprintprimes,
                         model,
                         d):
    """
    Function that sends images data to fortran code. Is used just once on each
    core.
    """
    from ase.data import atomic_numbers as an

    if model.parameters.mode == 'image-centered':
        mode_signal = 1
    elif model.parameters.mode == 'atom-centered':
        mode_signal = 2

    _fmodules.images_props.num_images = num_images
    _fmodules.images_props.actual_energies = actual_energies
    if train_forces:
        _fmodules.images_props.actual_forces = actual_forces

    _fmodules.model_props.energy_coefficient = energy_coefficient
    _fmodules.model_props.force_coefficient = force_coefficient
    _fmodules.model_props.overfit = overfit
    _fmodules.model_props.train_forces = train_forces
    _fmodules.model_props.mode_signal = mode_signal
    if d is None:
        _fmodules.model_props.numericprime = False
    else:
        _fmodules.model_props.numericprime = True
        _fmodules.model_props.d = d

    if model.parameters.mode == 'atom-centered':
        fprange = model.parameters.fprange
        elements = sorted(fprange.keys())
        num_elements = len(elements)
        elements_numbers = [an[elm] for elm in elements]
        min_fingerprints = \
            [[fprange[elm][_][0] for _ in range(len(fprange[elm]))]
             for elm in elements]
        max_fingerprints = [[fprange[elm][_][1]
                             for _
                             in range(len(fprange[elm]))]
                            for elm in elements]
        num_fingerprints_of_elements = \
            [len(fprange[elm]) for elm in elements]

        _fmodules.images_props.num_elements = num_elements
        _fmodules.images_props.elements_numbers = elements_numbers
        _fmodules.images_props.num_images_atoms = num_images_atoms
        _fmodules.images_props.atomic_numbers = atomic_numbers
        if train_forces:
            _fmodules.images_props.num_neighbors = num_neighbors
            _fmodules.images_props.raveled_neighborlists = \
                raveled_neighborlists

        _fmodules.fingerprint_props.num_fingerprints_of_elements = \
            num_fingerprints_of_elements
        _fmodules.fingerprint_props.raveled_fingerprints = raveled_fingerprints
        _fmodules.neuralnetwork.min_fingerprints = min_fingerprints
        _fmodules.neuralnetwork.max_fingerprints = max_fingerprints
        if train_forces:
            _fmodules.fingerprint_props.raveled_fingerprintprimes = \
                raveled_fingerprintprimes
    else:
        _fmodules.images_props.num_atoms = num_atoms
        _fmodules.images_props.atomic_positions = atomic_positions

    # for neural networks only
    """
    if model.parameters['importname'] == '.model.neuralnetwork.NeuralNetwork':

        hiddenlayers = model.parameters.hiddenlayers
        activation = model.parameters.activation

        if model.parameters.mode == 'atom-centered':
            from collections import OrderedDict
            no_layers_of_elements = \
                [3 if isinstance(hiddenlayers[elm], int)
                 else (len(hiddenlayers[elm]) + 2)
                 for elm in elements]
            nn_structure = OrderedDict()
            for elm in elements:
                len_of_fps = len(fprange[elm])
                if isinstance(hiddenlayers[elm], int):
                    nn_structure[elm] = \
                        ([len_of_fps] + [hiddenlayers[elm]] + [1])
                else:
                    nn_structure[elm] = \
                        ([len_of_fps] +
                         [layer for layer in hiddenlayers[elm]] + [1])

            no_nodes_of_elements = [nn_structure[elm][_]
                                    for elm in elements
                                    for _ in range(len(nn_structure[elm]))]

        else:
            num_atoms = model.parameters.num_atoms
            if isinstance(hiddenlayers, int):
                no_layers_of_elements = [3]
            else:
                no_layers_of_elements = [len(hiddenlayers) + 2]
            if isinstance(hiddenlayers, int):
                nn_structure = ([3 * num_atoms] + [hiddenlayers] + [1])
            else:
                nn_structure = ([3 * num_atoms] +
                                [layer for layer in hiddenlayers] + [1])
            no_nodes_of_elements = [nn_structure[_]
                                    for _ in range(len(nn_structure))]

        _fmodules.neuralnetwork.no_layers_of_elements = no_layers_of_elements
        _fmodules.neuralnetwork.no_nodes_of_elements = no_nodes_of_elements
        if activation == 'tanh':
            activation_signal = 1
        elif activation == 'sigmoid':
            activation_signal = 2
        elif activation == 'linear':
            activation_signal = 3
        _fmodules.neuralnetwork.activation_signal = activation_signal
    """
