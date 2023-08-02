import sys
import numpy as np
import threading
import time
import copy
from ase.calculators.calculator import Parameters
from ..utilities import (Logger, ConvergenceOccurred, make_sublists, now,
                         setup_parallel, MetaDict, get_overfit_mask)
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
        """Returns an evaluatable representation of the calculator that can
        be used to re-establish the calculator."""
        # Make sure numpy prints out enough data.
        np.set_printoptions(precision=30, threshold=999999999)
        return self.parameters.tostring()

    def calculate_energy(self, fingerprints):
        """Calculates the model-predicted energy for an image, based on its
        fingerprint.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            self.atomic_energies = []
            energy = 0.0
            for index, (symbol, afp) in enumerate(fingerprints):
                atom_energy = self.calculate_atomic_energy(afp=afp,
                                                           index=index,
                                                           symbol=symbol)
                self.atomic_energies.append(atom_energy)
                energy += atom_energy
        return energy

    def calculate_gc_energy(self, fingerprints, wf, qfp_append):
        """Calculates the model-predicted energy for an image, based on its
        fingerprint and the charge-learning scheme.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        wf : float
            Workfunction of an image.
        qfp_append: list
            List of charge fingerprint of an image, one per atom.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            self.atomic_energies = []
            self.atomic_charges = []
            charge = 0.
            energy = 0.

            for index, (symbol, afp) in enumerate(fingerprints):
                electrostatic_potentials = qfp_append[index]
                charge_afp = afp + electrostatic_potentials
                atom_charge = self.calculate_atomic_charge(afp=charge_afp,
                                                    index=index,
                                                    symbol=symbol,
                                                    potential=electrostatic_potentials)
                self.atomic_charges.append(atom_charge)
                charge += atom_charge
                atom_energy = self.calculate_atomic_electronegtivity(afp=afp,
                                                                     index=index,
                                                                     symbol=symbol,
                                                                     atomic_charge=atom_charge)
                atom_energy += atom_charge*wf
                self.atomic_energies.append(atom_energy)
                energy += atom_energy
        return energy, charge

    def calculate_forces(self, fingerprints, fingerprintprimes):
        """Calculates the model-predicted forces for an image, based on
        derivatives of fingerprints.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        fingerprintprimes : dict
            Dictionary of fingerprint derivatives, where the key is
            a tuple with (index, symbol, neighbor_index, neighbor_symbol,
            direction).
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            forces = np.zeros((len(selfindices), 3))
            for key, derafp in fingerprintprimes.items():
                selfindex, selfsymbol, nindex, nsymbol, direction = key
                afp = fingerprints[nindex][1]
                dforce = self.calculate_force(afp=afp,
                                              derafp=derafp,
                                              nindex=nindex,
                                              nsymbol=nsymbol,
                                              direction=direction)
                forces[selfindex][direction] += dforce
        return forces

    def calculate_gc_forces(self, fingerprints, fingerprintprimes, 
                            wf, qfp_append, qfpprime_append):
        """Calculates the model-predicted forces for an image, based on
        derivatives of fingerprints and the charge learning scheme.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        fingerprintprimes : dict
            Dictionary of fingerprint derivatives, where the key is
            a tuple with (index, symbol, neighbor_index, neighbor_symbol,
            direction).
        wf : float
            Workfunction of an image.
        qfp_append: list
            List of charge fingerprint of an image, one per atom.
        qfpprime_append: list
            List of charge fingerprint derivatives, one per atom.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            forces = np.zeros((len(selfindices), 3))
            for key, derafp in fingerprintprimes.items():
                selfindex, selfsymbol, nindex, nsymbol, direction = key
                electrostatic_potentials = qfp_append[nindex]
                delectrostatic_potentials = qfpprime_append[nindex]
                afp = fingerprints[nindex][1]
                charge_afp = afp + electrostatic_potentials
                if (selfindex == nindex) and (direction == 2):
                    charge_derafp = copy.copy(derafp)
                    charge_derafp += delectrostatic_potentials
                else:
                    charge_derafp = copy.copy(derafp)
                    charge_derafp += [0.] * len(delectrostatic_potentials)
                dforce = self.calculate_electroneg_force(
                              afp=afp,
                              charge_afp=charge_afp,
                              derafp=derafp,
                              charge_derafp=charge_derafp,
                              nindex=nindex,
                              nsymbol=nsymbol,
                              wf=wf,)
                forces[selfindex][direction] += dforce
        return forces

    def calculate_dEnergy_dParameters(self, fingerprints):
        """Calculates a list of floats corresponding to the derivative of
        model-predicted energy of an image with respect to model parameters.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
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


    def calculate_dgcEnergy_dParameters(self, fingerprints, wf, qfp_append):
        """Calculates a list of floats corresponding to the derivative of
        model-predicted energy of an image in charge learning scheme
        with respect to model parameters.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        wf : float
            Workfunction of an image.
        qfp_append: list
            List of charge fingerprint of an image, one per atom.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            denergy_dparameters = None
            dcharge_dparameters = None
            for index, (symbol, afp) in enumerate(fingerprints):
                electrostatic_potentials = qfp_append[index]
                charge_afp = afp + electrostatic_potentials
                temp = self.calculate_electrostatic_dAtomicEnergy_dParameters(
                            afp=afp,
                            afp_charge=charge_afp,
                            index=index,
                            symbol=symbol,
                            wf=wf)
                if denergy_dparameters is None:
                    denergy_dparameters = temp[0]
                else:
                    denergy_dparameters += temp[0]
                if dcharge_dparameters is None:
                    dcharge_dparameters = temp[1]
                else:
                    dcharge_dparameters += temp[1]
        return denergy_dparameters, dcharge_dparameters


    def calculate_numerical_dEnergy_dParameters(self, fingerprints, d=0.00001):
        """Evaluates dEnergy_dParameters using finite 
           difference in charge learning scheme.

        This will trigger two calls to calculate_gc_energy(), with each parameter
        perturbed plus/minus d.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
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

    def calculate_numerical_dgcEnergy_dParameters(self, fingerprints, 
                                                  wf, qfp_append, 
                                                  d=0.00001):
        """Evaluates dEnergy_dParameters using finite difference.

        This will trigger two calls to calculate_energy(), with each parameter
        perturbed plus/minus d.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        wf : float
            Workfunction of an image.
        qfp_append: list
            List of charge fingerprint of an image, one per atom.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            vector = self.vector
            denergy_dparameters = []
            dcharge_dparameters = []
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                eplus = self.calculate_gc_energy(fingerprints, wf, qfp_append)
                vector[_] -= 2 * d
                self.vector = vector
                eminus = self.calculate_gc_energy(fingerprints, wf, qfp_append)
                denergy_dparameters += [(eplus[0] - eminus[0]) / (2 * d)]
                dcharge_dparameters += [(eplus[1] - eminus[1]) / (2 * d)]
                vector[_] += d
                self.vector = vector
            denergy_dparameters = np.array(denergy_dparameters)
            dcharge_dparameters = np.array(dcharge_dparameters)
        return denergy_dparameters, dcharge_dparameters


    def calculate_dForces_dParameters(self, fingerprints, fingerprintprimes):
        """Calculates an array of floats corresponding to the derivative of
        model-predicted atomic forces of an image with respect to model
        parameters.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        fingerprintprimes : dict
            Dictionary of fingerprint derivatives, where the key is
            a tuple with (index, symbol, neighbor_index, neighbor_symbol,
            direction).
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

    def calculate_dgcForces_dParameters(self, fingerprints, fingerprintprimes, 
                                        wf,
                                        qfp_append, qfpprime_append):
        """Calculates an array of floats corresponding to the derivative of
        model-predicted atomic forces of an image in charge learning scheme
        with respect to model parameters.

        Parameters
        ----------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        fingerprintprimes : dict
            Dictionary of fingerprint derivatives, where the key is
            a tuple with (index, symbol, neighbor_index, neighbor_symbol,
            direction).
        wf : float
            Workfunction of an image.
        qfp_append: list
            List of charge fingerprint of an image, one per atom.
        qfpprime_append: list
            List of charge fingerprint derivatives, one per atom.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): None
                                   for selfindex in selfindices
                                   for i in range(3)}


            for key in fingerprintprimes.keys():
                selfindex, selfsymbol, nindex, nsymbol, direction = key
                derafp = fingerprintprimes[key]
                electrostatic_potentials = qfp_append[nindex]
                delectrostatic_potentials = qfpprime_append[nindex]
                afp = fingerprints[nindex][1]
                charge_afp = afp + electrostatic_potentials
                if (selfindex == nindex) and (direction == 2):
                    charge_derafp = copy.copy(derafp)
                    charge_derafp += delectrostatic_potentials
                else:
                    charge_derafp = copy.copy(derafp)
                    charge_derafp += [0.] * len(delectrostatic_potentials)
                temp = self.calculate_electroneg_dForce_dParameters(
                            afp=afp,
                            derafp=derafp,
                            charge_afp=charge_afp, 
                            charge_derafp=charge_derafp,
                            nindex=nindex,
                            nsymbol=nsymbol,
                            wf=wf)

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
        fingerprints : list
            List of fingerprints of an image, one per atom.
        fingerprintprimes : dict
            Dictionary of fingerprint derivatives, where the key is
            a tuple with (index, symbol, neighbor_index, neighbor_symbol,
            direction).
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


    def calculate_numerical_dgcForces_dParameters(self, fingerprints,
                                                fingerprintprimes, 
                                                wf,
                                                qfp_append, qfpprime_append,
                                                d=0.00001, 
                                                ):
        """Evaluates dForces_dParameters using finite difference
        in charge learning scheme. 
        This will trigger two calls to calculate_gc_forces(), 
        with each parameter perturbed plus/minus d.

        Parameters
        ---------
        fingerprints : list
            List of fingerprints of an image, one per atom.
        fingerprintprimes : dict
            Dictionary of fingerprint derivatives, where the key is
            a tuple with (index, symbol, neighbor_index, neighbor_symbol,
            direction).
        wf : float
            Workfunction of an image.
        qfp_append: list
            List of charge fingerprint of an image, one per atom.
        qfpprime_append: list
            List of charge fingerprint derivatives, one per atom.
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
                fplus = self.calculate_gc_forces(fingerprints, fingerprintprimes, wf,
                                                 qfp_append, qfpprime_append)
                vector[_] -= 2 * d
                self.vector = vector
                fminus = self.calculate_gc_forces(fingerprints, fingerprintprimes, wf,
                                                  qfp_append, qfpprime_append)
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

    If parallel is None, it will pull it from the model itself. Only use
    this keyword to override the model's specification.

    Also has parallelization methods built in.

    See self.default_parameters for the default values of parameters
    specified as None.

    Parameters
    ----------
    energy_coefficient : float
        Coefficient of the energy contribution in the loss function.
    charge_coefficient : float
        Coefficient of the charge contribution in the loss function.
    force_coefficient : float
        Coefficient of the force contribution in the loss function.
        Can set to None as shortcut to turn off force training.
    convergence : dict
        Dictionary of keys and values defining convergence.  Keys are
        'energy_rmse', 'energy_maxresid', 'force_rmse', 'force_maxresid', 
        'charge_rmse', and 'charge_maxresid'.
        If 'force_rmse' and 'force_maxresid' are both set to None, force
        training is turned off and force_coefficient is set to None.
    parallel : dict
        Parallel configuration dictionary. Will pull from model itself if
        not specified.
    overfit : float
        Multiplier of atomic neural network weights norm penalty term in
        the loss function.
    raise_ConvergenceOccurred : bool
        If True will raise convergence notice.
    log_losses : bool
        If True will log the loss function value in the log file else will not.
    d : None or float
        If d is None, both loss function and its gradient are calculated
        analytically. If d is a float, then gradient of the loss function is
        calculated by perturbing each parameter plus/minus d.
    weight_duplicates : bool
        If multiple identical images are present in the training set, whether
        to weight them as such in the loss function. E.g., if False, any
        duplicate images will only count as a single image, if True, then a
        triplicate image will weight the same as having that image three
        times. Default is False.
    maxiter: int
        Terminate loss function optimization at a give step/epoch.
    nft_ids: list of length-2 tuples
        If nft_ids is not None, it should have the form of
        [(hash_id_0, index_of_atom_0), (hash_id_1, index_of_atom_1), ...].
        Only force_loss on the atom of given index is included in the image
        with the corresponding hash id.
    """

    default_parameters = {'convergence': {'energy_rmse': 0.001,
                                          'energy_maxresid': None,
                                          'force_rmse': None,
                                          'force_maxresid': None, 
                                          'charge_rmse': 0.0005,
                                          'charge_maxresid': None,}
                          }

    def __init__(self, energy_coefficient=1.0, force_coefficient=0.04,
                 charge_coefficient=10.0, 
                 convergence=None, parallel=None, overfit=0.,
                 raise_ConvergenceOccurred=True, log_losses=True, d=None,
                 weight_duplicates=False, maxiter=100000, nft_ids=None):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        # 'dict' creates a copy; otherwise mutable in class.
        p['convergence'] = dict(self.default_parameters['convergence'])
        if convergence is not None:
            for key, value in convergence.items():
                p['convergence'][key] = value
        p['energy_coefficient'] = energy_coefficient
        p['force_coefficient'] = force_coefficient
        p['charge_coefficient'] = charge_coefficient
        p['overfit'] = overfit
        p['weight_duplicates'] = weight_duplicates
        p['maxiter'] = maxiter
        p['nft_ids'] = nft_ids
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self.log_losses = log_losses
        self.d = d
        self._step = 0
        self._initialized = False
        self._data_sent = False
        self._parallel = parallel

    def attach_model(self, model, images=None, fingerprints=None,
                     fingerprintprimes=None, charge_fp_appends=None, 
                     charge_fpprime_appends=None, log=None):
        """Attach the model to be used to the loss function.
        hashed images, fingerprints, fingerprintprimes, and
        charge training purposed fingerpint sets can optionally be
        specified; this is typically for use in parallelization.

        Parameters
        ----------
        model : object
            Class representing the regression model.
        images : dict
            Dictionary of hashed images to train on.
        fingerprints : dict
            Fingerprints of images to train on.
        fingerprintprimes : dict
            Fingerprint derivatives of images to train on.
        charge_fp_appends: dict
            charge fingerprints of images to train on.
        charge_fpprime_appends : dict
            charge fingerprint derivatives of images to train on.
        """
        self._model = model
        if not hasattr(self._model, 'trainingparameters'):
            self._model.trainingparameters = Parameters()
            self._model.trainingparameters.descriptor = Parameters()
        if images is not None:
            self._model.trainingparameters.images = images
        descriptor = self._model.trainingparameters.descriptor
        if fingerprints is not None:
            descriptor.fingerprints = fingerprints
        if fingerprintprimes is not None:
            descriptor.fingerprintprimes = fingerprintprimes
        if charge_fp_appends is not None:
            self._model.trainingparameters.charge_fp_append = charge_fp_appends
        if charge_fpprime_appends is not None:
            self._model.trainingparameters.charge_fpprime_append = charge_fpprime_appends
        
        if log is not None:
            self.log = log

    def _initialize(self, args=None):
        """Procedures to be run on the first call only, such as establishing
        SSH sessions, etc."""
        if self._initialized is True:
            return

        # Force training is controlled by the force_coefficent key.
        p = self.parameters
        convergence = p['convergence']
        if ((convergence['force_rmse'] is None) and
                (convergence['force_maxresid'] is None)):
            p['force_coefficient'] = None
        if p['force_coefficient'] is None:
            convergence['force_rmse'] = None
            convergence['force_maxresid'] = None

        # Charge training would be turned on if ChargeNeuralNetwork model is detected.
        if self._model.__class__.__name__ != 'ChargeNeuralNetwork':
            p['charge_coefficient'] = None
            convergence['charge_rmse'] = None
            convergence['charge_maxresid'] = None
        else:
            if ((convergence['charge_rmse'] is None) and
                    (convergence['charge_maxresid'] is None)):
                p['charge_coefficient'] = None
            if p['charge_coefficient'] is None:
                convergence['charge_rmse'] = None
                convergence['charge_maxresid'] = None


        if self._parallel is None:
            self._parallel = self._model._parallel
        if not hasattr(self, 'log'):
            self.log = self._model.log
        log = self.log

        if self._parallel['cores'] != 1:
            # Initialize workers and send them parameters.

            python = sys.executable
            workercommand = '%s -m %s' % (python, self.__module__)
            self._sessions = setup_parallel(self._parallel, workercommand,
                                            log, setup_publisher=True)
            n_pids = self._sessions['n_pids']
            images = self._model.trainingparameters.images
            workerkeys = make_sublists(images.keys(), n_pids)
            server = self._sessions['master']
            setup_complete = np.array([False] * n_pids)
            descriptor = self._model.trainingparameters.descriptor
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
                        subimages = {k: images[k] for k in
                                     workerkeys[int(message['id'])]}
                        subimages = MetaDict(subimages)
                        subimages.metadata = images.metadata
                        server.send_pyobj(subimages)
                    elif request == 'fortran':
                        server.send_pyobj(self._model.fortran)
                    elif request == 'modelstring':
                        server.send_pyobj(self._model.tostring())
                    elif request == 'lossfunctionstring':
                        server.send_pyobj(self.parameters.tostring())
                    elif request == 'fingerprints':
                        fingerprints = descriptor.fingerprints
                        server.send_pyobj({k: fingerprints[k] for k in
                                           workerkeys[int(message['id'])]})
                    elif request == 'fingerprintprimes':
                        try:
                            fingerprintprimes = descriptor.fingerprintprimes
                        except AttributeError:
                            server.send_pyobj(None)
                        else:
                            server.send_pyobj({k: fingerprintprimes[k]
                                               for k in
                                               workerkeys[int(message['id'])]})
                    elif request == 'charge_fp_append':
                        try:
                            charge_fp_append = self._model.trainingparameters.charge_fp_append
                            server.send_pyobj({k: charge_fp_append[k] for k in
                                          workerkeys[int(message['id'])]})
                        except AttributeError:
                            server.send_pyobj(None)
                    elif request == 'charge_fpprime_append':
                        try:
                            charge_fpprime_append = self._model.trainingparameters.charge_fpprime_append
                            server.send_pyobj({k: charge_fpprime_append[k] for k in
                                          workerkeys[int(message['id'])]})
                        except AttributeError:
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
            if p.charge_coefficient:
                log(' Loss function convergence criteria:')
                log('  energy_rmse: ' + str(convergence['energy_rmse']))
                log('  energy_maxresid: ' + str(convergence['energy_maxresid']))
                log('  force_rmse: ' + str(convergence['force_rmse']))
                log('  force_maxresid: ' + str(convergence['force_maxresid']))
                log('  charge_rmse: ' + str(convergence['charge_rmse']))
                log('  charge_maxresid: ' + str(convergence['charge_maxresid']))
                log(' Loss function set-up:')
                log('  energy_coefficient: ' + str(p.energy_coefficient))
                log('  force_coefficient: ' + str(p.force_coefficient))
                log('  charge_coefficient: ' + str(p.charge_coefficient))
                log('  overfit: ' + str(p.overfit))
                log('  weight duplicates:' + str(p.weight_duplicates))
                log('\n')
                if p.force_coefficient is None:
                    header = '%5s %19s %12s %12s %12s %12s %12s'
                    log(header %
                        ('', '', '', '', 'Energy', '', 'Charge'))
                    log(header %
                        ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid',
                         'ChargeRMSE', 'MaxResid'))
                    log(header %
                        ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12,
                         '=' * 12, '=' * 12))
                else:
                    header = '%5s %19s %12s %12s %12s %12s %12s %12s %12s'
                    log(header %
                        ('', '', '', '', 'Energy',
                         '', 'Charge', '', 'Force'))
                    log(header %
                        ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid',
                         'ChargeRMSE', 'MaxResid', 'ForceRMSE', 'MaxResid'))
                    log(header %
                        ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12,
                         '=' * 12, '=' * 12, '=' * 12, '=' * 12))
            else:
                log(' Loss function convergence criteria:')
                log('  energy_rmse: ' + str(convergence['energy_rmse']))
                log('  energy_maxresid: ' + str(convergence['energy_maxresid']))
                log('  force_rmse: ' + str(convergence['force_rmse']))
                log('  force_maxresid: ' + str(convergence['force_maxresid']))
                log(' Loss function set-up:')
                log('  energy_coefficient: ' + str(p.energy_coefficient))
                log('  force_coefficient: ' + str(p.force_coefficient))
                log('  overfit: ' + str(p.overfit))
                log('  weight duplicates:' + str(p.weight_duplicates))
                if p.nft_ids:
                    log('  nearsighted force training:' +
                        str(len(p.nft_ids)) + ' nft ids')
                else:
                    log('  nearsighted force training:' + str(p.nft_ids))
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

        self._data_sent = False  # (in case images changed) FIXME Still needed?
        self._initialized = True

    def _send_data_to_fortran(self,):
        """Procedures to be run in fortran mode for a single requested core
        only. Also just on the first call for sending data to fortran modules.
        """
        if self._data_sent is True:
            return

        images = self._model.trainingparameters.images
        num_images = len(images)
        p = self.parameters
        energy_coefficient = p.energy_coefficient
        overfit = p.overfit
        is_nft = [0] * num_images
        nft_indices = [-1] * num_images
        if p.nft_ids:
            hash_ids = [_ for _, __ in p.nft_ids]
            atom_indices = [__ for _, __ in p.nft_ids]
            for ind, hash_id in enumerate(list(images.keys())):
                if hash_id in hash_ids:
                    is_nft[ind] = 1
                    nft_indices[ind] = atom_indices[ind]
        if p.force_coefficient is None:
            train_forces = False
            force_coefficient = 0.
        else:
            train_forces = True
            force_coefficient = p.force_coefficient

        if p.charge_coefficient is None:
            train_charges = False
            charge_coefficient = 0.
        else:
            train_charges = True
            charge_coefficient = p.charge_coefficient


        mode = self._model.parameters.mode
        if mode == 'atom-centered':
            num_atoms = None
        elif mode == 'image-centered':
            raise NotImplementedError('Image-centered mode is not coded yet.')
        descriptor = self._model.trainingparameters.descriptor
        fingerprints = descriptor.fingerprints

        if train_charges:
            qfp_append = self._model.trainingparameters.charge_fp_append
            qfpprime_append = self._model.trainingparameters.charge_fpprime_append
        else:
            qfp_append = None
            qfpprime_append = None


        if hasattr(descriptor, 'fingerprintprimes'):
            fingerprintprimes = descriptor.fingerprintprimes
        else:
            fingerprintprimes = None

        (actual_energies, actual_forces, elements, atomic_positions,
         num_images_atoms, atomic_numbers, raveled_fingerprints, num_neighbors,
         raveled_neighborlists, raveled_fingerprintprimes) = (None,) * 10

        (actual_charges, atomic_charges, image_wfs,
         raveled_charge_fingerprints,
         raveled_charge_fingerprintprimes) = (None,) * 5
        
        value = ravel_data(train_forces,
                           train_charges,
                           mode,
                           images,
                           fingerprints,
                           fingerprintprimes,
                           qfp_append,
                           qfpprime_append,
                           p.weight_duplicates,)

        if mode == 'image-centered':
            if not train_forces:
                (actual_energies, atomic_positions) = value
            else:
                (actual_energies, actual_forces, atomic_positions) = value
        else:
            if not train_charges:
                if not train_forces:
                    (actual_energies, elements, num_images_atoms,
                     atomic_numbers, raveled_fingerprints, image_weights) = value
                else:
                    (actual_energies, actual_forces, elements, num_images_atoms,
                     atomic_numbers, raveled_fingerprints, num_neighbors,
                     raveled_neighborlists, raveled_fingerprintprimes,
                     image_weights) = value
            else:
                if not train_forces:
                    (actual_energies, actual_charges, image_wfs,
                     elements, num_images_atoms,
                     atomic_numbers, raveled_fingerprints, 
                     raveled_charge_fingerprints,
                     image_weights) = value
                else:
                    (actual_energies, actual_forces, actual_charges,
                     image_wfs, elements, num_images_atoms,
                     atomic_numbers, raveled_fingerprints, 
                     raveled_charge_fingerprints, num_neighbors,
                     raveled_neighborlists, raveled_fingerprintprimes,
                     raveled_charge_fingerprintprimes,
                     image_weights) = value

        send_data_to_fortran(fmodules,
                             energy_coefficient,
                             force_coefficient,
                             charge_coefficient,
                             overfit,
                             train_forces,
                             train_charges,
                             num_atoms,
                             num_images,
                             actual_energies,
                             actual_forces,
                             actual_charges,
                             image_wfs,
                             atomic_positions,
                             num_images_atoms,
                             atomic_numbers,
                             raveled_fingerprints,
                             raveled_charge_fingerprints,
                             num_neighbors,
                             raveled_neighborlists,
                             raveled_fingerprintprimes,
                             raveled_charge_fingerprintprimes,
                             self._model,
                             self.d,
                             image_weights,
                             is_nft,
                             nft_indices,)
        self._data_sent = True

    def _cleanup(self):
        """Closes SSH sessions."""
        self._initialized = False
        if not hasattr(self, '_sessions'):
            return
        # Need to properly close socket connections, due to bug in ZMQ with
        # python3. See: https://github.com/zeromq/pyzmq/issues/831
        self._sessions['master'].close()
        self._sessions['publisher'].close()

        for _ in self._sessions['connections']:
            if hasattr(_, 'logout'):
                _.logout()
        del self._sessions['connections']

    def get_loss(self, parametervector, lossprime):
        """Returns the current value of the loss function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.
        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """

        self._step += 1
        self._initialize(args={'lossprime': lossprime, 'd': self.d})
        if self._parallel['cores'] == 1:
            if self._model.fortran:
                self._model.vector = parametervector
                overfit_mask = get_overfit_mask(self._model, parametervector)
                self._send_data_to_fortran()
                (loss, dloss_dparameters, energy_loss, force_loss, charge_loss,
                 energy_maxresid, force_maxresid, charge_maxresid) = \
                    fmodules.calculate_loss(parameters=parametervector,
                                            num_parameters=len(
                                                parametervector),
                                            overfit_mask=overfit_mask,
                                            lossprime=lossprime)
            else:
                loss, dloss_dparameters, energy_loss, force_loss, charge_loss, \
                    energy_maxresid, force_maxresid, charge_maxresid = \
                    self.calculate_loss(parametervector,
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
            charge_loss = results['charge_loss']
            energy_maxresid = results['energy_maxresid']
            force_maxresid = results['force_maxresid']
            charge_maxresid = results['charge_maxresid']

        self.loss, self.energy_loss, self.force_loss, self.charge_loss,\
            self.energy_maxresid, self.force_maxresid, self.charge_maxresid = \
            loss, energy_loss, force_loss, charge_loss, \
            energy_maxresid, force_maxresid, charge_maxresid

        if lossprime:
            self.dloss_dparameters = dloss_dparameters


        if self.raise_ConvergenceOccurred:
            self._model.vector = parametervector
            converged = self.check_convergence(loss,
                                               energy_loss,
                                               force_loss,
                                               charge_loss,
                                               energy_maxresid,
                                               force_maxresid,
                                               charge_maxresid)
            if converged:
                self._cleanup()
                raise ConvergenceOccurred()

        return {'loss': self.loss,
                'dloss_dparameters': (self.dloss_dparameters
                                      if lossprime is True
                                      else dloss_dparameters),
                'energy_loss': self.energy_loss,
                'force_loss': self.force_loss,
                'charge_loss': self.charge_loss,
                'energy_maxresid': self.energy_maxresid,
                'force_maxresid': self.force_maxresid, 
                'charge_maxresid': self.charge_maxresid,}

    def calculate_loss(self, parametervector, lossprime):
        """Method that calculates the loss, derivative of the loss with respect
        to parameters (if requested), and max_residual.

        This is the reference (pure-python) version and should not be called in
        typical runs; the fortran version should be much faster.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.

        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """
        self._model.vector = parametervector
        p = self.parameters
        energyloss = 0.
        forceloss = 0.
        chargeloss = 0.
        energy_maxresid = 0.
        force_maxresid = 0.
        charge_maxresid = 0.
        dloss_dparameters = np.array([0.] * len(parametervector))
        model = self._model
        images = self._model.trainingparameters.images
        descriptor = self._model.trainingparameters.descriptor
        fingerprints = descriptor.fingerprints
        if p.charge_coefficient:
            qfps_append = self._model.trainingparameters.charge_fp_append
            qfpprimes_append = self._model.trainingparameters.charge_fpprime_append
        image_weight = 1.  # for weighting duplicates
          ### Loss for nft images, where only forces on the central atoms
        ### are trained.
        if p.nft_ids:
            hash_ids = [_ for _, __ in p.nft_ids]
            atom_indices = [__ for _, __ in p.nft_ids]
            if p.force_coefficient is not None:
                for ind, hash in enumerate(hash_ids):
                    if hash not in images.keys():
                        continue
                    image = images[hash]
                    no_of_atoms = len(image)
                    if p.weight_duplicates:
                        if hash in images.metadata['duplicates']:
                            image_weight = \
                               float(images.metadata['duplicates'][hash])
                        else:
                            image_weight = 1.
                    fingerprintprimes = descriptor.fingerprintprimes
                    amp_forces = \
                        model.calculate_forces(fingerprints[hash],
                                               fingerprintprimes[hash])
                    actual_forces = image.get_forces(apply_constraint=False)
                    image_forceloss = 0.
                    index = atom_indices[ind]
                    for i in range(3):
                        force_resid = abs(amp_forces[index][i] -
                                          actual_forces[index][i])
                        if force_resid > force_maxresid:
                            force_maxresid = force_resid
                        image_forceloss += force_resid**2
                    image_forceloss /= 3.
                    forceloss += image_weight * image_forceloss

                    if lossprime:
                        if self.d is None:
                            dforces_dparameters = \
                                model.calculate_dForces_dParameters(
                                    fingerprints[hash],
                                    fingerprintprimes[hash])
                        else:
                            dforces_dparameters = \
                                model.calculate_numerical_dForces_dParameters(
                                    fingerprints[hash],
                                    fingerprintprimes[hash],
                                    d=self.d)
                        image_dldp = 0.
                        for i in range(3):
                            image_dldp += (
                                (amp_forces[index][i] -
                                 actual_forces[index][i]) *
                                dforces_dparameters[(index, i)])
                        image_dldp *= (p.force_coefficient * 2. / 3.)
                        dloss_dparameters += image_weight * image_dldp

        ### Loss for regular images, including both energy and force losses
        for hash in images.keys():
            if p.nft_ids is not None:
                hash_ids = [_ for _, __ in p.nft_ids]
                if hash in hash_ids:
                    continue
            if p.weight_duplicates:
                if hash in images.metadata['duplicates']:
                    image_weight = float(images.metadata['duplicates'][hash])
                else:
                    image_weight = 1.
            image = images[hash]
            no_of_atoms = len(image)

            if p.charge_coefficient is None:
                amp_energy = model.calculate_energy(fingerprints[hash])
                actual_energy = image.get_potential_energy(apply_constraint=False)
                residual_per_atom = abs(amp_energy - actual_energy) / \
                    len(image)
                if residual_per_atom > energy_maxresid:
                    energy_maxresid = residual_per_atom
                energyloss += image_weight * residual_per_atom**2
            else:
                wf = image.calc.results['electrode_potential']
                amp_energy, amp_charge = model.calculate_gc_energy(
                                               fingerprints[hash], 
                                               wf,
                                               qfps_append[hash])
                actual_energy = image.get_potential_energy(apply_constraint=False)
                try:
                    actual_charge = image.calc.parameters.sj['excess_electrons'] * (-1.)
                except:
                    actual_charge = image.calc.results['ne']
                residual_per_atom = abs(amp_energy - actual_energy) / \
                    len(image)
                residual_per_atom_charge = abs(amp_charge - actual_charge) / \
                    len(image)
                if residual_per_atom > energy_maxresid:
                    energy_maxresid = residual_per_atom
                energyloss += image_weight * residual_per_atom**2
                if residual_per_atom_charge > charge_maxresid:
                    charge_maxresid = residual_per_atom_charge
                chargeloss += image_weight * residual_per_atom_charge**2
                # Calculates derivative of the loss function with respect to
                # parameters if lossprime is true

            if lossprime:
                if model.parameters.mode == 'image-centered':
                    raise NotImplementedError('This needs to be coded.')
                elif model.parameters.mode == 'atom-centered':
                    if p.charge_coefficient is None:
                        if self.d is None:
                            denergy_dparameters = \
                                model.calculate_dEnergy_dParameters(
                                    fingerprints[hash])
                        else:
                            denergy_dparameters = \
                                model.calculate_numerical_dEnergy_dParameters(
                                    fingerprints[hash], d=self.d)
                        temp = p.energy_coefficient * 2. * \
                            (amp_energy - actual_energy) * \
                            denergy_dparameters / \
                            (no_of_atoms ** 2.)
                        dloss_dparameters += image_weight * temp
                    else:
                        if self.d is None:
                            denergy_dparameters, dcharge_dparameters = \
                                model.calculate_dgcEnergy_dParameters(
                                    fingerprints[hash],
                                    wf,
                                    qfps_append[hash])
                        else:
                            denergy_dparameters, dcharge_dparameters = \
                                model.calculate_numerical_dgcEnergy_dParameters(
                                    fingerprints[hash],
                                    wf,
                                    qfps_append[hash], 
                                    d=self.d,)
                        temp = p.energy_coefficient * 2. * \
                            (amp_energy - actual_energy) * \
                            denergy_dparameters / \
                            (no_of_atoms ** 2.)
                        temp += p.charge_coefficient * 2. * \
                            (amp_charge - actual_charge) * \
                            dcharge_dparameters / \
                            (no_of_atoms ** 2.)
                        dloss_dparameters += image_weight * temp
                     
            if p.force_coefficient is not None:
                fingerprintprimes = descriptor.fingerprintprimes
                if p.charge_coefficient is None:
                    amp_forces = \
                    model.calculate_forces(fingerprints[hash],
                                           fingerprintprimes[hash])
                else:
                    amp_forces = \
                    model.calculate_gc_forces(fingerprints[hash],
                                              fingerprintprimes[hash],
                                              wf,
                                              qfps_append[hash], 
                                              qfpprimes_append[hash])

                actual_forces = image.get_forces(apply_constraint=False)
                image_forceloss = 0.
                for index in range(no_of_atoms):
                    for i in range(3):
                        force_resid = abs(amp_forces[index][i] -
                                          actual_forces[index][i])
                        if force_resid > force_maxresid:
                            force_maxresid = force_resid
                        image_forceloss += force_resid**2
                image_forceloss /= 3. * no_of_atoms  # mean over image
                forceloss += image_weight * image_forceloss

                # Calculates derivative of the loss function with respect to
                # parameters if lossprime is true
                if lossprime:
                    if model.parameters.mode == 'image-centered':
                        raise NotImplementedError('This needs to be coded.')
                    elif model.parameters.mode == 'atom-centered':
                        if p.charge_coefficient is None:
                            if self.d is None:
                                dforces_dparameters = \
                                    model.calculate_dForces_dParameters(
                                        fingerprints[hash],
                                        fingerprintprimes[hash])
                            else:
                                dforces_dparameters = \
                                    model.calculate_numerical_dForces_dParameters(
                                        fingerprints[hash],
                                        fingerprintprimes[hash],
                                        d=self.d)
                        else:
                            if self.d is None:
                                dforces_dparameters = \
                                    model.calculate_dgcForces_dParameters(
                                        fingerprints[hash],
                                        fingerprintprimes[hash],
                                        wf,
                                        qfps_append[hash],
                                        qfpprimes_append[hash])
                            else:
                                dforces_dparameters = \
                                    model.calculate_numerical_dgcForces_dParameters(
                                        fingerprints[hash],
                                        fingerprintprimes[hash],
                                        wf,
                                        qfps_append[hash],
                                        qfpprimes_append[hash],
                                        d=self.d,)
                                        
                        image_dldp = 0.
                        for selfindex in range(no_of_atoms):
                            for i in range(3):
                                image_dldp += (
                                    (amp_forces[selfindex][i] -
                                     actual_forces[selfindex][i]) *
                                    dforces_dparameters[(selfindex, i)])
                        image_dldp *= (p.force_coefficient * 2. / 3. /
                                       no_of_atoms)
                        dloss_dparameters += image_weight * image_dldp

        loss = p.energy_coefficient * energyloss
        if p.force_coefficient is not None:
            loss += p.force_coefficient * forceloss
        if p.charge_coefficient is not None:
            loss += p.charge_coefficient * chargeloss
        dloss_dparameters = np.array(dloss_dparameters)

        # if overfit coefficient is more than zero, overfit contribution to
        # loss and dloss_dparameters is also added.
        if p.overfit > 0.:
            overfitloss = 0.
            overfit_mask = get_overfit_mask(model, parametervector)
            overfit_vector = np.array(parametervector)[overfit_mask]
            for component in overfit_vector:
                overfitloss += component ** 2.
            overfitloss *= p.overfit
            loss += overfitloss
            doverfitloss_dparameters = np.zeros(len(dloss_dparameters))
            doverfitloss_dparameters[overfit_mask] = \
                2 * p.overfit * overfit_vector
            dloss_dparameters += doverfitloss_dparameters

        return loss, dloss_dparameters, energyloss, forceloss, chargeloss,\
            energy_maxresid, force_maxresid, charge_maxresid

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
                   'charge_loss': 0.,
                   'energy_maxresid': 0.,
                   'force_maxresid': 0.,
                   'charge_maxresid': 0.}

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
            results['charge_loss'] += result['charge_loss']
            if result['energy_maxresid'] > results['energy_maxresid']:
                results['energy_maxresid'] = result['energy_maxresid']
            if result['force_maxresid'] > results['force_maxresid']:
                results['force_maxresid'] = result['force_maxresid']
            if result['charge_maxresid'] > results['charge_maxresid']:
                results['charge_maxresid'] = result['charge_maxresid']
            finished[int(message['id'])] = True

        return results

    def check_convergence(self, loss, energy_loss, force_loss, charge_loss,
                          energy_maxresid, force_maxresid, charge_maxresid):
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
        charge_loss : float
            Value of the charge contribution of the loss function.
        energy_maxresid : float
            Maximum energy residual.
        force_maxresid : float
            Maximum force residual.
        charge_maxresid : float
            Maximum charge residual.
        """
        p = self.parameters
        images = self._model.trainingparameters.images
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

        if p.charge_coefficient is not None:
            charge_rmse_converged = True
            if p.convergence['charge_rmse'] is not None:
                charge_rmse = np.sqrt(charge_loss / len(images))
                if charge_rmse > p.convergence['charge_rmse']:
                    charge_rmse_converged = False
            charge_maxresid_converged = True
            if p.convergence['charge_maxresid'] is not None:
                if charge_maxresid > p.convergence['charge_maxresid']:
                    charge_maxresid_converged = False


            if p.force_coefficient is not None:
                if self.log_losses:
                    log('%5i %19s %12.4e %10.4e %1s'
                        ' %10.4e %1s %10.4e %1s %10.4e %1s'
                        ' %10.4e %1s %10.4e %1s' %
                        (self._step, now(), loss, energy_rmse,
                         'C' if energy_rmse_converged else '-',
                         energy_maxresid,
                         'C' if energy_maxresid_converged else '-',
                         charge_rmse,
                         'C' if charge_rmse_converged else '-',
                         charge_maxresid,
                         'C' if charge_maxresid_converged else '-',
                         force_rmse,
                         'C' if force_rmse_converged else '-',
                         force_maxresid,
                         'C' if force_maxresid_converged else '-'))
                if self._step > p.maxiter:
                    return True
                return energy_rmse_converged and energy_maxresid_converged and \
                       charge_rmse_converged and charge_maxresid_converged and \
                       force_rmse_converged and force_maxresid_converged
            else:
                if self.log_losses:
                    log('%5i %19s %12.4e %10.4e %1s'
                        ' %10.4e %1s %10.4e %1s %10.4e %1s' %
                        (self._step, now(), loss, energy_rmse,
                         'C' if energy_rmse_converged else '-',
                         energy_maxresid,
                         'C' if energy_maxresid_converged else '-',
                         charge_rmse,
                         'C' if charge_rmse_converged else '-',
                         charge_maxresid,
                         'C' if charge_maxresid_converged else '-'))
                if self._step > p.maxiter:
                    return True
                return energy_rmse_converged and energy_maxresid_converged and \
                       charge_rmse_converged and charge_maxresid_converged
        else:
            if p.force_coefficient is not None:
                if self.log_losses:
                    log('%5i %19s %12.4e %10.4e %1s %10.4e %1s' 
                        ' %10.4e %1s %10.4e %1s' %
                        (self._step, now(), loss, energy_rmse,
                         'C' if energy_rmse_converged else '-',
                         energy_maxresid,
                         'C' if energy_maxresid_converged else '-',
                         force_rmse,
                         'C' if force_rmse_converged else '-',
                         force_maxresid,
                         'C' if force_maxresid_converged else '-'))
                if self._step > p.maxiter:
                    return True
                return energy_rmse_converged and energy_maxresid_converged and \
                       force_rmse_converged and force_maxresid_converged
            else:
                if self.log_losses:
                    log('%5i %19s %12.4e %10.4e %1s %10.4e %1s' %
                        (self._step, now(), loss, energy_rmse,
                         'C' if energy_rmse_converged else '-',
                         energy_maxresid,
                         'C' if energy_maxresid_converged else '-'))
                if self._step > p.maxiter:
                    return True
                return energy_rmse_converged and energy_maxresid_converged


def calculate_fingerprints_range(fp, images):
    """Calculates the range for the fingerprints corresponding to images,
    stored in fp. fp is a fingerprints object with the fingerprints data
    stored in a dictionary-like object at fp.fingerprints. (Typically this
    is a .utilties.Data structure.) images is a hashed dictionary of atoms
    for which to consider the range.

    In image-centered mode, returns an array of (min, max) values for each
    fingerprint. In atom-centered mode, returns a dictionary of such
    arrays, one per element.
    """
    if fp.parameters.mode == 'image-centered':
        raise NotImplementedError()
    elif fp.parameters.mode == 'atom-centered':
        fprange = {}
        for hash in images.keys():
            imagefingerprints = fp.fingerprints[hash]
            for element, fingerprint in imagefingerprints:
                if element not in fprange:
                    fprange[element] = [[_, _] for _ in fingerprint]
                else:
                    assert len(fprange[element]) == len(fingerprint)
                    for i, ridge in enumerate(fingerprint):
                        if ridge < fprange[element][i][0]:
                            fprange[element][i][0] = ridge
                        elif ridge > fprange[element][i][1]:
                            fprange[element][i][1] = ridge
    for key, value in fprange.items():
        fprange[key] = value
    return fprange

def calculate_charge_fingerprints_range(fp, images, qfp_append):
    """Calculates the range for the fingerprints corresponding to images,
    stored in fp and charge fingerprints stored in qfp_append. The fp is 
    a fingerprints object with the fingerprints data
    stored in a dictionary-like object at fp.fingerprints. (Typically this
    is a .utilties.Data structure.) images is a hashed dictionary of atoms
    for which to consider the range. The qfp_append is a dictionary whose 
    keys are the hashed ids of images.

    In image-centered mode, returns an array of (min, max) values for each
    fingerprint. In atom-centered mode, returns a dictionary of such
    arrays, one per element.
    """
    if fp.parameters.mode == 'image-centered':
        raise NotImplementedError()
    elif fp.parameters.mode == 'atom-centered':
        fprange = {}
        for hash in images.keys():
            imagefingerprints = fp.fingerprints[hash]
            for index, (element, fingerprint) in enumerate(imagefingerprints):
                electrostatic_potentials = qfp_append[hash][index]
                charge_fingerprint = fingerprint + electrostatic_potentials

                if element not in fprange:
                    fprange[element] = [[_, _] for _ in charge_fingerprint]

                else:
                    assert len(fprange[element]) == len(fingerprint) + len(electrostatic_potentials)
                    assert len(fprange[element]) == len(charge_fingerprint)
                    for i, ridge in enumerate(charge_fingerprint):
                        if ridge < fprange[element][i][0]:
                            fprange[element][i][0] = ridge
                        elif ridge > fprange[element][i][1]:
                            fprange[element][i][1] = ridge
    for key, value in fprange.items():
        fprange[key] = value
    return fprange


def ravel_data(train_forces,
               train_charges,
               mode,
               images,
               fingerprints,
               fingerprintprimes,
               qfp_append,
               qfpprime_append,
               weight_duplicates
               ):
    """
    Reshapes data of images into lists.

    Parameters
    ---------
    train_forces : bool
        Determining whether forces are also trained or not.
    train_charges : bool
        Determining whether to use charge learning scheme.
    mode : str
        Can be either 'atom-centered' or 'image-centered'.
    images : dict
        Dictionary of hashed images, from amp.utilities.hash_images.
    fingerprints : dict
        Dictionary with images hashs as keys and the corresponding fingerprints
        as values.
    fingerprintprimes : dict
        Dictionary with images hashs as keys and the corresponding fingerprint
        derivatives as values.
    qfp_append : dict
        Dictionary with images hashs as keys and the corresponding charge
        fingerprints as values.
    qfpprime_append : dict
        Dictionary with images hashs as keys and the corresponding charge 
        fingerprint derivatives as values.
    weight_duplicates : bool
        If multiple identical images are present in the training set, whether
        to weight them as such in the loss function. E.g., if False, any
        duplicate images will only count as a single image, if True, then a
        triplicate image will weight the same as having that image three
        times. Default is False.
    """
    from ase.data import atomic_numbers

    keylist = list(images.keys())  # Make sure order stays constant.

    if train_charges:
        try:
            actual_charges = [-images[key].calc.parameters.sj['excess_electrons']
                           for key in keylist]
        except:
            actual_charges = [-images[key].calc.results['ne']  
                           for key in keylist]
        image_wfs = [images[key].calc.results['electrode_potential'] 
                     for key in keylist]

    actual_energies = [images[key].get_potential_energy(apply_constraint=False)
                       for key in keylist]
    image_weights = [images.metadata['duplicates'].get(key, 1)
                     for key in keylist]
        

    if mode == 'atom-centered':
        num_images_atoms = [len(images[key]) for key in keylist]
        atomic_numbers = [atomic_numbers[atom.symbol]
                          for key in keylist for atom in images[key]]

        def ravel_fingerprints(images,
                               fingerprints,
                               qfp_append,
                               qfpprime_append,
                               train_charges,
                               force_training):
            """
            Reshape fingerprints of images into a list.
            """
            raveled_fingerprints = []
            elements = []
            raveled_charge_fingerprints = []
            raveled_charge_fingerprintprimes = []
            for hash in keylist:
                image = images[hash]
                for index in range(len(image)):
                    elements += [fingerprints[hash][index][0]]
                    raveled_fingerprints += [fingerprints[hash][index][1]]
                    if train_charges:
                        raveled_charge_fingerprints += \
                            [fingerprints[hash][index][1] + qfp_append[hash][index]]
                        if force_training:
                            symfucs = [0.] * len(fingerprints[hash][index][1])
                            raveled_charge_fingerprintprimes += \
                                [symfucs + qfpprime_append[hash][index]]
            elements = sorted(set(elements))
            # Could also work without images:
#            raveled_fingerprints = [afp
#                    for hash, value in fingerprints.items()
#                    for (element, afp) in value]
            return elements, raveled_fingerprints, \
                   raveled_charge_fingerprints, \
                   raveled_charge_fingerprintprimes

        elements, raveled_fingerprints, raveled_charge_fingerprints, \
        raveled_charge_fingerprintprimes \
            = ravel_fingerprints(images, fingerprints,
                                 qfp_append, qfpprime_append,
                                 train_charges, train_forces)
        if len(raveled_fingerprints) != 0:
            # Add zero paddings to fingerprints
            len_of_fps = [len(_) for _ in raveled_fingerprints]
            max_len_of_fps = max(len_of_fps)
            raveled_fingerprints = [_ + [0]*(max_len_of_fps-len(_))
                                    for _ in raveled_fingerprints]
    else:
        atomic_positions = [images[key].positions.ravel() for key in keylist]

    if train_forces is True:

        actual_forces = \
            [images[key].get_forces(apply_constraint=False)[index]
             for key in keylist for index in range(len(images[key]))]

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
                for hash in keylist:
                    image = images[hash]
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
            if len(raveled_fingerprintprimes) != 0:
                # Add zero paddings to fingerprintprimes
                len_of_fp_primes = [len(_) for _ in raveled_fingerprintprimes]
                max_len_of_fp_primes = max(len_of_fp_primes)
                raveled_fingerprintprimes = \
                                    [_ + [0] * (max_len_of_fp_primes - len(_))
                                     for _ in raveled_fingerprintprimes]
    if mode == 'image-centered':
        if not train_forces:
            return (actual_energies, atomic_positions)
        else:
            return (actual_energies, actual_forces, atomic_positions)
    else:
        if not train_charges:
            if not train_forces:
                return (actual_energies, elements, num_images_atoms,
                        atomic_numbers, raveled_fingerprints, image_weights)
            else:
                return (actual_energies, actual_forces, elements, num_images_atoms,
                        atomic_numbers, raveled_fingerprints, num_neighbors,
                        raveled_neighborlists, raveled_fingerprintprimes,
                        image_weights)
        else:
            if not train_forces:
                return (actual_energies, actual_charges, image_wfs,
                        elements, num_images_atoms,
                        atomic_numbers, raveled_fingerprints, 
                        raveled_charge_fingerprints,
                        image_weights)
            else:
                return (actual_energies, actual_forces, actual_charges,
                        image_wfs, elements, num_images_atoms,
                        atomic_numbers, raveled_fingerprints, 
                        raveled_charge_fingerprints,
                        num_neighbors,
                        raveled_neighborlists, raveled_fingerprintprimes,
                        raveled_charge_fingerprintprimes,
                        image_weights)

def send_data_to_fortran(_fmodules,
                         energy_coefficient,
                         force_coefficient,
                         charge_coefficient,
                         overfit,
                         train_forces,
                         train_charges,
                         num_atoms,
                         num_images,
                         actual_energies,
                         actual_forces,
                         actual_charges,
                         image_wfs,
                         atomic_positions,
                         num_images_atoms,
                         atomic_numbers,
                         raveled_fingerprints,
                         raveled_charge_fingerprints,
                         num_neighbors,
                         raveled_neighborlists,
                         raveled_fingerprintprimes,
                         raveled_charge_fingerprintprimes,
                         model,
                         d,
                         image_weights,
                         is_nft,
                         nft_indices,
                         ):
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
    _fmodules.images_props.actual_charges = actual_charges
    _fmodules.images_props.is_nft = is_nft
    _fmodules.images_props.nft_indices = nft_indices
    _fmodules.images_props.image_wfs = image_wfs
    if train_forces:
        _fmodules.images_props.actual_forces = actual_forces
    _fmodules.images_props.image_weights = image_weights

    _fmodules.model_props.energy_coefficient = energy_coefficient
    _fmodules.model_props.force_coefficient = force_coefficient
    _fmodules.model_props.charge_coefficient = charge_coefficient
    _fmodules.model_props.overfit = overfit
    _fmodules.model_props.train_forces = train_forces
    _fmodules.model_props.train_charges = train_charges
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
        if len(min_fingerprints) != 0:
            # Add zero paddings to min_fingerprints and max_fingerprints
            len_of_min_fps = [len(_) for _ in min_fingerprints]
            max_len_of_min_fps = max(len_of_min_fps)
            min_fingerprints = [_ + [0]*(max_len_of_min_fps - len(_))
                                for _ in min_fingerprints]
            max_fingerprints = [_ + [0]*(max_len_of_min_fps - len(_))
                                for _ in max_fingerprints]
            num_fingerprints_of_elements = \
                [len(fprange[elm]) for elm in elements]
        num_fingerprints_of_elements = \
            [len(fprange[elm]) for elm in elements]
 
        if train_charges:
            charge_fprange = model.parameters.charge_fprange
            elements = sorted(charge_fprange.keys())
            num_elements = len(elements)
            charge_min_fingerprints = \
                [[charge_fprange[elm][_][0] for _ in range(len(charge_fprange[elm]))]
                 for elm in elements]
            charge_max_fingerprints = [[charge_fprange[elm][_][1]
                                 for _
                                 in range(len(charge_fprange[elm]))]
                                 for elm in elements]
            if len(charge_min_fingerprints) != 0:
                len_of_charge_min_fps = [len(_) for _ in charge_min_fingerprints]
                max_len_of_charge_min_fps = max(len_of_charge_min_fps)
                charge_min_fingerprints = [_ + [0]*(max_len_of_charge_min_fps - len(_))
                                    for _ in charge_min_fingerprints]
                charge_max_fingerprints = [_ + [0]*(max_len_of_charge_min_fps - len(_))
                                    for _ in charge_max_fingerprints]
                charge_num_fingerprints_of_elements = \
                    [len(charge_fprange[elm]) for elm in elements]

            _fmodules.chargeneuralnetwork.charge_min_fingerprints = charge_min_fingerprints
            _fmodules.chargeneuralnetwork.charge_max_fingerprints = charge_max_fingerprints
            charge_num_fingerprints_of_elements = \
                [len(charge_fprange[elm]) for elm in elements]
            _fmodules.fingerprint_props.num_charge_fps_of_elements = \
                charge_num_fingerprints_of_elements
            _fmodules.fingerprint_props.raveled_charge_fps = \
                raveled_charge_fingerprints
            _fmodules.fingerprint_props.raveled_charge_fpprimes = \
                raveled_charge_fingerprintprimes

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
        if train_charges:
            _fmodules.chargeneuralnetwork.en_min_fingerprints = min_fingerprints
            _fmodules.chargeneuralnetwork.en_max_fingerprints = max_fingerprints

        if train_forces:
            _fmodules.fingerprint_props.raveled_fingerprintprimes = \
                raveled_fingerprintprimes
    else:
        _fmodules.images_props.num_atoms = num_atoms
        _fmodules.images_props.atomic_positions = atomic_positions

    # For neural networks only.
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

    if model.parameters['importname'] == '.model.chargeneuralnetwork.ChargeNeuralNetwork':

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

        _fmodules.chargeneuralnetwork.en_no_layers_of_elements = no_layers_of_elements
        _fmodules.chargeneuralnetwork.en_no_nodes_of_elements = no_nodes_of_elements
        if activation == 'tanh':
            activation_signal = 1
        elif activation == 'sigmoid':
            activation_signal = 2
        elif activation == 'linear':
            activation_signal = 3
        _fmodules.chargeneuralnetwork.en_activation_signal = activation_signal

        charge_hiddenlayers = model.parameters.charge_hiddenlayers
        charge_activation = model.parameters.charge_activation

        if model.parameters.mode == 'atom-centered':
            from collections import OrderedDict
            charge_no_layers_of_elements = \
                [3 if isinstance(charge_hiddenlayers[elm], int)
                 else (len(charge_hiddenlayers[elm]) + 2)
                 for elm in elements]
            charge_nn_structure = OrderedDict()
            for elm in elements:
                charge_len_of_fps = len(charge_fprange[elm])
                if isinstance(charge_hiddenlayers[elm], int):
                    charge_nn_structure[elm] = \
                        ([charge_len_of_fps] + [charge_hiddenlayers[elm]] + [1])
                else:
                    charge_nn_structure[elm] = \
                        ([charge_len_of_fps] +
                         [layer for layer in charge_hiddenlayers[elm]] + [1])

            charge_no_nodes_of_elements = [charge_nn_structure[elm][_]
                                    for elm in elements
                                    for _ in range(len(charge_nn_structure[elm]))]
        else:
            num_atoms = model.parameters.num_atoms
            if isinstance(charge_hiddenlayers, int):
                charge_no_layers_of_elements = [3]
            else:
                charge_no_layers_of_elements = [len(charge_hiddenlayers) + 2]
            if isinstance(charge_hiddenlayers, int):
                charge_nn_structure = ([3 * num_atoms] + [charge_hiddenlayers] + [1])
            else:
                charge_nn_structure = ([3 * num_atoms] +
                                [layer for layer in charge_hiddenlayers] + [1])
            charge_no_nodes_of_elements = [charge_nn_structure[_]
                                    for _ in range(len(charge_nn_structure))]

        _fmodules.chargeneuralnetwork.charge_no_layers_of_elements = charge_no_layers_of_elements
        _fmodules.chargeneuralnetwork.charge_no_nodes_of_elements = charge_no_nodes_of_elements
        if charge_activation == 'tanh':
            charge_activation_signal = 1
        elif charge_activation == 'sigmoid':
            charge_activation_signal = 2
        elif charge_activation == 'linear':
            charge_activation_signal = 3
        _fmodules.chargeneuralnetwork.charge_activation_signal = charge_activation_signal

def calculate_charge_fp(z_position, image_eta, interface, image_potential):
    if z_position < interface:
        electrostatic_potential = 0. 
    else:
        electrostatic_potential = image_potential * \
            np.exp(-1. * image_eta * (z_position - interface))
    return electrostatic_potential



def calculate_charge_fpprime(z_position, image_eta, interface, image_potential):
    if z_position < interface:
        delectrostatic_potential = 0.
    else:
        delectrostatic_potential = -1. * image_potential * image_eta * \
            (z_position - interface) * \
            np.exp(-1. * image_eta * (z_position - interface))
    return delectrostatic_potential
