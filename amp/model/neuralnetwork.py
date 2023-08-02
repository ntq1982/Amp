import os
import numpy as np
from collections import OrderedDict
from scipy.optimize import fmin
from ase.calculators.calculator import Parameters

from . import LossFunction, calculate_fingerprints_range, Model
from ..regression import Regressor
from ..utilities import Logger, hash_images, make_filename
from .. import Amp


class NeuralNetwork(Model):
    """Class that implements a basic feed-forward neural network.

    Parameters
    ----------
    hiddenlayers : dict
        Dictionary of chemical element symbols and architectures of their
        corresponding hidden layers of the conventional neural network. Number
        of nodes of last layer is always one corresponding to energy.  However,
        number of nodes of first layer is equal to three times number of atoms
        in the system in the case of no descriptor, and is equal to length of
        symmetry functions of the descriptor. Can be fed using tuples as:

        >>> hiddenlayers = (3, 2,)

        for example, in which a neural network with two hidden layers, the
        first one having three nodes and the second one having two nodes is
        assigned (to the whole atomic system in the no descriptor case, and to
        each chemical element in the atom-centered mode). When setting only one
        hidden layer, the dictionary can be fed as:

        >>> hiddenlayers = (3,)

        In the atom-centered mode, neural network for each species can be
        assigned seperately, as:

        >>> hiddenlayers = {"O":(3,5), "Au":(5,6)}

        for example.

    activation : str
        Assigns the type of activation funtion. "linear" refers to linear
        function, "tanh" refers to tanh function, and "sigmoid" refers to
        sigmoid function.
    weights : dict
        In the atom-centered mode, keys correspond to chemical elements and
        values are dictionaries with layer keys and network weight two
        dimensional arrays as values. Arrays are set up to connect node i in
        the previous layer with node j in the current layer with indices
        w[i,j]. The last value for index i corresponds to bias.  In the case of
        no descriptor, keys correspond to layers and values are two dimensional
        arrays of network weight.  If weights is not given, arrays will be
        randomly generated.
    scalings : dict
        In the case of no descriptor, keys are "intercept" and "slope" and
        values are real numbers. In the fingerprinting scheme, keys correspond
        to chemical elements and values are dictionaries with "intercept" and
        "slope" keys and real number values. If scalings is None, it will
        be guessed based on the input range of energies.
    fprange : dict
        Range of fingerprints of each chemical species.  Should be fed as
        a dictionary of chemical species and a list of minimum and maximun,
        e.g.:

        >>> fprange={"Pd": [0.31, 0.59], "O":[0.56, 0.72]}

    mode : str
        Can be either 'atom-centered' or 'image-centered'.
    version : object
        Version of this class.
    regressor : object
        Regressor object for finding best fit model parameters, e.g. by loss
        function optimization via amp.regression.Regressor.
    lossfunction : object
        Loss function object.
    prescale : bool
        If True, will start with a simple single-parameter (per element)
        regression to find the best-fit values for the scaling parameters. This
        portion of the code is not parallelized, but can greatly improve the
        likelihood of and path to convergence, as this is the most sensitive
        parameter.  Ignored if scalings are explicitly supplied.
    fortran : bool
        Can optionally shut off fortran, primarily for debugging.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training step.  By default only
        the last checkpoint is retained; to keep all checkpoints make the
        frequency negative instead of positive.  Specify None for no
        checkpoints.  Note that checkpoints can be used to resume a training
        run simply by resubmitting the same script; if a checkpoint file is
        found it will be used.
    randomseed : None or int
        Seed to use in random number generator for making initial guess of
        training parameters. You should only set this if you want a script to
        always generate the same "random" initial parameters. This primarily
        useful for unit tests or benchmarking.

    .. note:: Dimensions of weight two dimensional arrays should be consistent
              with hiddenlayers.

    Raises
    ------
        RuntimeError, NotImplementedError
    """

    def __init__(self, hiddenlayers=(5, 5), activation='tanh', weights=None,
                 scalings=None, fprange=None, mode=None, version=None,
                 regressor=None, lossfunction=None, fortran=True,
                 checkpoints=100, randomseed=None, prescale=False):

        # Version check, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use NeuralNetwork'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               'newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # The parameters dictionary contains the minimum information
        # to produce a compatible model; e.g., one that gives
        # the identical energy (and/or forces) when fed a fingerprint.
        p = self.parameters = Parameters()
        p.importname = '.model.neuralnetwork.NeuralNetwork'
        p.version = version
        p.hiddenlayers = hiddenlayers
        p.weights = weights
        p.scalings = scalings
        p.fprange = fprange
        p.activation = activation
        p.mode = mode

        # Checking that the activation function is given correctly:
        if activation not in ['linear', 'tanh', 'sigmoid']:
            _ = ('Unknown activation function %s; must be one of '
                 '"linear", "tanh", or "sigmoid".' % activation)
            raise NotImplementedError(_)

        self.regressor = regressor
        self.parent = None  # Will hold a reference to main Amp instance.
        self.lossfunction = lossfunction
        self.fortran = fortran
        self.checkpoints = checkpoints
        if self.lossfunction is None:
            self.lossfunction = LossFunction()
        self.randomseed = randomseed
        self.prescale = prescale

    def fit(self,
            trainingimages,
            descriptor,
            log,
            parallel,
            only_setup=False,
            ):
        """Fit the model parameters such that the fingerprints can be used to
        describe the energies in trainingimages. log is the logging object.
        descriptor is a descriptor object, as would be in calc.descriptor.

        Parameters
        ----------
        trainingimages : dict
            Hashed dictionary of training images.
        descriptor : object
            Class representing local atomic environment.
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        parallel: dict
            Parallel configuration dictionary. Takes the same form as in
            amp.Amp.
        only_setup : bool
            only_setup is primarily for debugging.  It initializes all
            variables but skips the last line of starting the regressor.
        """

        self._log = log
        self._load_from_checkpoints()  # if present; resume training

        # Set all parameters and report to logfile.
        self._parallel = parallel

        if self.regressor is None:
            self.regressor = Regressor()

        p = self.parameters

        tp = self.trainingparameters = Parameters()
        tp.images = trainingimages
        tp.descriptor = descriptor

        if p.mode is None:
            p.mode = descriptor.parameters.mode
        else:
            assert p.mode == descriptor.parameters.mode
        log('Regression in %s mode.' % p.mode)

        if 'fprange' not in p or p.fprange is None:
            log('Calculating new fingerprint range; this range is part '
                'of the model.')
            p.fprange = calculate_fingerprints_range(descriptor,
                                                     trainingimages)

        if p.mode == 'atom-centered':
            # If hiddenlayers is a tuple/list, convert to a dictionary.
            if not hasattr(p.hiddenlayers, 'keys'):
                p.hiddenlayers = {element: p.hiddenlayers
                                  for element in p.fprange.keys()}

        log('Hidden-layer structure:')
        if p.mode == 'image-centered':
            log(' %s' % str(p.hiddenlayers))
        elif p.mode == 'atom-centered':
            for item in p.hiddenlayers.items():
                log(' %2s: %s' % item)

        if p.weights is None:
            log('Initializing with random weights.')
            self.randomize(scalings=False, seed=self.randomseed)
        else:
            log('Initial weights already present.')

        if p.scalings is None and self.prescale is True:
            self.log('Finding good guesses for scaling intercepts...',
                     tic='prescale')
            self.prescale_intercepts(trainingimages)
            self.log(' Atomic energies found:')
            for element in self.parameters.scalings.keys():
                self.log('{:2s}: {:14.4f}'.format(
                    element, self.parameters.scalings[element]['intercept']))
            self.log('...prescale complete.', toc='prescale')

        if p.scalings is None:
            log('Initializing with random scalings.')
            self.randomize(weights=False, trainingimages=trainingimages,
                           seed=self.randomseed)
        else:
            log('Initial scalings already present.')

        if only_setup:
            return

        # Regress the model.
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    def prescale_intercepts(self, trainingimages):
        """Calculates a reasonable guess for the per-atom energy, by regressing
        a one-parameter-per-element model (per-atom energies), and uses this as
        the scaling energy. This is a reasonably inefficient
        residual-minimization technique, but doesn't typically bottleneck the
        code. The slopes are not explicitly calculated, and take as w """

        # This still assumes slope=1 is a goood guess.
        # But in units of eV, it's not bad.

        elements = self.parameters.fprange.keys()

        def get_rmse_per_atom(scalings_list):
            scalings = {element: scaling for
                        (element, scaling) in zip(elements, scalings_list)}
            calc = OffsetCalculator(scalings=scalings)
            msea = 0.  # mean-squared (error per atom)
            for image in trainingimages.values():
                true_energy = image.get_potential_energy(
                        apply_constraint=False)
                predicted_energy = calc.get_potential_energy(image)
                msea += ((true_energy - predicted_energy) / len(image))**2
            return np.sqrt(msea)  # root-mean-squared (error per atom)

        answer = fmin(get_rmse_per_atom, x0=[1.]*len(elements))
        scaling_intercepts = {element: intercept for element, intercept in
                              zip(elements, answer)}
        p = self.parameters
        p.scalings = {}
        for element in elements:
            p.scalings[element] = {'intercept': scaling_intercepts[element],
                                   'slope': 1.}

    def randomize(self, trainingimages=None, weights=True, scalings=True,
                  seed=None):
        """Randomizes the model parameters (i.e., re-initializes them);
        this is typically used just before training.

        Parameters
        ----------
        trainingimages : list
            List of ASE atoms objects that are being trained. This is only
            needed if scalings is True.
        weights : bool
            If False, do not randomize weights.
        scalings : bool
            If False, do not randomize scalings.
        seed : None or int
            Seed to use in random number generator for making initial guess of
            training parameters. You should only set this if you want a script
            to always generate the same "random" initial parameters. This
            primarily useful for unit tests or benchmarking.
        """
        p = self.parameters
        if weights:
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                len_of_fps = {element: len(p.fprange[element])
                              for element in p.fprange.keys()}
                p.weights = get_random_weights(hiddenlayers=p.hiddenlayers,
                                               activation=p.activation,
                                               len_of_fps=len_of_fps,
                                               seed=seed,)
        if scalings:
            if p.mode == 'image-centered':
                raise NotImplementedError('Need to code.')
            elif p.mode == 'atom-centered':
                p.scalings = get_initial_scalings(trainingimages, p.activation,
                                                  p.fprange.keys())

    @property
    def forcetraining(self):
        """Returns true if forcetraining is turned on (as determined by
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
        """Access to get or set the model parameters (weights, scaling for
        each network) as a single vector, useful in particular for
        regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights, p.scalings)
        return self.ravel.to_vector(weights=p.weights, scalings=p.scalings)

    @vector.setter
    def vector(self, vector):
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights, p.scalings)
        weights, scalings = self.ravel.to_dicts(vector)
        p['weights'] = weights
        p['scalings'] = scalings

    def get_loss(self, vector, lossprime):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.lossfunction._step == 0:
            filename = make_filename(self.parent.label,
                                     '-initial-parameters.amp')
            if not os.path.exists(filename):
                # If it exists, must be resuming from checkpoints.
                filename = self.parent.save(filename)
        result = self.lossfunction.get_loss(vector, lossprime=lossprime)
        if self.checkpoints:
            if self.lossfunction._step % self.checkpoints == 0:
                self._log('Saving checkpoint data.')
                if self.checkpoints < 0:
                    path = os.path.join(self.parent.label + '-checkpoints')
                    if not os.path.exists(path):
                        os.mkdir(path)
                    filename = os.path.join(
                        path, '{}.amp'.format(int(self.lossfunction._step)))
                else:
                    filename = make_filename(self.parent.label,
                                             '-checkpoint.amp')
                self.parent.save(filename, overwrite=True)
        if hasattr(self, 'observer'):
            self.observer(self, vector, loss)

        if lossprime:
            return result['loss'], result['dloss_dparameters']
        else:
            return result['loss']

    def _load_from_checkpoints(self):
        """If checkpoints are present, this will load from them and therefore
        resume a previous training run."""
        # Check default checkpoint pattern.
        filename = make_filename(self.parent.label, '-checkpoint.amp')
        dirname = os.path.join(self.parent.label + '-checkpionts')
        if os.path.exists(filename):
            calc = Amp.load(filename, logging=False)
        elif os.path.exists(dirname):
            checkpoints = os.listdir(dirname)
            last = sorted([int(_[:-4]) for _ in checkpoints])[-1]
            filename = os.path.join(dirname, '{}.amp'.format(last))
            calc = Amp.load(filename, logging=False)
            self.lossfunction._step = last
        else:
            return  # No checkpoints present; run normally.
        self._log('Found checkpoint file: {}.'.format(filename))
        p = calc.model.parameters
        self.parameters = p
        self._log('Loaded last neural network parameters from checkpoint. '
                  'Resuming training run.')

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

    def calculate_atomic_energy(self, afp, index, symbol,):
        """
        Given input to the neural network, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        Parameters
        ---------
        afp : list
            Atomic fingerprints in the form of a list to be used as input to
            the neural network.
        index: int
            Index of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        symbol : str
            Symbol of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).

        Returns
        -------
        float
            Energy.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        scaling = self.parameters.scalings[symbol]
        outputs = calculate_nodal_outputs(self.parameters, afp, symbol,)
        atomic_amp_energy = scaling['slope'] * \
            float(outputs[len(outputs) - 1]) + \
            scaling['intercept']

        return atomic_amp_energy

    def calculate_force(self, afp, derafp,
                        direction,
                        nindex=None, nsymbol=None,):
        """Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        afp : list
            Atomic fingerprints in the form of a list to be used as input to
            the neural network.
        derafp : list
            Derivatives of atomic fingerprints in the form of a list to be used
            as input to the neural network.
        direction : int
            Direction of force.
        nindex : int
            Index of the atom at which force is acting.  (only used in
            the atom-centered mode)
        nsymbol : str
            Symbol of the atom at which force is acting.  (only used
            in the atom-centered mode)

        Returns
        -------
        float
            Force.
        """

        scaling = self.parameters.scalings[nsymbol]
        outputs = calculate_nodal_outputs(self.parameters, afp, nsymbol,)
        dOutputs_dInputs = calculate_dOutputs_dInputs(self.parameters, derafp,
                                                      outputs, nsymbol,)

        force = float((scaling['slope'] *
                       dOutputs_dInputs[len(dOutputs_dInputs) - 1][0]))
        # Force is multiplied by -1, because it is -dE/dx and not dE/dx.
        return -force

    def calculate_dAtomicEnergy_dParameters(self, afp, index=None,
                                            symbol=None):
        """Returns the derivative of energy square error with respect to
        variables.

        Parameters
        ----------
        afp : list
            Atomic fingerprints in the form of a list to be used as input to
            the neural network.
        index : int
            Index of the atom for which atomic energy is calculated (only used
            in the atom-centered mode)
        symbol : str
            Symbol of the atom for which atomic energy is calculated (only used
            in the atom-centered mode)

        Returns
        -------
        list of float
            The value of the derivative of energy square error with respect to
            variables.
        """
        p = self.parameters
        scaling = p.scalings[symbol]
        # W dictionary initiated.
        W = {}
        for elm in p.weights.keys():
            W[elm] = {}
            weight = p.weights[elm]
            for _ in range(len(weight)):
                W[elm][_ + 1] = np.delete(weight[_ + 1], -1, 0)
        W = W[symbol]

        dAtomicEnergy_dParameters = np.zeros(self.ravel.count)
        dAtomicEnergy_dWeights, dAtomicEnergy_dScalings = \
            self.ravel.to_dicts(dAtomicEnergy_dParameters)

        outputs = calculate_nodal_outputs(self.parameters, afp, symbol,)
        ohat, D, delta = calculate_ohat_D_delta(self.parameters, outputs, W)

        dAtomicEnergy_dScalings[symbol]['intercept'] = 1.
        dAtomicEnergy_dScalings[symbol][
            'slope'] = float(outputs[len(outputs) - 1])
        for k in range(1, len(outputs)):
            dAtomicEnergy_dWeights[symbol][k] = float(scaling['slope']) * \
                np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)

        dAtomicEnergy_dParameters = \
            self.ravel.to_vector(
                dAtomicEnergy_dWeights, dAtomicEnergy_dScalings)

        return dAtomicEnergy_dParameters

    def calculate_dForce_dParameters(self, afp, derafp,
                                     direction,
                                     nindex=None, nsymbol=None,):
        """Returns the derivative of force square error with respect to
        variables.

        Parameters
        ----------
        afp : list
            Atomic fingerprints in the form of a list to be used as input to
            the neural network.
        derafp : list
            Derivatives of atomic fingerprints in the form of a list to be used
            as input to the neural network.
        direction : int
            Direction of force.
        nindex : int
            Index of the atom at which force is acting.  (only used in
            the atom-centered mode)
        nsymbol : str
            Symbol of the atom at which force is acting.  (only used
            in the atom-centered mode)

        Returns
        -------
        list of float
            The value of the derivative of force square error with respect to
            variables.
        """
        p = self.parameters
        scaling = p.scalings[nsymbol]
        activation = p.activation
        # W dictionary initiated.
        W = {}
        for elm in p.weights.keys():
            W[elm] = {}
            weight = p.weights[elm]
            for _ in range(len(weight)):
                W[elm][_ + 1] = np.delete(weight[_ + 1], -1, 0)
        W = W[nsymbol]

        dForce_dParameters = np.zeros(self.ravel.count)

        dForce_dWeights, dForce_dScalings = \
            self.ravel.to_dicts(dForce_dParameters)

        outputs = calculate_nodal_outputs(self.parameters, afp, nsymbol,)
        ohat, D, delta = calculate_ohat_D_delta(self.parameters, outputs, W)
        dOutputs_dInputs = calculate_dOutputs_dInputs(self.parameters, derafp,
                                                      outputs, nsymbol,)

        N = len(outputs) - 2
        dD_dInputs = {}
        for k in range(1, N + 2):
            # Calculating coordinate derivative of D matrix
            dD_dInputs[k] = np.zeros(shape=(np.size(outputs[k]),
                                            np.size(outputs[k])))
            for j in range(np.size(outputs[k])):
                if activation == 'linear':  # linear
                    dD_dInputs[k][j, j] = 0.
                elif activation == 'tanh':  # tanh
                    dD_dInputs[k][j, j] = \
                        - 2. * outputs[k][0, j] * dOutputs_dInputs[k][j]
                elif activation == 'sigmoid':  # sigmoid
                    dD_dInputs[k][j, j] = dOutputs_dInputs[k][j] - \
                        2. * outputs[k][0, j] * dOutputs_dInputs[k][j]
        # Calculating coordinate derivative of delta
        dDelta_dInputs = {}
        # output layer
        dDelta_dInputs[N + 1] = dD_dInputs[N + 1]
        # hidden layers
        temp1 = {}
        temp2 = {}
        for k in range(N, 0, -1):
            temp1[k] = np.dot(W[k + 1], delta[k + 1])
            temp2[k] = np.dot(W[k + 1], dDelta_dInputs[k + 1])
            dDelta_dInputs[k] = \
                np.dot(dD_dInputs[k], temp1[k]) + np.dot(D[k], temp2[k])
        # Calculating coordinate derivative of ohat and
        # coordinates weights derivative of atomic_output
        dOhat_dInputs = {}
        dOutput_dInputsdWeights = {}
        for k in range(1, N + 2):
            dOhat_dInputs[k - 1] = [None] * (1 + len(dOutputs_dInputs[k - 1]))
            bound = len(dOutputs_dInputs[k - 1])
            for count in range(bound):
                dOhat_dInputs[k - 1][count] = dOutputs_dInputs[k - 1][count]
            dOhat_dInputs[k - 1][count + 1] = 0.
            dOutput_dInputsdWeights[k] = \
                np.dot(np.matrix(dOhat_dInputs[k - 1]).T,
                       np.matrix(delta[k]).T) + \
                np.dot(np.matrix(ohat[k - 1]).T,
                       np.matrix(dDelta_dInputs[k]).T)

        for k in range(1, N + 2):
            dForce_dWeights[nsymbol][k] = float(scaling['slope']) * \
                dOutput_dInputsdWeights[k]
        dForce_dScalings[nsymbol]['slope'] = dOutputs_dInputs[N + 1][0]
        dForce_dParameters = self.ravel.to_vector(dForce_dWeights,
                                                  dForce_dScalings)
        # force is multiplied by -1, because it is -dE/dx and not dE/dx.
        dForce_dParameters *= -1.

        return dForce_dParameters

# Auxiliary functions #########################################################


def calculate_nodal_outputs(parameters, afp, symbol,):
    """
    Given input to the neural network, output (which corresponds to energy)
    is calculated about the specified atom. The sum of these for all
    atoms is the total energy (in atom-centered mode).

    Parameters
    ----------
    parameters : dict
        ASE dictionary object.
    afp : list
        Atomic fingerprints in the form of a list to be used as input to the
        neural network.
    symbol : str
        Symbol of the atom for which atomic energy is calculated (only used in
        the atom-centered mode)

    Returns
    -------
    dict
        Outputs of neural network nodes
    """

    _afp = np.array(afp).copy()
    hiddenlayers = parameters.hiddenlayers[symbol]
    weight = parameters.weights[symbol]
    activation = parameters.activation

    fprange = parameters.fprange[symbol]
    # Scale the fingerprints to be in [-1, 1] range.
    for _ in range(np.shape(_afp)[0]):
        if (fprange[_][1] - fprange[_][0]) > (10.**(-8.)):
            _afp[_] = -1.0 + 2.0 * ((_afp[_] - fprange[_][0]) /
                                    (fprange[_][1] - fprange[_][0]))

    # Calculate node values.
    o = {}  # node values
    layer = 1  # input layer
    net = {}  # excitation
    ohat = {}  # ohat is the nodal output matrix o concatenated by 1 for biases

    len_of_afp = len(_afp)
    # a temp variable is defined to construct the output matix o
    temp = np.zeros((1, len_of_afp + 1))
    for _ in range(len_of_afp):
        temp[0, _] = _afp[_]
    temp[0, len(_afp)] = 1.0
    ohat[0] = temp
    net[1] = np.dot(ohat[0], weight[1])
    if activation == 'linear':
        o[1] = net[1]  # linear activation
    elif activation == 'tanh':
        o[1] = np.tanh(net[1])  # tanh activation
    elif activation == 'sigmoid':  # sigmoid activation
        o[1] = 1. / (1. + np.exp(-net[1]))
    temp = np.zeros((1, np.shape(o[1])[1] + 1))
    bound = np.shape(o[1])[1]
    for _ in range(bound):
        temp[0, _] = o[1][0, _]
    temp[0, np.shape(o[1])[1]] = 1.0
    ohat[1] = temp
    for hiddenlayer in hiddenlayers[1:]:
        layer += 1
        net[layer] = np.dot(ohat[layer - 1], weight[layer])
        if activation == 'linear':
            o[layer] = net[layer]  # linear activation
        elif activation == 'tanh':
            o[layer] = np.tanh(net[layer])  # tanh activation
        elif activation == 'sigmoid':
            # sigmoid activation
            o[layer] = 1. / (1. + np.exp(-net[layer]))
        temp = np.zeros((1, np.size(o[layer]) + 1))
        bound = np.size(o[layer])
        for _ in range(bound):
            temp[0, _] = o[layer][0, _]
        temp[0, np.size(o[layer])] = 1.0
        ohat[layer] = temp
    layer += 1  # output layer
    net[layer] = np.dot(ohat[layer - 1], weight[layer])
    if activation == 'linear':
        o[layer] = net[layer]  # linear activation
    elif activation == 'tanh':
        o[layer] = np.tanh(net[layer])  # tanh activation
    elif activation == 'sigmoid':
        # sigmoid activation
        o[layer] = 1. / (1. + np.exp(-net[layer]))

    del hiddenlayers, weight, ohat, net

    len_of_afp = len(_afp)
    temp = np.zeros((1, len_of_afp))
    for _ in range(len_of_afp):
        temp[0, _] = _afp[_]
    o[0] = temp

    return o


def calculate_dOutputs_dInputs(parameters, derafp, outputs, nsymbol,):
    """
    Calculates the derivative of the neural network nodes with respect
    to the inputs.

    Parameters
    ----------
    parameters : dict
        ASE dictionary object.
    derafp : list
        Derivatives of atomic fingerprints in the form of a list to be used as
        input to the neural network.
    outputs : dict
        Outputs of neural network nodes.
    nsymbol : str
        Symbol of the atom for which atomic energy is calculated (only used in
        the atom-centered mode)

    Returns
    -------
    dict
        Derivatives of outputs of neural network nodes w.r.t.  inputs.
    """

    _derafp = np.array(derafp).copy()
    hiddenlayers = parameters.hiddenlayers[nsymbol]
    weight = parameters.weights[nsymbol]
    activation = parameters.activation

    fprange = parameters.fprange[nsymbol]
    # Scaling derivative of fingerprints.
    for _ in range(len(_derafp)):
        if (fprange[_][1] - fprange[_][0]) > (10.**(-8.)):
            _derafp[_] = 2.0 * (_derafp[_] / (fprange[_][1] - fprange[_][0]))

    dOutputs_dInputs = {}  # node values
    dOutputs_dInputs[0] = _derafp
    layer = 0  # input layer
    for hiddenlayer in hiddenlayers[0:]:
        layer += 1
        temp = np.dot(np.matrix(dOutputs_dInputs[layer - 1]),
                      np.delete(weight[layer], -1, 0))
        dOutputs_dInputs[layer] = [None] * np.size(outputs[layer])
        bound = np.size(outputs[layer])
        for j in range(bound):
            if activation == 'linear':  # linear function
                dOutputs_dInputs[layer][j] = float(temp[0, j])
            elif activation == 'sigmoid':  # sigmoid function
                dOutputs_dInputs[layer][j] = float(temp[0, j]) * \
                    float(outputs[layer][0, j] * (1. - outputs[layer][0, j]))
            elif activation == 'tanh':  # tanh function
                dOutputs_dInputs[layer][j] = float(temp[0, j]) * \
                    float(1. - outputs[layer][0, j] * outputs[layer][0, j])
    layer += 1  # output layer
    temp = np.dot(np.matrix(dOutputs_dInputs[layer - 1]),
                  np.delete(weight[layer], -1, 0))
    if activation == 'linear':  # linear function
        dOutputs_dInputs[layer] = float(temp)
    elif activation == 'sigmoid':  # sigmoid function
        dOutputs_dInputs[layer] = \
            float(outputs[layer] * (1. - outputs[layer]) * temp)
    elif activation == 'tanh':  # tanh function
        dOutputs_dInputs[layer] = \
            float((1. - outputs[layer] * outputs[layer]) * temp)

    dOutputs_dInputs[layer] = [dOutputs_dInputs[layer]]

    return dOutputs_dInputs


def calculate_ohat_D_delta(parameters, outputs, W):
    """Calculates extra matrices ohat, D, delta needed in mathematical
    manipulations.

    Notations are consistent with those of 'Rojas, R. Neural Networks
    - A Systematic Introduction.  Springer-Verlag, Berlin, first edition 1996'

    Parameters
    ----------
    parameters : dict
        ASE dictionary object.
    outputs : dict
        Outputs of neural network nodes.
    W : dict
        The same as weight dictionary, but the last rows associated with biases
        are deleted in W.
    """

    activation = parameters.activation

    N = len(outputs) - 2  # number of hiddenlayers
    D = {}
    for k in range(N + 2):
        D[k] = np.zeros(shape=(np.size(outputs[k]), np.size(outputs[k])))
        for j in range(np.size(outputs[k])):
            if activation == 'linear':  # linear
                D[k][j, j] = 1.
            elif activation == 'sigmoid':  # sigmoid
                D[k][j, j] = float(outputs[k][0, j]) * \
                    float((1. - outputs[k][0, j]))
            elif activation == 'tanh':  # tanh
                D[k][j, j] = float(1. - outputs[k][0, j] * outputs[k][0, j])
    # Calculating delta
    delta = {}
    # output layer
    delta[N + 1] = D[N + 1]
    # hidden layers

    for k in range(N, 0, -1):  # backpropagate starting from output layer
        delta[k] = np.dot(D[k], np.dot(W[k + 1], delta[k + 1]))
    # Calculating ohat
    ohat = {}
    for k in range(1, N + 2):
        bound = np.size(outputs[k - 1])
        ohat[k - 1] = np.zeros(shape=(1, bound + 1))
        for j in range(bound):
            ohat[k - 1][0, j] = outputs[k - 1][0, j]
        ohat[k - 1][0, bound] = 1.0

    return ohat, D, delta


def get_random_weights(hiddenlayers, activation,
                       len_of_fps=None, no_of_atoms=None, seed=None):
    """Generates random weight arrays from variables.

    hiddenlayers: dict
        Dictionary of chemical element symbols and architectures of their
        corresponding hidden layers of the conventional neural network. Number
        of nodes of last layer is always one corresponding to energy.  However,
        number of nodes of first layer is equal to three times number of atoms
        in the system in the case of no descriptor, and is equal to length of
        symmetry functions in the atom-centered mode. Can be fed as:

        >>> hiddenlayers = (3, 2,)

        for example, in which a neural network with two hidden
        layers, the first one having three nodes and the
        second one having two nodes is assigned (to the whole
        atomic system in the case of no descriptor, and to
        each chemical element in the atom-centered mode).  In
        the atom-centered mode, neural network for each
        species can be assigned seperately, as:

        >>> hiddenlayers = {"O":(3,5), "Au":(5,6)}

        for example.
    activation : str
        Assigns the type of activation funtion. "linear" refers to linear
        function, "tanh" refers to tanh function, and "sigmoid" refers to
        sigmoid function.
    len_of_fps : dict
        Length of fingerprints of each element, e.g:

        >>> len_of_fps={"O": 20, "Pd":20}

    no_of_atoms : int
        Number of atoms in atomic systems; used only in the case of no
        descriptor.
    seed : None or int
        Seed to use in random number generator for making initial guess of
        training parameters. You should only set this if you want a script to
        always generate the same "random" initial parameters. This primarily
        useful for unit tests or benchmarking.

    Returns
    -------
    float
        weights
    """

    rs = np.random.RandomState(seed=seed)
    weight = {}
    nn_structure = {}

    if no_of_atoms is not None:  # image-centered mode

        if isinstance(hiddenlayers, int):
            nn_structure = ([3 * no_of_atoms] + [hiddenlayers] + [1])
        else:
            nn_structure = (
                [3 * no_of_atoms] +
                [layer for layer in hiddenlayers] + [1])
        weight = {}
        # Instead try Andrew Ng coursera approach. +/- epsilon
        # epsilon = sqrt(6./(n_i + n_o))
        # where the n's are the number of input and output nodes.
        # Note: need to double that here with the math below.
        epsilon = np.sqrt(6. / (nn_structure[0] +
                                nn_structure[1]))
        normalized_arg_range = 2. * epsilon
        weight[1] = rs.random_sample((3 * no_of_atoms + 1,
                                      nn_structure[1])) * \
            normalized_arg_range - \
            normalized_arg_range / 2.
        len_of_hiddenlayers = len(list(nn_structure)) - 3
        for layer in range(len_of_hiddenlayers):
            epsilon = np.sqrt(6. / (nn_structure[layer + 1] +
                                    nn_structure[layer + 2]))
            normalized_arg_range = 2. * epsilon
            weight[layer + 2] = rs.random_sample(
                (nn_structure[layer + 1] + 1,
                 nn_structure[layer + 2])) * \
                normalized_arg_range - normalized_arg_range / 2.

        epsilon = np.sqrt(6. / (nn_structure[-2] +
                                nn_structure[-1]))
        normalized_arg_range = 2. * epsilon
        weight[len(list(nn_structure)) - 1] = \
            rs.random_sample((nn_structure[-2] + 1, 1)) \
            * normalized_arg_range - normalized_arg_range / 2.

        if False:  # This seemed to be setting all biases to zero?
            len_of_weight = len(weight)
            for _ in range(len_of_weight):  # biases
                size = weight[_ + 1][-1].size
                for __ in range(size):
                    weight[_ + 1][-1][__] = 0.

    else:
        elements = hiddenlayers.keys()
        for element in sorted(elements):
            _len_of_fps = len_of_fps[element]
            if isinstance(hiddenlayers[element], int):
                nn_structure[element] = ([_len_of_fps] +
                                         [hiddenlayers[element]] + [1])
            else:
                nn_structure[element] = (
                    [_len_of_fps] +
                    [layer for layer in hiddenlayers[element]] + [1])
            weight[element] = {}
            # Instead try Andrew Ng coursera approach. +/- epsilon
            # epsilon = sqrt(6./(n_i + n_o))
            # where the n's are the number of input and output nodes.
            # Note: need to double that here with the math below.
            epsilon = np.sqrt(6. / (nn_structure[element][0] +
                                    nn_structure[element][1]))
            normalized_arg_range = 2. * epsilon
            weight[element][1] = (rs.random_sample(
                (_len_of_fps + 1, nn_structure[element][1])) *
                normalized_arg_range - normalized_arg_range / 2.)
            len_of_hiddenlayers = len(list(nn_structure[element])) - 3
            for layer in range(len_of_hiddenlayers):
                epsilon = np.sqrt(6. / (nn_structure[element][layer + 1] +
                                        nn_structure[element][layer + 2]))
                normalized_arg_range = 2. * epsilon
                weight[element][layer + 2] = rs.random_sample(
                    (nn_structure[element][layer + 1] + 1,
                     nn_structure[element][layer + 2])) * \
                    normalized_arg_range - normalized_arg_range / 2.

            epsilon = np.sqrt(6. / (nn_structure[element][-2] +
                                    nn_structure[element][-1]))
            normalized_arg_range = 2. * epsilon
            weight[element][len(list(nn_structure[element])) - 1] = \
                rs.random_sample((nn_structure[element][-2] + 1, 1)) \
                * normalized_arg_range - normalized_arg_range / 2.

            if False:  # This seemed to be setting all biases to zero?
                len_of_weight = len(weight[element])
                for _ in range(len_of_weight):  # biases
                    size = weight[element][_ + 1][-1].size
                    for __ in range(size):
                        weight[element][_ + 1][-1][__] = 0.

    return weight


def get_initial_scalings(images, activation, elements=None, seed=None):
    """Generates initial scaling matrices, such that the range of activation is
    scaled to the range of actual energies.

    images : dict
        ASE atoms objects (the training set).
    activation: str
        Assigns the type of activation funtion. "linear" refers to linear
        function, "tanh" refers to tanh function, and "sigmoid" refers to
        sigmoid function.
    elements: list of str
        List of atom symbols; used in the atom-centered mode only.

    Returns
    -------
    float
        scalings
    """
    hashs = list(images.keys())

    no_of_images = len(hashs)

    max_act_energy = max(image.get_potential_energy(apply_constraint=False)
                         for image in images.values())
    min_act_energy = min(image.get_potential_energy(apply_constraint=False)
                         for image in images.values())

    for count in range(no_of_images):
        hash = hashs[count]
        image = images[hash]
        no_of_atoms = len(image)
        if image.get_potential_energy(apply_constraint=False) == \
                max_act_energy:
            no_atoms_of_max_act_energy = no_of_atoms
        if image.get_potential_energy(apply_constraint=False) == \
                min_act_energy:
            no_atoms_of_min_act_energy = no_of_atoms

    max_act_energy_per_atom = max_act_energy / no_atoms_of_max_act_energy
    min_act_energy_per_atom = min_act_energy / no_atoms_of_min_act_energy

    scaling = {}

    if elements is None:  # image-centered mode

        scaling = {}
        if activation == 'sigmoid':  # sigmoid activation function
            scaling['intercept'] = min_act_energy_per_atom
            scaling['slope'] = (max_act_energy_per_atom -
                                min_act_energy_per_atom)
        elif activation == 'tanh':  # tanh activation function
            scaling['intercept'] = (max_act_energy_per_atom +
                                    min_act_energy_per_atom) / 2.
            scaling['slope'] = (max_act_energy_per_atom -
                                min_act_energy_per_atom) / 2.
        elif activation == 'linear':  # linear activation function
            scaling['intercept'] = (max_act_energy_per_atom +
                                    min_act_energy_per_atom) / 2.
            scaling['slope'] = (10. ** (-10.)) * \
                (max_act_energy_per_atom -
                 min_act_energy_per_atom) / 2.

    else:  # atom-centered mode

        for element in elements:
            scaling[element] = {}
            if activation == 'sigmoid':  # sigmoid activation function
                scaling[element]['intercept'] = min_act_energy_per_atom
                scaling[element]['slope'] = (max_act_energy_per_atom -
                                             min_act_energy_per_atom)
            elif activation == 'tanh':  # tanh activation function
                scaling[element]['intercept'] = (max_act_energy_per_atom +
                                                 min_act_energy_per_atom) / 2.
                scaling[element]['slope'] = (max_act_energy_per_atom -
                                             min_act_energy_per_atom) / 2.
            elif activation == 'linear':  # linear activation function
                scaling[element]['intercept'] = (max_act_energy_per_atom +
                                                 min_act_energy_per_atom) / 2.
                scaling[element]['slope'] = (10. ** (-10.)) * \
                                            (max_act_energy_per_atom -
                                             min_act_energy_per_atom) / 2.

    return scaling


class Raveler:
    """Class to ravel and unravel variable values into a single vector.

    This is used for feeding into the optimizer. Feed in a list of dictionaries
    to initialize the shape of the transformation. Note no data is saved in the
    class; each time it is used it is passed either the dictionaries or vector.
    The dictionaries for initialization should be two levels deep.

    weights, scalings are the variables to ravel and unravel
    """

    def __init__(self, weights, scalings):

        self.count = 0
        self.weightskeys = []
        self.scalingskeys = []
        for key1 in sorted(weights.keys()):  # element
            for key2 in sorted(weights[key1].keys()):  # layer
                value = weights[key1][key2]
                self.weightskeys.append({'key1': key1,
                                         'key2': key2,
                                         'shape': np.array(value).shape,
                                         'size': np.array(value).size})
                self.count += np.array(weights[key1][key2]).size
        for key1 in sorted(scalings.keys()):  # element
            for key2 in sorted(scalings[key1].keys()):  # slope / intercept
                self.scalingskeys.append({'key1': key1,
                                          'key2': key2})
                self.count += 1
        self.vector = np.zeros(self.count)

    def to_vector(self, weights, scalings):
        """Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with."""

        vector = np.zeros(self.count)
        count = 0

        for k in self.weightskeys:
            lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in self.scalingskeys:
            vector[count] = scalings[k['key1']][k['key2']]
            count += 1

        return vector

    def to_dicts(self, vector):
        """Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel."""

        assert len(vector) == self.count
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()

        for k in self.weightskeys:
            if k['key1'] not in weights.keys():
                weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = matrix.flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            weights[k['key1']][k['key2']] = matrix.tolist()
            count += k['size']
        for k in self.scalingskeys:
            if k['key1'] not in scalings.keys():
                scalings[k['key1']] = OrderedDict()
            scalings[k['key1']][k['key2']] = vector[count]
            count += 1
        return weights, scalings

# Analysis tools ##############################################################


class NodePlot:

    """Creates plots to visualize the output of the nodes in the neural
    networks.

    initialize with a calculator that has parameters; e.g. a trained
    calculator or else one in which fit has been called with the setup_only
    flag turned on.

    Call with the 'plot' method, which takes as argment a list of images
    """

    def __init__(self, calc):
        self.calc = calc
        self.data = {}  # For accumulating the data.
        # Local imports; these are not package-wide dependencies.
        from matplotlib import pyplot
        from matplotlib.backends.backend_pdf import PdfPages
        self.pyplot = pyplot
        self.PdfPages = PdfPages

    def plot(self, images, filename='nodeplot.pdf'):
        """ Creates a plot of the output of each node, as a violin plot.
        """
        calc = self.calc
        log = Logger('develop.log')
        images = hash_images(images, log=log)
        calc.descriptor.calculate_fingerprints(images=images,
                                               parallel={'cores': 1},
                                               log=log,
                                               calculate_derivatives=False)
        for hash in images.keys():
            fingerprints = calc.descriptor.fingerprints[hash]
            for fp in fingerprints:
                outputs = calculate_nodal_outputs(calc.model.parameters,
                                                  afp=fp[1],
                                                  symbol=fp[0])
                self._accumulate(symbol=fp[0], output=outputs)

        self._finalize_table()

        with self.PdfPages(filename) as pdf:
            for symbol in self.data.keys():
                fig = self._makefig(symbol)
                pdf.savefig(fig)
                self.pyplot.close(fig)

    def _makefig(self, symbol, save=False):
        """Makes a figure for one element."""

        fig = self.pyplot.figure(figsize=(8.5, 11.0))
        lm = 0.1
        rm = 0.05
        bm = 0.05
        tm = 0.05
        vg = 0.05
        numplots = 1 + self.data[symbol]['header'][-1][0]
        axwidth = 1. - lm - rm
        axheight = (1. - bm - tm - (numplots - 1) * vg) / numplots

        d = self.data[symbol]
        for layer in range(1 + d['header'][-1][0]):
            ax = fig.add_axes((lm,
                               1. - tm - axheight - (axheight + vg) * layer,
                               axwidth, axheight))
            indices = [_ for _, label in enumerate(d['header'])
                       if label[0] == layer]
            sub = d['table'][:, indices]
            ax.violinplot(dataset=sub, positions=range(len(indices)))
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlim(-0.5, len(indices) - 0.5)
            ax.set_ylabel('Layer %i' % layer)
        ax.set_xlabel('node')
        fig.text(0.5, 1. - 0.5 * tm, 'Node outputs for %s' % symbol,
                 ha='center', va='center')

        if save:
            fig.savefig(save)
        return fig

    def _accumulate(self, symbol, output):
        """Accumulates the data for the symbol."""
        data = self.data
        layerkeys = list(output.keys())  # Correspond to layers.
        layerkeys.sort()

        if symbol not in data:
            # Create headers, structure.
            data[symbol] = {'header': [],
                            'table': []}
            for layerkey in layerkeys:
                v = output[layerkey]
                v = v.reshape(v.size).tolist()
                data[symbol]['header'].extend([(layerkey, _) for _ in
                                              range(len(v))])
        # Add as a row to data table.
        row = []
        for layerkey in layerkeys:
            v = output[layerkey]
            v = v.reshape(v.size).tolist()
            row.extend(v)
        data[symbol]['table'].append(row)

    def _finalize_table(self):
        """Converts the data table into a numpy array."""
        for symbol in self.data:
            self.data[symbol]['table'] = np.array(self.data[symbol]['table'])


class OffsetCalculator:
    """A calculator in which energy is only a sum of one-body atomic terms,
    with an energy per element type. This is used to calculate the scaling
    parameters."""

    def __init__(self, scalings):
        """Scalings is a dictionary [element:energy]."""
        self.scalings = scalings

    def get_potential_energy(self, atoms):
        """Calculate a crude potential energy."""
        energy = 0.
        for atom in atoms:
            energy += self.scalings[atom.symbol]
        return energy
