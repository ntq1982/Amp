.. _Building:

==================================
Building modules
==================================

Amp is designed to be modular, so if you think you have a great descriptor scheme or machine-learning model, you can try it out.
This page describes how to add your own modules; starting with the bare-bones requirements to make it work, and building up with how to construct it so it integrates with respect to parallelization, etc.

----------------------------------
Descriptor: minimal requirements
----------------------------------

To build your own descriptor, it needs to have certain minimum requirements met, in order to play with *Amp*. The below code illustrates these minimum requirements::

    from ase.calculators.calculator import Parameters

    class MyDescriptor(object):

        def __init__(self, parameter1, parameter2):
            self.parameters = Parameters({'mode': 'atom-centered',})
            self.parameters.parameter1 = parameter1
            self.parameters.parameter2 = parameter2

        def tostring(self):
            return self.parameters.tostring()

        def calculate_fingerprints(self, images, cores, log):
            # Do the calculations...
            self.fingerprints = fingerprints  # A dictionary.


The specific requirements, illustrated above, are:

* Has a parameters attribute (of type `ase.calculators.calculator.Parameters`), which holds the minimum information needed to re-build your module.
  That is, if your descriptor has user-settable parameters such as a cutoff radius, etc., they should be stored in this dictionary.
  Additionally, it must have the keyword "mode"; which must be set to either "atom-centered" or "image-centered".
  (This keyword will be used by the model class.)

* Has a "tostring" method, which converts the minimum parameters into a dictionary that can be re-constructed using `eval`.
  If you used the ASE `Parameters` class above, this class is simple::

    def tostring():
        return self.parameters.tostring()

* Has a "calculate_fingerprints" method.
  The images argument is a dictionary of training images, with keys that are unique hashes of each image in the set produced with :py:func:`amp.utilities.hash_images`.
  The log is a :py:class:`amp.utilities.Logger` instance, that the method can optionally use as `log('Message.')`.
  The cores keyword describes parallelization, and can safely be ignored if serial operation is desired.
  This method must save a sub-attribute `self.fingerprints` (which will be accessible in the main *Amp* instance as `calc.descriptor.fingerprints`) that contains a dictionary-like object of the fingerprints, indexed by the same keys that were in the images dictionary.
  Ideally, `descriptor.fingerprints` is an instance of :py:class:`amp.utilities.Data`, but probably any mapping (dictionary-like) object will do.

  A fingerprint is a vector.
  In **image-centered** mode, there is one fingerprint for each image.
  This will generally be just the Cartesian positions of all the atoms in the system, but transformations are possible.
  For example this could be accessed by the images key

  >>> calc.descriptor.fingerprints[key]
  >>> [3.223, 8.234, 0.0322, 8.33]

  In **atom-centered** mode, there is a fingerprint for each atom in the image.
  Therefore, calling `calc.descriptor.fingerprints[key]` returns a list of fingerprints, in the same order as the atom ordering in the original ASE atoms object.
  So to access an individual atom's fingerprints one could do

  >>> calc.descriptor.fingerprints[key][index]
  >>> ('Cu', [8.832, 9.22, 7.118, 0.312])

  That is, the first item is the element of the atom, and the second is a 1-dimensional array which is that atom's fingerprint.
   Thus, `calc.descriptor.fingerprints[hash]` gives a list of fingerprints, in the same order the atoms appear in the image they were fingerprinted from.

  If you want to train your model to forces also (besides energies), your "calculate_fingerprints" method needs to calculate derivatives of the fingerprints with respect to coordinates as well.
  This is because forces (as the minus of coordinate-gradient of the potential energy) can be written, according to the chain rule of calculus, as the derivative of your model output (which represents energy here) with respect to model inputs (which is fingerprints) times the derivative of fingerprints with respect to spatial coordinates. 
  These derivatives are calculated for each image for each possible pair of atoms (within the cutoff distance in the **atom-centered** mode).
  They can be calculated either analytically or simply numerically with finite-difference method.
  If a piece of code is written to calculate coordinate-derivatives of fingerprints, then the "calculate_fingerprints" method can save it as a sub-attribute `self.fingerprintprimes` (which will be accessible in the main *Amp* instance as `calc.descriptor.fingerprintprimes`) along with `self.fingerprints`.
  `self.fingerprintprimes` is a dictionary-like object, indexed by the same keys that were in the images dictionary.
  Ideally, `descriptor.fingerprintprimes` is an instance of :py:class:`amp.utilities.Data`, but probably any mapping (dictionary-like) object will do.

  Calling `calc.descriptor.fingerprintprimes[key]` returns the derivatives of fingerprints for the image key of interest.
  This is a dictionary where each key is a tuple representing the indices of the derivative, and each value is a list of finperprintprimes.
  (This list has the same length as the fingerprints.)
  For example, to retrieve derivatives of the fingerprints of atom indexed 2 (which is say Pt) with respect to :math:`x` coordinate of atom indexed 1 (which is say Cu), we should do

  >>> calc.descriptor.fingerprintprimes[key][(1, 'Cu', 2, 'Pt', 0)]
  >>> [-1.202, 0.130, 4.511, -0.721]

  Or to retrieve derivatives of the fingerprints of atom indexed 1 with respect to :math:`z` coordinate of atom indexed 1, we do

  >>> calc.descriptor.fingerprintprimes[key][(1, 'Cu', 1, 'Cu', 2)]
  >>> [3.48, -1.343, -2.561, -8.412]

----------------------------------
Descriptor: standard practices
----------------------------------

The below describes standard practices we use in building modules. It is not necessary to use these, but it should make your life easier to follow standard practices. And, if your code is ultimately destined to be part of an Amp release, you should plan to make it follow these practices unless there is a compelling reason not to.

We have an example of a minimal descriptor in :py:mod:`amp.descriptor.example`; it's probably easiest to copy this file and modify it to become your new descriptor. For a complete example of a working descriptor, see :py:mod:`amp.descriptor.gaussian`.

The Data class
^^^^^^^^^^^^^^^^^^^

The key element we use to make our lives easier is the :py:class:`~amp.utilities.Data` class. It should be noted that, in the development version, this is still a work in progress. The :py:class:`~amp.utilities.Data` class acts like a dictionary in that items can be accessed by key, but also saves the data to disk (it is persistent), enables calculation of missing items, and can even parallelize these calculations across cores and nodes.

It is recommended to first construct a pure python version that fits with the :py:class:`~amp.utilities.Data` scheme for 1 core, then expanding it to work with multiple cores via the following procedure. See the :py:class:`~amp.descriptor.gaussian.Gaussian` descriptor for an example of implementation.



Basic data addition
"""""""""""""""""""
To make the descriptor work with the :py:class:`~amp.utilities.Data` class, the :py:class:`~amp.utilities.Data` class needs a keyword `calculator`. The simplest example of this is our :py:class:`~amp.descriptor.gaussian.NeighborlistCalculator`, which is basically a wrapper around ASE's Neighborlist class::

    class NeighborlistCalculator:
        """For integration with .utilities.Data
        For each image fed to calculate, a list of neighbors with offset
        distances is returned.
        """

        def __init__(self, cutoff):
            self.globals = Parameters({'cutoff': cutoff})
            self.keyed = Parameters()
            self.parallel_command = 'calculate_neighborlists'

        def calculate(self, image, key):
            cutoff = self.globals.cutoff
            n = NeighborList(cutoffs=[cutoff / 2.] * len(image),
                             self_interaction=False,
                             bothways=True,
                             skin=0.)
            n.update(image)
            return [n.get_neighbors(index) for index in range(len(image))]

Notice there are two categories of parameters saved in the init statement: `globals` and `keyed`. The first are parameters that apply to every image; here the cutoff radius is the same regardless of the image. The second category contains data that is specific to each image, in a dictionary format keyed by the image hash. In this example, there are no keyed parameters, but in the case of the fingerprint calculator, the dictionary of neighborlists is an example of a `keyed` parameter. The class must have a function called `calculate`, which when fed an image and its key, returns the desired value: in this case a neighborlist. Structuring your code as above is enough to make it play well with the `Data` container in serial mode. (Actually, you don't even need to worry about dividing the parameters into globals and keyed in serial mode.) Finally, there is a `parallel_command` attribute which can be any string which describes what this function does, which will be used later.

Parallelization
"""""""""""""""
The parallelization should work provided the scheme is `embarassingly parallel <https://en.wikipedia.org/wiki/Embarrassingly_parallel>`_; that is, each image's fingerprint is independent of all other images' fingerprints. We implement this in building the :py:class:`~amp.utilities.Data` dictionaries, using a scheme of establishing SSH sessions (with pxssh) for each worker and passing messages with ZMQ.

The :py:class:`~amp.utilities.Data` class itself serves as the master, and the workers are instances of the specific module; that is, for the Gaussian scheme the workers are started with `python -m amp.descriptor.gaussian id hostname:port` where id is a unique identifier number assigned to each worker, and hostname:port is the socket at which the workers should open the connection to the mater (e.g., "node243:51247"). The master expects the worker to print two messages to the screen: "<amp-connect>" which confirms the connection is established, and "<stderr>"; the text that is between them alerts the master (and the user's log file) where the worker will write its standard error to. All messages after this are passed via ZMQ. I.e., the bottom of the module should contain something like::

    if __name__ == "__main__":
        import sys
        import tempfile

        hostsocket = sys.argv[-1]
        proc_id = sys.argv[-2]

        print('<amp-connect>')
        sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                 suffix='.stderr')
        print('stderr written to %s<stderr>' % sys.stderr.name)


After this, the worker communicates with the master in request (from the worker) / reply (from the master) mode, via ZMQ. (It's worth checking out the `ZMQ Guide <http://zguide.zeromq.org/>`_; (ZMQ Guide examples). Each request from the worker needs to take the form of a dictionary with three entries: "id", "subject", and (optionally) "data". These are easily created with the :py:class:`~amp.utilities.MessageDictionary` class. The first thing the worker needs to do is establish the connection to the master and ask its purpose::

    import zmq
    from ..utilities import MessageDictionary
    msg = MessageDictionary(proc_id)

    # Establish client session via zmq; find purpose.
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://%s' % hostsocket)
    socket.send_pyobj(msg('<purpose>'))
    purpose = socket.recv_pyobj()

In the final line above, the master has sent a string with the `parallel_command` attribute mentioned above. You can have some if/elif statements to choose what to do next, but for the calculate_neighborlist example, the worker routine is as simple as requesting the variables, performing the calculations, and sending back the results, which happens in these few lines. This is all that is needed for parallelization (in pure python)::

    # Request variables.
    socket.send_pyobj(msg('<request>', 'cutoff'))
    cutoff = socket.recv_pyobj()
    socket.send_pyobj(msg('<request>', 'images'))
    images = socket.recv_pyobj()

    # Perform the calculations.
    calc = NeighborlistCalculator(cutoff=cutoff)
    neighborlist = {}
    while len(images) > 0:
        key, image = images.popitem()  # Reduce memory.
        neighborlist[key] = calc.calculate(image, key)

    # Send the results.
    socket.send_pyobj(msg('<result>', neighborlist))
    socket.recv_string() # Needed to complete REQ/REP.


Note that in python3, there is apparently an issue that garbage collection does not work correctly. Thus, we also need to call socket.close() on each zmq.Context.socket object before it is destroyed, otherwise the program may hang when trying to make new connections.
