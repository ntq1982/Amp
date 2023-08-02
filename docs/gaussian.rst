.. _Gaussian:


Gaussian descriptor
===================

Custom parameters
-----------------

The Gaussian descriptor creates feature vectors based on the Behler scheme, and defaults to a small set of reasonable values. The values employed are always written to the log file and within saved instances of Amp calculators. You can specify custom parameters for the elements of the feature vectors as listed in the documentation of the :py:class:`~amp.descriptor.gaussian.Gaussian` class.

There is also a helper function :py:func:`~amp.descriptor.gaussian.make_symmetry_functions` within the :py:mod:`amp.descriptor.gaussian` module to assist with this. An example of making a custom fingerprint is given below for a two-element system.

.. code-block:: python

 import numpy as np
 from amp import Amp
 from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
 from amp.model.neuralnetwork import NeuralNetwork

 elements = ['Cu', 'Pt']
 G = make_symmetry_functions(elements=elements, type='G2',
                             etas=np.logspace(np.log10(0.05), np.log10(5.),
                                              num=4),
                             offsets=[0., 2.])
 G += make_symmetry_functions(elements=elements, type='G4',
                              etas=[0.005],
                              zetas=[1., 4.],
                              gammas=[+1., -1.])

 G = {'Cu': G,
      'Pt': G}
 calc = Amp(descriptor=Gaussian(Gs=G),
            model=NeuralNetwork())


To include angular symmetry functions of triplets inside the cutoff sphere but
with distances larger than the cutoff radius you need to slightly modify the
snippet above:

.. code-block:: python

 import numpy as np
 from amp import Amp
 from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
 from amp.model.neuralnetwork import NeuralNetwork

 elements = ['Cu', 'Pt']
 G = make_symmetry_functions(elements=elements, type='G2',
                             etas=np.logspace(np.log10(0.05), np.log10(5.),
                                              num=4),
                             offsets=[0., 2.])
 G += make_symmetry_functions(elements=elements, type='G5',
                              etas=[0.005],
                              zetas=[1., 4.],
                              gammas=[+1., -1.])

 G = {'Cu': G,
      'Pt': G}
 calc = Amp(descriptor=Gaussian(Gs=G),
            model=NeuralNetwork())
