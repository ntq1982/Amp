"""
This script creates a list of three images. It then calculates Gaussian
fingerprints of images with and without fortran modules on different number of
cores, and check consistency between them.

"""

import numpy as np
from ase import Atoms
from amp.descriptor.gaussian import Gaussian
from amp.utilities import hash_images

# Making the list of images


def make_images():
    """Makes test images."""

    images = [Atoms(symbols='Pd3O2',
                    pbc=np.array([True,  True, False], dtype=bool),
                    cell=np.array(
                        [[7.78,   0.,   0.],
                         [0.,   5.50129076,   0.],
                            [0.,   0.,  15.37532269]]),
                    positions=np.array(
                        [[3.89,  0.,  8.37532269],
                            [0.,  2.75064538,  8.37532269],
                            [3.89,  2.75064538,  8.37532269],
                            [5.835,  1.37532269,  8.5],
                            [5.835,  7.12596807,  8.]])),
              Atoms(symbols='Pd3O2',
                    pbc=np.array([True,  True, False], dtype=bool),
                    cell=np.array(
                        [[7.78,   0.,   0.],
                         [0.,   5.50129076,   0.],
                            [0.,   0.,  15.37532269]]),
                    positions=np.array(
                        [[3.88430768e+00,   5.28005966e-03,
                          8.36678641e+00],
                            [-1.01122240e-02,   2.74577426e+00,
                                8.37861758e+00],
                            [3.88251383e+00,   2.74138906e+00,
                                8.37087611e+00],
                            [5.82067191e+00,   1.19156898e+00,
                                8.97714483e+00],
                            [5.83355445e+00,   7.53318593e+00,
                             8.50142020e+00]])),
              Atoms(symbols='Pd3O2',
                    pbc=np.array([True,  True, False], dtype=bool),
                    cell=np.array(
                        [[7.78,   0.,   0.],
                         [0.,   5.50129076,   0.],
                            [0.,   0.,  15.37532269]]),
                    positions=np.array(
                        [[3.87691266e+00,   9.29708987e-03,
                          8.35604207e+00],
                            [-1.29700138e-02,   2.74373753e+00,
                                8.37941484e+00],
                            [3.86813484e+00,   2.73488653e+00,
                                8.36395999e+00],
                            [5.80386111e+00,   7.98192190e-01,
                                9.74324179e+00],
                            [5.83223956e+00,   8.23855393e+00,
                             9.18295137e+00]]))]

    return images


def test():
    """Gaussian fingerprints consistency.

    Tests that pure-python and fortran, plus different number of cores
    give same results.
    """

    images = make_images()
    images = hash_images(images, ordered=True)

    ref_fps = {}
    ref_fp_primes = {}
    count = 0
    for fortran in [False, True]:
        for cores in range(1, 2):
            descriptor = Gaussian(fortran=fortran,
                                  dblabel='Gaussian-%s-%d' % (fortran, cores))
            descriptor.calculate_fingerprints(images,
                                              parallel={'cores': cores},
                                              log=None,
                                              calculate_derivatives=True)
            for hash, image in images.items():
                if count == 0:
                    ref_fps[hash] = descriptor.fingerprints[hash]
                    ref_fp_primes[hash] = descriptor.fingerprintprimes[hash]
                else:
                    fps = descriptor.fingerprints[hash]
                    # Checking consistency between fingerprints
                    for (element1, afp1), \
                            (element2, afp2) in zip(ref_fps[hash], fps):
                        assert element1 == element2, \
                            'fortran-python consistency for Gaussian '
                        'fingerprints broken!'
                        for _, __ in zip(afp1, afp2):
                            assert (abs(_ - __) < 10 ** (-15.)), \
                                'fortran-python consistency for Gaussian '
                            'fingerprints broken!'
                    # Checking consistency between fingerprint primes
                    fpprime = descriptor.fingerprintprimes[hash]
                    for key, value in ref_fp_primes[hash].items():
                        for _, __ in zip(value, fpprime[key]):
                            assert (abs(_ - __) < 10 ** (-15.)), \
                                'fortran-python consistency for Gaussian '
                            'fingerprint primes broken!'
            count += 1

if __name__ == '__main__':
    test()
