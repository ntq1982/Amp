"""
This script creates a list of three images. It then calculates Zernike
fingerprints of images with and without fortran modules on different number of
cores, and checks consistency between them.

"""

import numpy as np
from ase import Atoms
from amp.descriptor.zernike import Zernike
from amp.utilities import hash_images, assign_cores


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
                            [5.835,  7.12596807,  8.]]))]

    return images


def test():
    """Zernike fingerprints consistency.

    Tests that pure-python and fortran, plus different number of cores
    give same results.
    """

    images = make_images()
    images = hash_images(images, ordered=True)

    ref_fps = {}
    ref_fp_primes = {}
    count = 0
    for fortran in [True, False]:
        for ncores in range(1, 4):
            cores = assign_cores(ncores)
            descriptor = Zernike(fortran=fortran,
                                 dblabel='Zernike-%s-%d' % (fortran, ncores))
            descriptor.calculate_fingerprints(images,
                                              parallel={'cores': cores,
                                                        'envcommand': None},
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
                            'fortran-python consistency for Zernike '
                        'fingerprints broken!'
                        for _, __ in zip(afp1, afp2):
                            assert (abs(_ - __) < (10 ** (-10.))), \
                                'fortran-python consistency for Zernike '
                            'fingerprints broken!'
                    # Checking consistency between fingerprint primes
                    fpprime = descriptor.fingerprintprimes[hash]
                    for key, value in ref_fp_primes[hash].items():
                        for _, __ in zip(value, fpprime[key]):
                            assert (abs(_ - __) < (10 ** (-10.))), \
                                'fortran-python consistency for Zernike '
                            'fingerprint primes broken!'
            count += 1

if __name__ == '__main__':
    test()
