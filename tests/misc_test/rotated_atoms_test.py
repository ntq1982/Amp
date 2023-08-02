#!/usr/bin/env python
"""This test checks rotation and translation invariance of descriptor schemes.
Fingerprints both before and after a random rotation (+ translation) are
calculated and compared."""

import numpy as np
from numpy import sin, cos
from ase import Atom, Atoms
from amp.descriptor.gaussian import Gaussian
from amp.utilities import hash_images
import random


def rotate_atom(x, y, z, phi, theta, psi):
    """Rotate atom in three dimensions."""

    rotation_matrix = [
        [cos(theta) * cos(psi),
         cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi),
         sin(phi) * sin(psi) - cos(phi) * sin(theta) * cos(psi)],
        [-cos(theta) * sin(psi),
         cos(phi) * cos(psi) - sin(phi) * sin(theta) * sin(psi),
         sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)],
        [sin(theta),
         -sin(phi) * cos(theta),
         cos(phi) * cos(theta)]
    ]

    [[xprime], [yprime], [zprime]] = np.dot(rotation_matrix, [[x], [y], [z]])

    return (xprime, yprime, zprime)


def test():
    """Rotational/translational invariance."""

    for descriptor in [Gaussian(fortran=False), ]:

        # Non-rotated atomic configuration
        atoms = Atoms([Atom('Pt', (0., 0., 0.)),
                       Atom('Pt', (0., 0., 1.)),
                       Atom('Pt', (0., 2., 1.))])

        images = hash_images([atoms], ordered=True)
        descriptor1 = descriptor
        descriptor1.calculate_fingerprints(images)
        fp1 = descriptor1.fingerprints[list(images.keys())[0]]

        # Randomly Rotated (and translated) atomic configuration
        rot = [random.random(), random.random(), random.random()]
        for i in range(1, len(atoms)):
            (atoms[i].x,
             atoms[i].y,
             atoms[i].z) = rotate_atom(atoms[i].x,
                                       atoms[i].y,
                                       atoms[i].z,
                                       rot[0] * np.pi,
                                       rot[1] * np.pi,
                                       rot[2] * np.pi)
        disp = [random.random(), random.random(), random.random()]
        for atom in atoms:
            atom.x += disp[0]
            atom.y += disp[1]
            atom.z += disp[2]

        images = hash_images([atoms], ordered=True)
        descriptor2 = descriptor
        descriptor2.calculate_fingerprints(images)
        fp2 = descriptor2.fingerprints[list(images.keys())[0]]

        for (element1, afp1), (element2, afp2) in zip(fp1, fp2):
            assert element1 == element2, 'rotated atoms test broken!'
            for _, __ in zip(afp1, afp2):
                assert (abs(_ - __) < 10 ** (-10.)), \
                    'rotated atoms test broken!'


if __name__ == '__main__':
    test()
