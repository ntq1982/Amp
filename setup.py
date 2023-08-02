#!/usr/bin/env python

import os
import warnings

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    msg = ("Please install numpy (version 1.7.0 or greater) before installing "
           "Amp. (Amp uses numpy's installer so it can compile the fortran "
           "modules with f2py.) You should be able to do this with a command"
           " like:"
           "   $ pip install numpy")
    raise RuntimeError(msg)


name = 'amp-atomistics'
version = open(os.path.join('amp', 'VERSION')).read().strip()
description = 'Atomistic Machine-learning Package'
long_description = open('README').read()
packages = ['amp', 'amp.descriptor', 'amp.model', 'amp.nft',
            'amp.preprocess', 'amp.regression', 'amp.stats']
classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2.6',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.3']
install_requires = ['numpy>=1.7.0', 'matplotlib', 'ase', 'pyzmq',
                    'pexpect']
ext_modules = [Extension(name='amp.fmodules',
                         sources=['amp/model/neuralnetwork.f90',
                                  'amp/model/chargeneuralnetwork.f90',
                                  'amp/descriptor/gaussian.f90',
                                  'amp/descriptor/cutoffs.f90',
                                  'amp/descriptor/zernike.f90',
                                  'amp/model.f90'])]
author = 'Andrew Peterson'
author_email = 'andrew_peterson@brown.edu'
url = 'https://bitbucket.org/andrewpeterson/amp'
package_data = {'amp': ['VERSION']}
scripts = ['tools/amp-compress', 'tools/amp-plotconvergence']

try:
    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          packages=packages,
          classifiers=classifiers,
          install_requires=install_requires,
          scripts=scripts,
          ext_modules=ext_modules,
          author=author,
          author_email=author_email,
          url=url,
          package_data=package_data,
          )
except SystemExit as ex:
    if 'amp.fmodules' in ex.args[0]:
        warnings.warn('It looks like no fortran compiler is present. Retrying '
                      'installation without fortran modules.')
    else:
        raise ex
    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          packages=packages,
          classifiers=classifiers,
          install_requires=install_requires,
          scripts=scripts,
          ext_modules=[],
          author=author,
          author_email=author_email,
          url=url,
          package_data=package_data,
          )
    warnings.warn('Installed Amp without fortran modules since no fortran '
                  'compiler was found. The code may run slow as a result.')
