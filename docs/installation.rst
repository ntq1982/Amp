.. _install:

==================================
Installation
==================================

*Amp* is python-based and is designed to integrate closely with the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (ASE).
Because of this tight integration, *Amp* is compatible with every major electronic structure calculator and has ready access to all standard atomistic methods, such as structure optimization or molecular dynamics.
In its most basic form, it has few requirements:

* Python, version 3.6 is recommended (but Python 2 is still supported)
* ASE
* NumPy + SciPy

To get more features, such as parallelization in training, a few more packages are recommended:

* Pexpect (or pxssh)
* ZMQ (or PyZMQ, the python version of ØMQ).

Certain advanced modules may contain dependencies that will be noted when they are used; for example Tensorflow for the tflow module or matplotlib (version > 1.5.0) for the plotting modules.

We have three suggested means of installation, depending on your needs:

* via :ref:`pip` (python's automatic package installer)
* via :ref:`Ubuntu`,
* or :ref:`manual installation <manual-install>`.

After you install, you should :ref:`run the tests <runthetests>`.


.. _pip:

----------------------------------
Pip
----------------------------------

You can install quickly with using pip; if you have pip installed, you should be able to install the latest release quickly with::

   $ pip3 install numpy
   $ pip3 install amp-atomistics

If you already have numpy you should be able to skip the first line.
If you would rather have the development version, replace the second line with::

   $ pip3 install git+https://bitbucket.org/andrewpeterson/amp

Note that you may want to also supply the `--user` flag to pip3, which will install only in your home directory and not system wide. Another good strategy is to install inside of a virtualenv. You will need to take one of these approaches if installing on your own account within a cluster, for example.

If you see errors relating to fortran, make sure you have an f77 compiler installed (such as gfortran). If you see an error related to Python.h, make sure you have a version of python meant for compiling (e.g., `python3-dev` on ubuntu).


.. _Ubuntu:

----------------------------------
Ubuntu's package manager
----------------------------------

If you use Debian or Ubuntu, *Amp* is now included in the package manager, and you can install just like any other program (e.g., through 'Ubuntu Software' or 'synaptic' package manager).
Or most simply, just type::

   $ sudo apt install python3-amp

Note that there is a long lead time between when we submit a package to Debian/Ubuntu and when it is included in an official release, so this version will typically be a bit old.


.. _manual-install:

----------------------------------
Manual installation
----------------------------------

If the above fails, or you want to have your own version of the code that you can hack away on, you should try the manual installation, by following the below procedure.

**Python version.**
We recommend Python 3.6.
However, if you are Python 2 user the code will also work in Python 2 indefinitely.
(But you should really change over to 3!)

**Install ASE.**
We always test against the latest release of ASE, but slightly older versions (>=3.9) are likely to work as well.
Follow the instructions at the `ASE <https://wiki.fysik.dtu.dk/ase>`_ website.
ASE itself depends upon python with the standard numeric and scientific packages.
Verify that you have working versions of `NumPy <http://numpy.org>`_ and `SciPy <http://scipy.org>`_.
We also recommend `matplotlib <http://matplotlib.org>`_ (version > 1.5.0) in order to generate plots.
After you are successful, you should be able to run the following without errors::

   $ python3
   >>> import ase
   >>> import numpy
   >>> import scipy
   >>> import matplotlib

**Get the code.**
You can download a stable (numbered) release, which is citable by DOI, via the links on the Release Notes page.
You should make sure that the documentation that you are reading corresponds to the release you have downloaded; the documentation is included in the package or you can choose a version number on `http://amp.readthedocs.io <http://amp.readthedocs.io>`_.

We are constantly improving *Amp* and adding features, so depending on your needs it may be preferable to use the development version rather than "stable" releases.
We run daily unit tests to try to make sure that our development code works as intended.
We recommend checking out or downloading the latest version of the code via `the project's bitbucket page <https://bitbucket.org/andrewpeterson/amp/>`_.
If you use git, check out the code with::

   $ cd ~/path/to/my/codes
   $ git clone https://andrewpeterson@bitbucket.org/andrewpeterson/amp.git

where you should replace '~/path/to/my/codes' with wherever you would like the code to be located on your computer.

**Simple option: use setup.py.**
After you have downloaded the code, the fastest way to compile it is by running (from inside the amp directory)::

    $ python setup.py install --user

If that works, you are done!
If it doesn't work or you want to use the fully manual option, keep reading.

**Set the environment.**
You need to let your python version know where to find *Amp*.
Add the following line to your '.bashrc' (or other appropriate spot), with the appropriate path substituted for '~/path/to/my/codes'::

   $ export PYTHONPATH=~/path/to/my/codes/amp:$PYTHONPATH

You can check that this works by starting python and typing the below command, verifying that the location listed from
the second command is where you expect::

   >>> import amp
   >>> print(amp.__file__)

See also the section on parallel processing (in :ref:`UseAmp`) for any issues that arise in making the environment work with *Amp* in parallel.

**Recommended: Build fortran modules.**
*Amp* works in pure python, however, it will be annoyingly slow unless the associated Fortran modules are compiled to speed up several parts of the code.
The compilation of the Fortran code and integration with the python parts is accomplished with f2py, which is part of NumPy.
A Fortran compiler will also be necessary on the system; a reasonable open-source option is GNU Fortran, or gfortran.
This compiler will generate Fortran modules (.mod).
gfortran will also be used by f2py to generate extension module fmodules.so on Linux or fmodules.pyd on Windows.
We have included a `Makefile` that automatizes the building of Fortran modules.
To use it, install `GNU Makefile <https://www.gnu.org/software/make/>`_
on your Linux distribution or macOS.
Then you can simply do::

    $ cd <installation-directory>/amp/
    $ make

Note that you have to make sure your `f2py` is pointing to the right Python version.

If you do not have the GNU Makefile installed, you can prepare the Fortran extension modules manually in the following steps:

1. Compile model Fortran subroutines inside the model and descriptor folders by::

    $ cd <installation-directory>/amp/model
    $ gfortran -c neuralnetwork.f90
    $ cd ../descriptor
    $ gfortran -c cutoffs.f90


2. Move the modules "neuralnetwork.mod" and "cutoffs.mod" created in the last step, to the parent directory by::

    $ cd ..
    $ mv model/neuralnetwork.mod .
    $ mv descriptor/cutoffs.mod .

3. Compile the model Fortran subroutines in companion with the descriptor and neuralnetwork subroutines by something like::

    $ f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

Note that for Python3, you need to use `f2py3` instead of `f2py`.

or on a Windows machine by::

    $ f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90 --fcompiler=gnu95 --compiler=mingw32

Note that if you update your code (e.g., with 'git pull origin master') and the fortran code changes but your version of fmodules.f90 is not updated, an exception will be raised telling you to re-compile your fortran modules.

.. _runthetests:

----------------------------------
Run the tests
----------------------------------

We include tests in the package to ensure that it still runs as intended as we continue our development; we run these tests automatically with every commit (on bitbucket) to try to keep bugs out.
It is a good idea to run these tests after you install the package to see if your installation is working.
The tests are in the folder `tests`; they are designed to run with `nose <https://nose.readthedocs.org/>`_.
If you have nose and GNU Makefile installed, simply do::

   $ make py2tests      # (for Python2)
   $ make py3tests      # (for Python3)

This will create a temporary directory and run the tests there.
Otherwise, if you have only nose installed (and not GNU Makefile), run the commands below::

   $ mkdir <installation-directory>/tests/amptests
   $ cd <installation-directory>/tests/amptests
   $ nosetests -v ../../


----------------------------------------------
Note: Special ASE for grand-canonical learning
----------------------------------------------

If you plan to use the electronically grand-canonical learning scheme (for electrochemical simulations), you will need to install a special version of ASE.
The default ASE does not have the ability to save the electrode potential and the excess electrons into ``atoms.calc.results``.
We have a version that allows this; we have proposed to the ASE developers that this restriction be dropped, so hopefully in the future you will be able to use the default version of ASE.

For now, you can install the version of ASE that allows this with PIP as::

    $ python -m pip install 'ase @ git+https://gitlab.com/andrew_peterson/ase@calc_results'

Or if you prefer to manually download ASE, you can find it at https://gitlab.com/andrew_peterson/ase/-/tree/calc_results .
