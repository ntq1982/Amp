.. _fastforcecalls:


********************************
Fast Force Calls
********************************

Force calls (or potential-energy calls) are done inside Amp in a single thread, which means they should be fast compared to something like DFT, but inherently much slower than an optimized molecular dynamics code.
If you would like to run big, fast simulations, it's advisable to link the output of Amp to such a code, then run your simulation in a molecluar dynamics code.

Here, we describe three ways to do fast force calls in Amp using n2p2, PROPhet/LAMMPS and KIM/LAMMPS interface, respectively.
As of this writing, the former two approaches are more stable.

==========
Using n2p2
==========
`n2p2 <https://github.com/CompPhysVienna/n2p2>`__ is a ready-to-use software for high-dimensional neural network potentials, originally developed by Andreas Singraber at the University of Vienna.
Importantly, it provides python interface for fast predictions of energy and forces, which makes it really easy to be incorporated into other python-based code, such as Amp.
It also allows using existing neural network potentials in `LAMMPS <https://github.com/lammps/lammps>`__.
The LAMMPS NNP interface is detailed in the `documentation <https://compphysvienna.github.io/n2p2/interfaces/if_lammps.html>`__.


The connection from Amp to n2p2's fast force calls is made possible via a utility function :py:func:`amp.convert.save_to_n2p2` written by Cheng Zeng (Brown).
If you use this interface with Amp, please cite the n2p2 paper in addition to the Amp paper:

    Singraber, A.; Behler, J.; Dellago, C. Library-Based LAMMPS Implementation of High-Dimensional Neural Network Potentials. J. Chem. Theory Comput. 2019, 15 (3), 1827–1840. |n2p2_paper|


.. |n2p2_paper| raw:: html

   <a href="https://doi.org/10.1021/acs.jctc.8b00770" target="_blank">[doi:10.1021/acs.jctc.8b00770] </a>

In the next, an example for using this interface is described.
Suppose you have a trained Amp calculator saved as "amp.amp", and you want to predict on an image in a trajectory file 'image.traj', you can convert the Amp calculator and the `ASE` trajectory to n2p2 input files by :

.. code-block:: python

   from amp import Amp
   from amp.convert import save_to_n2p2
   from ase.io import read

   calc = Amp.load('amp.amp')
   desc_pars = calc.descriptor.parameters
   model_pars = calc.model.parameters
   atoms = read('image.traj')
   save_to_n2p2(desc_pars, model_pars, images=atoms)

Then you can make prediction via the python interface shown below:

.. code-block:: python

   import pynnp

   # Initialize NNP prediction mode.
   p = pynnp.Prediction()
   # Read settings and setup NNP.
   p.setup()
   # Read in structure.
   p.readStructureFromFile()
   # Predict energies and forces.
   p.predict()
   # Shortcut for structure container.
   s = p.structure
   print("------------")
   print("Structure 1:")
   print("------------")
   print("numAtoms           : ", s.numAtoms)
   print("numAtomsPerElement : ", s.numAtomsPerElement)
   print("------------")
   print("Energy (Ref) : ", s.energyRef)
   print("Energy (NNP) : ", s.energy)
   print("------------")
   forces = []
   for atom in s.atoms:
       print(atom.index, atom.f.r)

or simply use a command::

   $ nnp-predict 1


==================================
Using PROPhet/LAMMPS
==================================

`PROPhet <https://github.com/biklooost/PROPhet/>`__ was a nice atomistic machine-learning code developed by Brian Kolb and Levi Lentz in the group of Alexie Kolpak at MIT.

.. code-block:: none

   //     _____________________________________      _____   |
   //     ___/ __ \__/ __ \_/ __ \__/ __ \__/ /________/ /   |
   //     __/ /_/ /_/ /_/ // / / /_/ /_/ /_/ __ \/ _ \/ __/  |
   //     _/ ____/_/ _, _// /_/ /_/ ____/_/ / / /  __/ /_    |
   //     /_/     /_/ |_| \____/ /_/     /_/ /_/\___/\__/    |
   //---------------------------------------------------------

Included in PROPhet was a potential that could be installed into `LAMMPS <https://github.com/lammps/lammps>`__ (a very fast molecular dynamics code); this potential allowed for neural-network potentials in the Behler--Parinello scheme to run in LAMMPS.
If you install this potential into your own copy of LAMMPS, you can then use the utility function :py:func:`amp.convert.save_to_prophet` to output your data in a format where you can use LAMMPS for your force calls.


The work of making the connection from Amp to PROPhet's LAMMPS potential was done by Efrem Braun (Berkeley), Levi Lentz (MIT), and August Edwards Guldberg Mikkelsen (DTU).
If you use this interface with Amp, please cite the PROPhet paper in addition to the Amp paper in any publications that result:

    Kolb, Lentz & Kolpak, "Discovering charge density functionals and structure-property relationships with PROPhet: A general framework for coupling machine learning and first-principles methods", *Scientific Reports* 7:1192, 2017. |prophet_paper|


.. |prophet_paper| raw:: html

   <a href="http://dx.doi.org/10.1038/s41598-017-01251-z" target="_blank">[doi:10.1038/s41598-017-01251-z] </a>


The instructions below assume you are on a linux-like system and have Amp already installed.
It also uses git to download codes and change branches.
Create a folder, where everything will be stored called (e.g., LAMPHET) and go into it::

   $ mkdir LAMPHET
   $ cd LAMPHET

Download the latest stable LAMMPS version into the LAMPHET directory::

   $ git clone https://github.com/lammps/lammps.git

For this purpose, we will not be using the PROPhet version from the official repository, but instead from this `fork <https://github.com/Augustegm/PROPhet>`__.
Download it and then change to the amp compatible branch::

   $ git clone https://github.com/Augustegm/PROPhet.git
   $ cd PROPhet
   $ git checkout amp_compatible

Now we need to set the following environment variables in our .bashrc::

   export LAMPHET=path_to_your_codes/LAMPHET
   export PROPhet_DIR=$LAMPHET/PROPhet/src
   export LAMMPS_DIR=$LAMPHET/lammps/src 
   export PATH=$PATH:$LAMMPS_DIR
   export PYTHONPATH=$LAMPHET/lammps/python:$PYTHONPATH
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LAMPHET/lammps/src

The next step is to compile PROPhet. To do this correctly, you will need to first write the `Makefile` and then we will manually edit it::

   $ cd $PROPhet_DIR
   $ ./configure --prefix=$LAMPHET/prophet-install --enable-lammps=$LAMMPS_DIR

Append `-fPIC` to line 8 in the `Makefile`.
It should look like one of the two lines below::

   CFLAGS =-O3 -DUSE_MPI -fPIC
   CFLAGS =-O3 -fPIC

Now build PROPhet by typing::

   $ make

The next step is to compile LAMMPS. To do this we first need to copy over a file from PROPhet::

   $ cd $LAMMPS_DIR
   $ cp $PROPhet_DIR/pair_nn.h .

We also need to change some lines in the `Makefile.package.empty` file. Edit lines 4-6 to::

   PKG_INC = -I$(PROPhet_DIR)
   PKG_PATH = -L$(PROPhet_DIR)
   PKG_LIB = -lPROPhet_lammps

Now we can compile LAMMPS. It is recommended to compile it in the four different ways
giving a serial and parallel version as well as shared library versions, which are needed if one
wants to use it from Python (needed for using the LAMMPS interface in ASE)::

   $ make serial
   $ make mpi
   $ make serial mode=shlib
   $ make mpi mode=shlib


==================================
Using OpenKIM
==================================

*Note*: The forces predicted with the KIM approach may not be compatible with Amp forces as described in these merge-request `comments <https://bitbucket.org/andrewpeterson/amp/pull-requests/41/update-to-used-kim-api-version-200-final/diff>`__.
Use this approach with caution.

Machine-learning parameters trained in *Amp* can be used to perform fast molecular dynamics simulations, via the `Knowledge Base for Interatomic Models <https://openkim.org/>`__ (KIM).
`LAMMPS <http://www.afs.enea.it/software/lammps/doc17/html/Section_packages.html#kim>`__ recognizes *kim* as a pair style that interfaces with the KIM repository of interatomic potentials.

To build LAMMPS with the KIM package you must first install the KIM API (library) on your system.
Below are the minimal steps you need in order to install the KIM API.
After KIM API is installed, you will need to install LAMMMPS from its `github repository <https://github.com/lammps/lammps>`__.
Finally we will need to install the model driver that is provided in the *Amp* repository.
In the followings we discuss each of these steps.

In this installation instruction, we assume that the following requirements are installed on your system:

* git
* make
* cmake (If it is not installed on your system see `here <https://cmake.org/install/>`__.)
* GNU compilers (gcc, g++, gfortran) version 4.8.x or higher.


----------------------------------
Installation of KIM API
----------------------------------

You can follow the instructions given at the OpenKIM `github repository <https://github.com/openkim/kim-api/blob/master/INSTALL>`__ to install KIM API.
In short, you need to clone the repository by::

   $ git clone https://github.com/openkim/kim-api.git

Next do the following::

   $ cd kim-api-master && mkdir build && cd build
   $ FC=gfortran-4.8 cmake .. -DCMAKE_BUILD_TYPE=Release
   $ make
   $ sudo make install
   $ sudo ldconfig

The second line forces cmake to use gfortran-4.8 as the fortran compiler.
We saw gfortran-5 throws error "Error: TS 29113/TS 18508: Noninteroperable array" but gfortran-4.8 should work fine.
Now you can list model and model drivers available in KIM API by::

   $ kim-api-collections-management list

or install and remove models and model drivers, etc.
For a detailed explanation of possible options see `here <https://openkim.org/kim-api/>`__.


----------------------------------
Building LAMMPS
----------------------------------

Clone LAMMPS source files from the `github repository <https://github.com/lammps/lammps>`__::

   $ git clone https://github.com/lammps/lammps.git

Now you can do the following to build LAMMPS::

   $ cd lammps && mkdir build && cd build
   $ cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -D CMAKE_Fortran_COMPILER=gfortran -D PKG_KIM=on -D KIM_LIBRARY=$"/usr/local/lib/libkim-api.so" -D KIM_INCLUDE_DIR=$"/usr/local/include/kim-api" ../cmake
   $ make


----------------------------------
Installation of *amp_model_driver*
----------------------------------

Now you are ready to install the *amp_model_driver* provided on this repository.
To do that first change to *amp-kim* directory by::

   $ cd /amp_directory/amp/tools/amp-kim/

where *amp_directory* is where your *Amp* source files are located.

Then make a copy of the fortran modules inside the *amp_model_driver* directory by::

   $ cp ../../amp/descriptor/gaussian.f90 amp_model_driver/gaussian.F90
   $ cp ../../amp/descriptor/cutoffs.f90 amp_model_driver/cutoffs.F90
   $ cp ../../amp/model/neuralnetwork.f90 amp_model_driver/neuralnetwork.F90

Finally you can install the *amp_model_driver* by::

   $ kim-api-collections-management install user ./amp_model_driver

You can now remove the fortran modules that you copied earlier::

   $ rm amp_model_driver/gaussian.F90
   $ rm amp_model_driver/cutoffs.F90
   $ rm amp_model_driver/neuralnetwork.F90


----------------------------------------
Installation of *amp_parametrized_model*
----------------------------------------

Now that you have *amp_model_driver* installed, you need to install the parameters also as the final step.
**Note that this is the only step that you need to repeat when you change the parameters of the machine-learning model.**
You should first parse all of the parameters of your *Amp* calculator to a text file by:

.. code-block:: python

   from amp import Amp
   from amp.convert import save_to_openkim
   
   calc = Amp(...)
   calc.train(...)
   save_to_openkim(calc)

where the last line parses the parameters of the calc object into a text file called *amp.params*.

You should then copy the generated text file into the *amp_parameterized_model* sub-directory of the *Amp* source directory::

   $ cp /working_directory/amp.params amp_directory/amp/tools/amp-kim/amp_parameterized_model/.

where *working_directory* is where *amp.params* is located initially, and *amp_directory* is the directory of the *Amp* source files.
Finally you change back to the *amp-kim* directory by::

   $ cd /amp_directory/amp/tools/amp-kim/

Note that installation of *amp_parameterized_model* will not work without *amp.params* being located in the */amp_directory/amp/tools/amp-kim/amp_parameterized_model* directory.
Next install your parameters by::

   $ kim-api-collections-management install user ./amp_parameterized_model

Congrats!
Now you are ready to use the *Amp* calculator with *amp.params* in you molecular dynamics simulation by an input file like this:

.. code-block:: bash

   variable       x index 1
   variable       y index 1
   variable       z index 1

   variable       xx equal 10*$x
   variable       yy equal 10*$y
   variable       zz equal 10*$z
   
   units          metal
   atom_style     atomic

   lattice        fcc 3.5
   region         box block 0 ${xx} 0 ${yy} 0 ${zz}
   create_box     1 box
   create_atoms   1 box
   mass           1 1.0
   
   velocity       all create 1.44 87287 loop geom
   
   pair_style     kim amp_parameterized_model
   pair_coeff     * * Pd
   
   neighbor       0.3 bin
   neigh_modify   delay 0 every 20 check no
   
   fix            1 all nve
   
   run            10

which, for example, is an input script for LAMMPS to do a molecular dynamics simulation of a Pd system for 10 units of time.

