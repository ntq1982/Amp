.. _introduction:

==================================
Introduction
==================================

Amp is an open-source package designed to easily bring machine-learning to atomistic calculations.
This allows one to predict (or really, interpolate) calculations on the potential energy surface,
by first building up a regression representation from a "training set" of atomic images.
The Amp calculator works by first
learning from any other calculator (usually quantum mechanical calculations) that can provide energy and forces as a
function of atomic coordinates.
Depending upon the model choice, the predictions from Amp can take place with arbitrary accuracy, approaching that of the original calculator.

Amp is designed to integrate closely with the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (ASE).
As such, the interface is in pure python, although several compute-heavy parts of the underlying codes also have fortran
versions to accelerate the calculations. The close integration with ASE means that any calculator that works with ASE
- including EMT, GPAW, DACAPO, VASP, NWChem, and Gaussian - can easily be used as the parent method.
