.. Amp documentation master file, created by
   sphinx-quickstart on Thu Jul 30 17:27:50 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Amp: Atomistic Machine-learning Package
=======================================

Amp is an open-source package designed to easily bring machine-learning to atomistic calculations.
This project is being developed at Brown University in the School of Engineering, primarily by **Andrew Peterson** and **Alireza Khorshidi**, and is released under the GNU General Public License.

The latest stable release of Amp is version 1.0.1, released on January 25, 2023; see the :ref:`ReleaseNotes` page for a download link.
Please see the project's `git repository <https://bitbucket.org/andrewpeterson/amp>`_ for the latest development version or a place to report an issue.

You can read about Amp in the below paper; if you find this project useful, we would appreciate if you cite this work:

    Khorshidi & Peterson, "Amp: A modular approach to machine learning in atomistic simulations", *Computer Physics Communications* 207:310-324, 2016. |amp_paper|


.. |amp_paper| raw:: html

   <a href="http://dx.doi.org/10.1016/j.cpc.2016.05.010" target="_blank">DOI:10.1016/j.cpc.2016.05.010</a>

An amp-users mailing list exists for general discussions about the use and development of Amp. You can subscribe via listserv at:

https://listserv.brown.edu/?SUBED1=AMP-USERS&A=1

Amp is now part of the Debian archives! This means it will soon be available via your package manager in linux releases like Ubuntu.

Amp is now installable via pip! This means you should be able to install with just::

   $ pip3 install amp-atomistics

**Manual**:

.. toctree::
   :maxdepth: 1

   introduction.rst
   installation.rst
   useamp.rst
   community.rst
   theory.rst
   credits.rst
   releasenotes.rst
   examplescripts.rst
   analysis.rst
   building.rst
   moredescriptor.rst
   moremodel.rst
   gaussian.rst
   tensorflow.rst
   bootstrap.rst
   nearsightedforcetraining.rst
   grandcanonical.rst
   databases.rst
   fastforcecalls.rst
   develop.rst

**Module autodocumentation**:

.. toctree::
   :maxdepth: 1

   modules/main.rst
   modules/descriptor.rst
   modules/model.rst
   modules/regression.rst
   modules/utilities.rst
   modules/analysis.rst
   modules/stats.rst
   modules/convert.rst
   modules/preprocess.rst
   modules/nft.rst


**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
