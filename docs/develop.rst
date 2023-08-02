.. _Develop:

==================================
Development
==================================

This page contains standard practices for developing Amp, focusing on repositories and documentation.

----------------------------------
Official repository
----------------------------------

The official Amp repository lives on bitbucket, `andrewpeterson/amp <https://bitbucket.org/andrewpeterson/amp>`_ .
We employ a branching model where the `master` branch is the main development branch, containing day-to-day commits from the core developers and honoring merge requests from others.
From time to time, we create a new branch that corresponds to a release.
This release branch contains only the tagged release and any bug fixes.

   .. image:: _static/branches.svg
      :width: 400 px
      :align: center


----------------------------------
Contributing
----------------------------------

You are welcome to contribute new features, bug fixes, better documentation, etc., to Amp.
If you would like to contribute, please create a private fork and a branch for your new commits.
When it is ready, send us a merge request.
We follow the same basic model as ASE.
Please see the ASE documentation for complete instructions; a summary is also listed below.

As good coding practice, make sure your code passes both the pyflakes and pep8 tests.
(On linux, you should be able to run `pyflakes file.py` and `pep8 file.py`; then correct your code until the warnings disappear.)
If adding a new feature: please add a (very brief) test to the tests folder to ensure your new code continues to work as the project evolves, and also be sure to write clear documentation.
Finally, to make users aware of your new feature or change, add a bullet point to the release notes page of the documentation under the Development version heading.

It is also a good idea to send us an email if you are planning something complicated.


----------------------------------
Your fork and branches
----------------------------------

If you would like to contribute, here is our recommended way of using git to ultimately create a merge request that contains all of your changes to be included in *Amp*.

**Initial setup.**
First, create an account on bitbucket, and from the official Amp repository click the button to create a *fork* into your own account.
From the website for your fork, find the button to clone it, and use this to create a copy on your own filesystem.
This means you will run a command similar to this on your own machine:

.. code-block:: bash

    git clone git@bitbucket.org:myusername/amp.git

On your local computer, the term "origin" will refer to your own fork of Amp; we will also need to be able to access the original fork; we'll name this "upstream" and link it with a command like:

.. code-block:: bash

    git remote add upstream git@bitbucket.org:andrewpeterson/amp.git

You can check that the above makes sense by running `git remote -v`.

**Making changes.**
Before making any changes, it's a good idea to make sure your local copy is up-to-date with the parent fork.
You can do this with

.. code-block:: bash

    git checkout master  # Make sure we are on the right branch.
    git pull upstream master

To make changes, first create a local branch with a descriptive name, for example "fix-fingerprints". You can do this with

.. code-block:: bash

    git checkout -b fix-fingerprints

Your local code is now in a new branch, which you can verify by typing `git status` (or `git branch` to see all your branches).
Now, go ahead and edit your code and commit your changes with `git commit`.
You can make as many commits to your local copy as you like as you develop.
When you think your code is ready to be part of the official Amp repository, first make sure it is still up-to-date with the upstream repository, then push your branch to your own fork:

.. code-block:: bash

   git pull upstream master
   git push origin fix-fingerprints

Now you are ready to put in a merge request.
You will likely see a local message telling you how to do this after you push, but if not, just go to your own bitbucket page, open the branch there, and look for a button for a merge request.
Type a clear description and submit.

If you'd like to discuss some aspects of your code before it is ready, you can do the above but prefix the merge request title with "WIP: " (work in progress).
Then others can review your code before you submit it officially.

----------------------------------
Documentation
----------------------------------

This documentation is built with sphinx.
(Mkdocs doesn't seem to support autodocumentation.)
To build a local copy, cd into the docs directory and try a command such as

.. code-block:: bash

   sphinx-build . /tmp/ampdocs
   firefox /tmp/ampdocs/index.html &  # View the local copy.

This uses the style "bizstyle"; if you find this is missing on your system, you can likely install it with

.. code-block:: bash

   pip install --user sphinxjp.themes.bizstyle


You should then be able to update the documentation rst files and see changes on your own machine.
For line breaks, please use the style of containing each sentence on a new line.

----------------------------------
Releases
----------------------------------

To create a release, we go through the following steps.

* Be sure that `setup.py` has any new module, tools, or fortran scripts, and that the `Makefile` s are likewise updated.

* Reserve a DOI for the new release via zenodo.org.
  Do this by creating a new upload, and choosing "pre-reserve" before adding any files.

* Prepare the master branch for the release.
  (1) Update Release Notes, where the changes should have been catalogued under a "Development version" heading; move these to a new heading for this release, along with a release date and the DOI from above. Keep an empty "Development version" section for future develompents.
  (2) Also note the latest stable release on the index.rst page.
  (3) Commit these changes to the master branch.

* Create a new branch on the bitbucket repository with the version name, as in `v0.5`.
  (Don't create a separate branch if this is a bugfix release, e.g., 0.5.1 --- just add those to the v0.5 branch.)
  Note the branch name starts with "v", while the tag names will not, to avoid naming conflicts.

* Check out the new branch to your local machine (e.g., ``git fetch && git checkout v0.5``).
  All subsequent work is in the new branch.

* Change `amp/VERSION` to reflect the release number (without 'beta'). Note this will automatically change it in `docs/conf.py`, the Amp log files, and `setup.py`.

* On the Release Notes page, delete the "Development version" heading.

* Commit and push the changes to the new branch on bitbucket.

* Tag the release with the release number, e.g., '0.5' or '0.5.1', the latter being for bug fixes.
  Do this on a local machine (on the correct branch) with ``git tag -a 0.5``, followed by ``git push origin --tags``.

* Add the version to readthedocs' available versions; also set it as the default stable version.
  (This may already be done automatically.)

* Upload an archive and finalize the DOI via zenodo.org.
  Note that all the ".git" files and folders should be removed from the .tar.gz archive before uploading to Zenodo.

* Prepare and upload to PyPI (for pip)::

    $ python3 setup.py sdist
    $ twine upload dist/*

* Send a note to the amp-users list summarizing the release.

* In the master branch, update the VERSION file to reflect the new beta version; e.g., if you just released 1.0, then set the version to `1.1-beta`.
