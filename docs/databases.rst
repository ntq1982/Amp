.. _Databases:

==================================
Fingerprint databases
==================================

Often, a user will want to train multiple calculators to a common set of images. This may be just in routine development of a trained calculator (e.g., trying different neural network sizes), in using multiple training instances trying to find a good initial guess of parameters, or in making a committee of calculators. In this case, it can be a waste of computational time to calculate the fingerprints (and more expensively, the fingerprint derivatives) more than once.

To deal with this, Amp saves the fingerprints to a database, the location of which can be specified by the user. If you want multiple calculators to avoid re-fingerprinting the same images, just point them to the same database location.


Format
---------------------------------

The database format is custom for Amp, and is designed to be as simple as possible.
Amp databases end in the extension `.ampdb`.
In its simplest form, it is just a directory with one file per image; that is, you will see something like below::

    label-fingerprints.ampdb/
        loose/
            f60b3324f6001d810afbab9f85a6ea5f
            aeaaa21e5faccc62bae94c5c48b04031

In the above, each file in the directory "loose" is the hash of an image, and contains that image's fingerprint. We use a file-based "database" to avoid conflicts with multiple processes accessing a database at the same time, which can cause conflicts.

However, for large training sets this can lead to lots of loose files, which can eat up a lot of memory, and with the large number of files slow down indexing jobs (like backups and scans). Therefore, you can compress the database with the `amp-compress` tool, described below.

Compress
---------------------------------

To save disk space, you may periodically want to run the utility `amp-compress` (contained in the `tools` directory of the amp package; this should be on your path for normal installations). In this case, you would run `amp-compress <filename>`, which would result in the above `.ampdb` file being changed to::

    label-fingerprints.ampdb/
        archive.tar.gz
        loose/

That is, the two fingerprints that were in the "loose" directory are now in the file "archive.tar.gz".

You can also use the `--recursive` (or `-r`) flag to compress all ampdb files in or below the specified directory.

When Amp reads from the above database, it first looks in the "loose" directory for the fingerprint. If it is not there, it looks in "archive.tar.gz". If it is not there, it calculates the fingerprint and adds it to the "loose" directory.


Future
---------------------------------

We plan to make the amp-compress tool more automated.
If the user does not supply a separate `dblabel` keyword, then we assume that their process is the only process using the database, and it is safe to compress the database at the end of their training job.
This would automatically clean up the loose files at the end of the job.
