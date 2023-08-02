import os
import sys
import shutil
import numpy as np
from string import Template
import time
import json
import pickle
from scipy.stats.mstats import mquantiles
import tarfile
import tempfile

import ase.io

from ..utilities import (hash_images, Logger, now, make_filename,
                         hash_with_potential)
from .. import Amp

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

calc_text = """
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork

calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(),
           dblabel='../amp-db')
calc.model.lossfunction.parameters['weight_duplicates'] = False
"""

train_line = "calc.train(images=trainfile)"

headerlines = ''

nft_ids = None

script = """#!/usr/bin/env python3
${headerlines}

import os, pickle
from ase.parallel import paropen
from amp.utilities import TrainingConvergenceError

ensemble_index = int(os.path.split(os.getcwd())[-1])
trainfile = '../training-images/%i.traj' % ensemble_index

${calc_text}

nft_ids = None
if os.path.exists('../saved-info'):
    with open('../saved-info/nft_ids.pkl', 'rb') as pf:
        nft_ids = pickle.load(pf)
calc.model.lossfunction.parameters['nft_ids'] = nft_ids

converged = True
try:
    ${train_line}
except TrainingConvergenceError:
    converged = False

f = paropen('converged', 'w')
f.write(str(converged))
f.close()
"""


class BootStrap:
    """A bootstrap ensemble calculator which serves as a wrapper around and
    Amp calculator. Initiate with an amp.utilities.Logger instance as log.

    If an existing trained bootstrap calculator is available, it can be
    loaded by providing its filename to the load keyword.

    Note that the 'train' method is meant to be a job-submission and
    -management script;
    e.g., it will typically be run at the command line to both submit jobs
    and monitor their convergence.
    """

    def __init__(self, load=None, log=None, label='amp',
                 retrain_initiated=False, from_scratch=False):
        if log is None:
            log = Logger(sys.stdout)
        self.log = log
        self.retrain_initiated = retrain_initiated
        self.from_scratch = from_scratch
        log('=' * 70)
        log('Amp bootstrap initiated.')
        log('Date: %s' % now(with_utc=True))
        self.ensemble = []
        if load is None:
            return

        with open(load) as f:
            calctexts = json.load(f)
        for calctext in calctexts:
            f = StringIO(calctext)
            calc = Amp.load(file=f, label=label)
            calc.log = Logger(None)
            self.ensemble.append(calc)
        log(f'Loaded {load} of {len(self.ensemble)} calculators.')

    def train(self, images, n=50, calc_text=calc_text, headerlines=headerlines,
              start_command='python run.py', sleep=0.1, expired=3600.,
              train_line=train_line, label='bootstrap', new_images=None,
              nft_ids=None, charge_training=False, archive=True,
              remove_dir=False):
        """Trains a bootstrap ensemble of calculators.

        This is set up to enable the submission of each as a job through
        the local queuing system, but can also run in serial.
        On first call to this method, jobs are created/submitted.
        On subsequent calls, jobs are analyzed for convergence.
        If all are converged, an ensemble is created and the training
        directory is archived.

        Parameters
        ----------
        n: int
           size of ensemble (number of calculators to train)
        calc_text: str
           text that is used to initiate the Amp calculator.
           see the example in this module in calc_text; must produce
           a 'calc' object
        headerlines: str
           lines in the top of the python script that will be submitted
           this would typically contain comment lines for the batching
           system, such as '#SBATCH -n=8...'
        start_command: str
           command to start the job in the current queuing system,
           such as 'sbatch run.py' ('run.py' is the scriptname here)
           for serial operation use 'python run.py'
        sleep : float
           time (s) to sleep between job submissions
        train_line: str
           line to use to train each amp instance; usually the default is
           fine but user may want to use this to insert additional keywords
           such as train_forces=False
        label: string
           label to give final trained calculator
        expired: float
           When checking jobs, age (s) of log file at which to consider
           that the job is no longer running (timed out) and should be
           restarted.
        retrain: bool
            if the run is for retraining.
        new_images: str
            new_images added to the original images for retraining.
        nft_ids: list of tuples
            list of length-two tuples indicating images to be trained
            only on forces of central atoms.
        charge_training: bool
            if charge training is applied.
        archive: bool
            after training, if the training directory is archived.
        remove_dir: bool
            after archiving, if the directory is deleted.

        Returns
        -------
        results: dict
            A dictionary indicating the state of training. This dictionary
            always contains a key 'complete' key with value of True or
            False indicating if training is complete. If False, also
            provides statistics on number converged.
        """
        log = self.log
        trainingpath = '-'.join((label, 'training'))
        script_dict = dict({'headerlines': headerlines,
                            'calc_text': calc_text,
                            'train_line': train_line})

        if new_images:
            if not os.path.exists(trainingpath):
                raise RuntimeError('Training path not found')

        if os.path.exists(trainingpath):
            if not new_images or self.retrain_initiated:
                log('Path exists. Checking for　which jobs are finished.')
                results = self._manage_jobs(n, trainingpath, expired,
                                            start_command, sleep, label,
                                            archive, remove_dir)
                return results
            else:
                log("Retrain preprocessing...")
                self._retrain_preprocess(n, trainingpath)
        if not new_images:
            self._generate_training_images(trainingpath, images, n,
                                           charge_training=charge_training)
        else:
            self.retrain_initiated = True
            self._accelerated_resample(trainingpath, images, n, new_images,
                                       charge_training=charge_training)

        if nft_ids:
            self._add_nft_ids(trainingpath, nft_ids)
        self._submit_jobs(n, trainingpath, start_command,
                          sleep=0.1, **script_dict)
        return {'complete': False,
                'n_converged': 0}

    def _manage_jobs(self, n, trainingpath, expired, start_command, sleep,
                     label, archive, remove_dir):
        """Checks the running jobs to see which have finished, tries
        to restart any that are stuck, and creates a bundled trajectory
        when everything is finished."""
        def clean_and_restart():
            for _ in os.listdir(os.getcwd()):
                if _ != 'run.py':
                    if os.path.isdir(_):
                        shutil.rmtree(_)
                    else:
                        os.remove(_)
            os.system(start_command)
            time.sleep(sleep)
            log('  ---> restarted.')

        log = self.log
        n_unfinished = 0
        n_converged = 0
        n_unconverged = 0
        n_expired = 0
        n_notstarted = 0
        pwd = os.getcwd()
        os.chdir(trainingpath)
        fulltrainingpath = os.getcwd()
        for index in range(n):
            os.chdir('%i' % index)
            if not os.path.exists('converged'):
                if not os.path.exists('amp-log.txt'):
                    log('%i: Not started; no amp-log.txt file.' % index)
                    n_notstarted += 1
                else:
                    age = time.time() - os.path.getmtime('amp-log.txt')
                    log('{:d}: Still running? No converged file. Age: '
                        '{:.1f} hr'.format(index, age / 3600.))
                    if age > expired:
                        log(' Assumed expired. Cleaning up directory and '
                            'restarting.')
                        n_expired += 1
                        clean_and_restart()
                    else:
                        n_unfinished += 1
                os.chdir(fulltrainingpath)
                continue
            with open('converged') as f:
                converged = f.read()

            if converged == 'True':
                log('%i: Converged.' % index)
                n_converged += 1
            else:
                log('%i: Not converged. Cleaning up directory to '
                    'restart job.' % index)
                n_unconverged += 1
                clean_and_restart()
            os.chdir(fulltrainingpath)
        log('')
        log('Stats:')
        log('%10i converged' % n_converged)
        log('%10i not yet started' % n_notstarted)
        log('%10i apparently still running' % n_unfinished)
        log('%10i did not converge, restarted' % n_unconverged)
        log('%10i expired, restarted' % n_expired)
        log('=' * 10)
        log('%10i total' % n)
        log('\n')

        if n_converged < n:
            log('Not all runs converged; not creating bundled amp '
                'calculator.')
            os.chdir(pwd)
            return {'complete': False,
                    'n_converged': n_converged}

        log('Creating bundled amp calculator.')
        ensemble = []
        for index in range(n):
            os.chdir('%i' % index)
            with open('amp.amp') as f:
                text = f.read()
            ensemble.append(text)
            os.chdir(fulltrainingpath)
        os.chdir(pwd)
        with open('%s.ensemble' % label, 'w') as f:
            json.dump(ensemble, f)
            log('Saved in json format as "%s.ensemble".' % label)
        if archive:
            log('Converting training directory into tar archive...')
            archive_directory(trainingpath, remove_dir)
            log('...converted.')
        return {'complete': True}

    def get_potential_energy(self, atoms, output=(.5,), k=None, mean=True):
        """Returns the potential energy from the ensemble for the atoms
        object.

        By default only returns the average prediction of the ensemble,
        such that it works like a normal ASE calculator.
        To get uncertainty information, use the output keyword with the
        following codes:

            <q>: (where <q> is a float) return the q quantile of the
            ensemble (where the quantile is a decimal, as in 0.5 for 50th
            percentile)

            e: return the whole ensemble prediction as a list

        Join the arguments with commas. For example, to return the median
        prediction plus a centered spread covering 90% of the ensemble
        prediction, use output=[.5, .05, .95].
        If the ensemble is requested, it must be the last argument, e.g.,
        output=[.5, .025, .975, 'e'].
        Note a list is typically returned, but if only one attribute is
        requested it returns it as a float, so that it's ASE-like.
        """
        ensembles = self.ensemble if k is None else self.ensemble[:k]
        energies = [calc.get_potential_energy(atoms) for calc in ensembles]
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Quantiles must be between 0 and 1.')
        result = mquantiles(energies, prob=quantiles)
        result = list(result)
        if return_ensemble:
            result.append(energies)
        if len(result) == 1:
            result = result[0]
            if mean:
                result = np.mean(energies)
        return result

    def get_excess_Ne(self, atoms, output=(.5,), k=None, mean=True):
        """Returns the number of excess electrons from the ensemble for the atoms
        object.

        By default only returns the average prediction of the ensemble,
        such that it works like a normal ASE calculator.
        To get uncertainty information, use the output keyword with the
        following codes:

            <q>: (where <q> is a float) return the q quantile of the
            ensemble (where the quantile is a decimal, as in 0.5 for 50th
            percentile)

            e: return the whole ensemble prediction as a list

        Join the arguments with commas. For example, to return the median
        prediction plus a centered spread covering 90% of the ensemble
        prediction, use output=[.5, .05, .95].
        If the ensemble is requested, it must be the last argument, e.g.,
        output=[.5, .025, .975, 'e'].
        Note a list is typically returned, but if only one attribute is
        requested it returns it as a float, so that it's ASE-like.
        """
        ensembles = self.ensemble if k is None else self.ensemble[:k]
        # The get_charges function returns the atomic charges to be consistant
        # with normal ASE calculators.
        excess_Nes = [-np.sum(calc.get_charges(atoms)) for calc in ensembles]
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Quantiles must be between 0 and 1.')
        result = mquantiles(excess_Nes, prob=quantiles)
        result = list(result)
        if return_ensemble:
            result.append(excess_Nes)
        if len(result) == 1:
            result = result[0]
            if mean:
                result = np.mean(excess_Nes)
        return result

    def get_forces(self, atoms, output=(.5,), k=None, mean=True):
        """Returns the atomic forces from the ensemble for the atoms
        object.

        By default only returns the mean prediction of the ensemble,
        such that it works like a normal ASE calculator.
        To get uncertainty information, use the output keyword with the
        following codes:

            <q>: (where <q> is a float) return the q quantile of the
            ensemble (where the quantile is a decimal, as in 0.5 for 50th
            percentile)

            e: return the whole ensemble prediction as a list

        Join the arguments with commas. For example, to return the median
        prediction plus a centered spread covering 90% of the ensemble
        prediction, use output=[.5, .05, .95].
        If the ensemble is requested, it must be the last argument, e.g.,
        output=[.5, .025, .97.5, 'e'].
        Note a list is typically returned, but if only one attribute is
        requested it returns it as a float, so that it's ASE-like.
        """
        ensembles = self.ensemble if k is None else self.ensemble[:k]
        forces = [calc.get_forces(atoms) for calc in ensembles]
        forces = np.array(forces)
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Quantiles must be between 0 and 1.')
        # FIXME/ap: Had to switch to np.percentile from scipy mquantiles.
        # Because mquantiles doesn't support higher dimensions.
        # Should probably switch to percentiles throughout the code as
        # it's easier to read.
        percentiles = np.array(quantiles) * 100.
        result = np.percentile(forces, percentiles, axis=0)
        result = list(result)
        if return_ensemble:
            result.append(forces)
        if len(result) == 1:
            result = result[0]
            if mean:
                result = np.mean(forces, axis=0)
        return result

    def get_atomic_energies(self, atoms, output=(.5,), k=None, mean=True):
        """ Returns the energy per atom from ensemble.
        The output parameter works as get_potential_energy."""
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Percentiles must be between 0 and 1.')
        self.get_potential_energy(atoms, k=k)  # Assure calculation is fresh.
        ensembles = self.ensemble if k is None else self.ensemble[:k]
        atomic_energies = np.array([calc.model.atomic_energies for calc in
                                    ensembles])
        result = mquantiles(atomic_energies, prob=quantiles, axis=0)
        result = list(result)
        if return_ensemble:
            result.append(atomic_energies)
        if len(result) == 1:
            result == result[0]
            if mean:
                result = np.mean(atomic_energies, axis=0)
        return result

    def get_charges(self, atoms, output=(.5,), k=None, mean=True):
        """ Returns the atomic charges from ensemble. The charges are
            given in units of the negative electron charge (i.e.,
            -.1 means one electron more than the neutral.
            The output parameter works as get_excess_Ne."""
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Percentiles must be between 0 and 1.')
        self.get_excess_Ne(atoms, k=k)  # Assure calculation is fresh.
        ensembles = self.ensemble if k is None else self.ensemble[:k]
        atomic_charge = np.array([calc.model.atomic_charges for calc in
                                  ensembles])
        result = mquantiles(atomic_charge, prob=quantiles, axis=0)
        result = list(result)
        if return_ensemble:
            result.append(atomic_charges)
        if len(result) == 1:
            result == result[0]
            if mean:
                result = np.mean(atomic_charge, axis=0)
        return result

    def _generate_training_images(self, trainingpath, images, n,
                                  charge_training=False):
        """Generate bootstrap training images."""
        log = self.log
        if isinstance(images, list):
            log('Training set: ' + str(len(images)) + ' images')
        elif isinstance(images, str):
            log('Training set: ' + str(images))
        if charge_training:
            images = hash_with_potential(images)
        else:
            images = hash_images(images)
        log('%i images in training set after hashing.' % len(images))
        image_keys = list(images.keys())

        trajpath = os.path.join(trainingpath, 'training-images')
        if not os.path.exists(trainingpath):
            os.mkdir(trainingpath)
        if not os.path.exists(trajpath):
            os.mkdir(trajpath)

        log('Creating bootstrapped training images in %s.' % trajpath)
        for index in range(n):
            log(' Choosing images for %i.' % index)
            chosen = bootstrap(image_keys)
            log(' Writing trajectory for %i.' % index)
            traj = ase.io.Trajectory(
                os.path.join(trajpath, '%i.traj' % index), 'w')
            for key in chosen:
                traj.write(images[key])

    def _accelerated_resample(self, trainingpath, images, n, new_images,
                              charge_training=False):
        """New image sampling based on an accelerated re-training scheme:
        see the SI of https://doi.org/10.1039/C7CP00375G for more details."""
        if not os.path.exists(trainingpath):
            raise RuntimeError('Training path does not exist.')
        log = self.log
        trajpath = os.path.join(trainingpath, 'training-images')
        log('Accelerated retraining scheme...')
        if isinstance(new_images, str):
            log('Images to be added: ' + str(new_images))
        if charge_training:
            images = hash_with_potential(images)
            image_keys = list(images.keys())
            new_images = hash_with_potential(new_images)
            new_image_keys = list(new_images.keys())
        else:
            images = hash_images(images)
            image_keys = list(images.keys())
            new_images = hash_images(new_images)
            new_image_keys = list(new_images.keys())
        length_old, length_new = len(images), len(new_images)
        if length_new == 0:
            log('No new image found, not changing the trajectory.')
            return
        traj = ase.io.read(
                os.path.join(trajpath, '0.traj'), index=':')
        length_traj = len(traj)
        if length_traj > length_old:
            length_old = length_traj
        log(f'{length_new} new images after hashing added.')
        length = (length_new + length_old)
        r = length_new / length
        log(f'{length} images in the updated trajectory.')
        log('Creating bootstrapped retraining images in %s.' % trajpath)
        for index in range(n):
            traj_old = ase.io.read(
                os.path.join(trajpath, '%i.traj' % index), index=':')
            temp_images = []
            keys_for_new_images = bootstrap(new_image_keys, size=length)
            for i, atoms in enumerate(traj_old):
                if np.random.rand() < r:
                    temp_images.append(new_images[keys_for_new_images[i]])
                else:
                    temp_images.append(atoms)
            keys_for_old_images = bootstrap(image_keys, size=length_new)
            for j in range(length_new):
                if np.random.rand() < r:
                    temp_images.append(new_images[keys_for_new_images[-(j+1)]])
                else:
                    temp_images.append(images[keys_for_old_images[j]])

            traj = ase.io.Trajectory(
                os.path.join(trajpath, '%i.traj' % index), 'w')
            for _ in temp_images:
                traj.write(_)

    def _submit_jobs(self, n, trainingpath, start_command,
                     sleep=0.1, **script_dict):
        log = self.log
        from_scratch = self.from_scratch
        originalpath = os.getcwd()
        log('Creating and submitting jobs.')
        os.chdir(trainingpath)
        template = Template(script)
        pwd = os.getcwd()
        cond = len(self.ensemble) > 0
        for index in range(n):
            if not os.path.exists('%i' % index):
                os.mkdir('%i' % index)
            os.chdir('%i' % index)
            if not from_scratch:
                if cond:
                    # initialize calc with existing parameters
                    _calc = self.ensemble[index]
                    _calc.save(filename=make_filename(_calc.label,
                               '-checkpoint.amp'), overwrite=True)
            with open('run.py', 'w') as f:
                f.write(template.substitute(script_dict))
            os.system(start_command)
            time.sleep(sleep)
            os.chdir(pwd)
        self.ensemble = []
        os.chdir(originalpath)

    def _retrain_preprocess(self, n, trainingpath):
        pwd = os.getcwd()
        os.chdir(trainingpath)
        fulltrainingpath = os.getcwd()
        for index in range(n):
            os.chdir('%i' % index)
            # amp automatically reads checkpoint calculator
            if os.path.exists('amp.amp'):
                os.system('mv amp.amp amp-checkpoint.amp')
            elif not os.path.exists('amp-checkpoint.amp'):
                raise RuntimeError("No checkpoint Amp calculator.")
            for _ in os.listdir(os.getcwd()):
                if _ != 'run.py' and _ != 'amp-checkpoint.amp':
                    if os.path.isdir(_):
                        if not _.endswith('ampdb'):
                            shutil.rmtree(_)
                    else:
                        os.remove(_)
            os.chdir(fulltrainingpath)
        os.chdir(pwd)

    def _add_nft_ids(self, trainingpath, nft_ids):
        nft_path = os.path.join(trainingpath, 'saved-info')
        if not os.path.exists(nft_path):
            os.mkdir(nft_path)
        nft_pickle = 'nft_ids.pkl'
        nft_file = os.path.join(nft_path, nft_pickle)
        if os.path.exists(nft_file):
            with open(nft_file, 'rb') as pf:
                old_nft_ids = pickle.load(pf)
            for nft_id in nft_ids:
                if nft_id not in old_nft_ids:
                    old_nft_ids.append(nft_id)
            nft_ids = old_nft_ids
        with open(nft_file, 'wb') as pf:
            pickle.dump(nft_ids, pf)


def bootstrap(vector, size=None, return_missing=False):
    """Returns a randomly chosen, with replacement, version of the data
    set. If size is None returns a vector of same length.
    To pull from sample from multiple vectors, zip and unzip them like:

    >>> xsbs, ysbs = zip(*bootstrap(zip(xs, ys)))

    If return_missing == True, also finds and returns the missing elements
    not sampled from the vector as a second output.
    """

    size = len(vector) if size is None else size
    ids = np.random.choice(len(vector), size=size, replace=True)
    chosen = [vector[_] for _ in ids]
    if return_missing is False:
        return chosen
    unchosen = set(range(len(vector))).difference(set(ids))
    unchosen = [vector[_] for _ in unchosen]
    return chosen, unchosen


def archive_directory(source_dir, remove_dir=False, suffix=''):
    """Turns <source_dir> into a .tar.gz file and removes the original
    directory."""
    outputname = source_dir + suffix + '.tar.gz'
    if os.path.exists(outputname):
        pass
        # raise RuntimeError('%s exists.' % outputname)
    with tarfile.open(outputname, 'w:gz') as tar:
        tar.add(source_dir)
    if remove_dir:
        shutil.rmtree(source_dir)


class TrainingArchive:
    """Helper to get training trajectories and Amp calc instances from the
    training tar ball. Initialize with archive name. The get commands use
    the path the file would have had if the archive were extracted."""

    def __init__(self, name):
        self.tf = tarfile.open(name)

    def get_trajectory(self, path):
        # Doesn't work with extractfile because of numpy bug.
        tempdir = tempfile.mkdtemp()
        self.tf.extract(member=path, path=tempdir)
        return ase.io.Trajectory(os.path.join(tempdir, path))

    def get_amp_calc(self, path):
        return Amp.load(self.tf.extractfile(path))
