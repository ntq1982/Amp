import os
import sys
import time
import pickle
import shutil
import numpy as np
from copy import copy
from ase.utils.timing import Timer, timer
from ase.io import read, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

from ..utilities import hash_images, get_hash, Logger
from ..utilities import extract_an_atomic_chunk, get_atomic_uncertainties
from ..stats.bootstrap import BootStrap
from .dft_jobs import submit_dft_jobs


calc_text = """
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(),
           dblabel='../amp-db')
calc.model.lossfunction.parameters['weight_duplicates'] = False
"""

start_command = 'python3 run.py &'

train_line = "calc.train(images=trainfile)"

headerlines = ''

nft_ids = None

dft_command = 'gpaw-submit -n 1 -c 24 -t 50:00:00 -m 40G run.py gpaw'


class NFT:
    """An automatic protocol, which trains a bootstrap ensemble calculator
    using the nearsighted force-training approach. Logging with an
    `amp.utilities.Logger` instance.

    If an ensemble is given, it will be loaded and used as the starting
    model for training the initial images.

    Parameters
    ----------
    load: str
        If an ensemble model is given, it will be loaded as the initial model
        for training the initial images.
    log: str or an amp.utilities.Logger instance
        Logging file.
    threshold: float
        Controls the number of atomic chunks to be evaluated by single-point
        calculations. If threshold is positive, the chunks centering on atoms
        whose atomic uncertainty is above the threshold will be calculated.
        If threshold is in the range of (-1.0, 0.), 100*abs(1+threshold)
        percent of all possible atomic chunks will be calculated. For example,
        `threshold=-0.9` indicates that chunks with top 10 percent
        uncertainties will be calculated.
    stop_delta: float
        Termination criterion---if the maximum atomic uncertainty is below the
        `stop_delta`, the NFT iteration is stopped.
    max_iterations: int
        Termination criterion---if the number of NFT iterations is above
        `max_iterations`, the NFT iteration is stopped.
    steps_not_improved: int
        Termination criterion---if the structure uncertainty has not improved
        for consecutive `steps_not_improved`, the NFT iteration is stopped.
    dblabel : str
        Optional separate prefix/location for database files, including
        fingerprints, fingerprint derivatives, and neighborlists.
    """
    def __init__(self, load=None, log=None, threshold=None, stop_delta=.1,
                 max_iterations=5, steps_not_improved=2, dblabel='amp-data'):
        self.timer = Timer()
        if log is None:
            log = Logger(sys.stdout)
        elif isinstance(log, str):
            log = Logger(log)
        self.log = log
        self.threshold = threshold
        self.stop_delta = stop_delta
        self.max_iterations = max_iterations
        self.steps_not_improved = steps_not_improved
        self.dblabel = dblabel
        self.calc = BootStrap(load=load, log=log)

    def __del__(self):
        """ Write timings to log file when calculator is closed."""
        if hasattr(self, 'timer') and self.log.file is not None:
            self.timer.write(self.log.file)

    @timer('Fit ensemble activelearner')
    def run(self, images, target_image, n=10, calc_text=calc_text,
            headerlines=headerlines, start_command=start_command,
            train_line=train_line, label='al', parent_calc=None,
            expired=600., cutoff=6.5, init_nft_ids=None,
            dft_cores=None, dft_memory=None):
        """Trains a bootstrap ensemble in the NFT framework.

        For bootstrap training jobs, it can be submitted sequentially
        or in parallel, depending on the start_command.

        As for single-point calculations, if simple calculator, for example
        EMT, is used, it will be calculated sequentially. In comparison,
        for DFT calculations, single-point calculations are submitted
        independently.

        The NFT iteration is terminated if either stopping criterion is met.
        The model with the lowest structure uncertainty is saved in a 'json'
        file named 'best.[label].ensemble'.

        Parameters
        ----------
        images : list or str
            List of ASE atoms objects with positions, symbols, energies, and
            forces in ASE format. This is the initial training data, for
            example simple bulk cells.
        target_images: str or list
            List of ASE atoms objects which should only have one atoms object,
            which is the target large structure to be learned by NFT.
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
        train_line: str
           line to use to train each amp instance; usually the default is
           fine but user may want to use this to insert additional keywords
           such as train_forces=False
        label: string
           label to give final trained calculator
        parent_calc: instance
            a parent calculator instance. For example `EMT()`.
        expired: float
           When checking jobs, age (s) of log file at which to consider
           that the job is no longer running (timed out) and should be
           restarted.
        cutoff: float
            Cutoff radius to extract atomic chunks.
        init_nft_ids:
            list of length-two tuples to specify initial force only ids.
            In cases where the initial training data consists of some images
            which require nearsighted force training, it should be supplied.
        dft_cores: int
            Number of DFT cores to be requested for DFT jobs.
        dft_memory: str
            Amount of DFT memory per node to be requested. For example, '40G'
            indicates that memory of 40 Gigabytes per node will be requested.

        Returns
        -------
        boolean: whether the NFT is converged on the target image.
        """
        calc = self.calc
        threshold = self.threshold
        stop_delta = self.stop_delta
        max_iterations = self.max_iterations
        steps_not_improved = self.steps_not_improved
        dblabel = self.dblabel
        load_file = label + '.ensemble'
        target_image = hash_images(target_image)
        target_image = list(target_image.values())[0]
        images = hash_images(images)
        images = list(images.values())
        # this trainingpath will be created during the initial training.
        trainingpath = f"{label}-training"
        # Fit energy and forces on the initial images
        # Train an initial ensemble model
        self.timer.start('initial training')
        complete = False
        _count = 0
        while not complete:
            results = calc.train(images=images,
                                 n=n,
                                 calc_text=calc_text,
                                 start_command=start_command,
                                 label=label,
                                 headerlines=headerlines,
                                 train_line=train_line,
                                 nft_ids=init_nft_ids,
                                 archive=False,
                                 expired=expired)
            _count += 1
            complete = results['complete']
            if not complete:
                calc.log('Initial train loop: ' + str(_count))
                time.sleep(60.)
        self.timer.stop('initial training')

        # path to save NFT results, including
        # - atomic chunks at each iteration is saved in a trajectory file
        # - all uncertainty results are saved in a numpy data file (deltas.npy)
        # - chunk indices in the target image are saved in numpy files
        saved_info_path = os.path.join(trainingpath, 'saved-info')
        self._make_chunk_indexing_dict(saved_info_path, target_image, cutoff)
        # read existing uncertainty results.
        if os.path.exists(f'{saved_info_path}/deltas.npy'):
            with open(f'{saved_info_path}/deltas.npy', 'rb') as f:
                deltas = list(np.load(f))
                self._step = len(deltas) - 1
            calc.log('Uncertainty file exists.'
                     f'Restart from the step {self._step}.')
        else:
            deltas = []
            self._step = 0

        # collect existing saved results.
        # nft_ids is a list of length-two tuples. The 1st element is
        # the hash id of the chunk, and the 2nd element is the index in the
        # chunk whose single force  will be trained by the NFT approach.
        # seen_atomic_chunks and seen_nft_ids are dictionaries for
        # fast indexing.
        calc.log('Collect existing atomic chunks.This is helpful for '
                 'restarting a terminated job.')
        nft_ids, seen_atomic_chunks, seen_nft_ids = \
            self._collect_saved_info(saved_info_path)
        calc.log(f'# atomic chunks collected: {len(seen_atomic_chunks)}')

        # Nearsighted force training iterations
        best_step = 0
        lowest_max_uncertainty = np.inf
        self.timer.start('NFT iteration')
        while self._step <= max_iterations:
            calc.log(" ")
            calc.log("="*12)
            calc.log(f"Iteration {self._step}")
            calc.retrain_initiated = False
            atomic_chunks = []
            # indices: the index of an uncertain central atom in the target
            # for which a chunk is carved out.
            indices, _, hs_all = get_atomic_uncertainties(load_file,
                                                          target_image,
                                                          threshold=threshold,
                                                          label=dblabel)
            deltas.append(hs_all)
            with open(f'{saved_info_path}/deltas.npy', 'wb') as f:
                np.save(f, np.array(deltas))
            calc.log(f'# chunks at iter {self._step}: {len(indices)}')
            np.save(f"{saved_info_path}/indices_{self._step}.npy", indices)
            if max(hs_all) < lowest_max_uncertainty:
                best_step = self._step
                lowest_max_uncertainty = max(hs_all)
                command = f'cp {load_file} best.{label}.ensemble'
                os.system(command)
                calc.log(f'Best model saved as best.{label}.ensemble, '
                         f'structure uncertainty: {max(hs_all):.2f}.')
            else:
                calc.log(f'structure uncertainty at iter {self._step}: '
                         f'{max(hs_all):.2f}')
            calc.log(f"Save current model to {label}.{self._step}.ensemble")
            _command = f"cp {label}.ensemble {label}.{self._step}.ensemble"
            os.system(_command)
            if max(hs_all) < stop_delta:
                calc.log(f'Ensemble calculator {load_file} converged with '
                         f'structure uncertainty: {max(hs_all):.2f}')
                self.timer.stop('NFT iteration')
                return True
            elif self._step >= best_step + steps_not_improved:
                calc.log(f'Model has not improved for {steps_not_improved}'
                         ' steps.')
                calc.log('Terminate the NFT iterations.')
                self.timer.stop('NFT iteration')
                return False
            self.timer.start(f'SP calculation for iter {self._step}')
            if parent_calc.name == 'gpaw':
                self._prepare_dft_jobs_to_submit(indices, parent_calc)

            for index in indices:
                if index not in seen_atomic_chunks:
                    calc.log(f"Submit job for index: {str(index)}")
                    # For EMT, the returned atomic chunk already has
                    # force and energy properties.
                    atomic_chunk, nft_id = extract_an_atomic_chunk(
                            target_image, index, parent_calc=parent_calc,
                            cutoff=cutoff)
                    # For GPAW, independent jobs need to be submitted.
                    if parent_calc.name == 'gpaw':
                        dftnewimagespath = "dft-jobs/dft-images"
                        traj = Trajectory(f"{dftnewimagespath}/{index}.traj",
                                          'w')
                        traj.write(atomic_chunk)
                        submit_dft_jobs(index=index, cores=dft_cores,
                                        memory=dft_memory)
                    # atomic chunk requiring dft calculations does not have
                    # energy and force properties till now.
                    seen_atomic_chunks[index] = atomic_chunk
                    seen_nft_ids[index] = nft_id
                    nft_ids.append(nft_id)
                else:
                    atomic_chunk = seen_atomic_chunks[index]
                    nft_id = seen_nft_ids[index]
            if parent_calc.name == 'gpaw':
                complete = False
                while not complete:
                    results = self._check_dft_jobs(indices)
                    complete = results['complete']
                    if not complete:
                        time.sleep(300./(self._step + 1))
                # Now dft seen_atomic_chunks have force and energy properties.
                seen_atomic_chunks = self._assign_dft_energy_and_forces(
                                                indices, seen_atomic_chunks)
            for index in indices:
                atomic_chunks.append(seen_atomic_chunks[index])
            self.timer.stop(f'SP calculation for iter {self._step}')
            calc.log('Save atomic chunks with high uncertainty...')
            self._insert_new_images(saved_info_path, atomic_chunks,
                                    nft_ids)
            calc.log(f'...saved in {str(saved_info_path)}')
            complete = False
            _count = 0
            while not complete:
                results = calc.train(images=images,
                                     n=n,
                                     new_images=atomic_chunks,
                                     calc_text=calc_text,
                                     start_command=start_command,
                                     label=label,
                                     headerlines=headerlines,
                                     train_line=train_line,
                                     nft_ids=nft_ids,
                                     archive=False,
                                     expired=expired)
                _count += 1
                complete = results['complete']
                if not complete:
                    calc.log('NFT retrain loop: ' + str(_count))
                    time.sleep(60.)
            self._step += 1
            images.extend(atomic_chunks)

        calc.log(f'Number of iterations exceeds allowed {max_iterations}.'
                 f' Current structure uncertainty: {max(hs_all):.2f}.')
        self.timer.stop('NFT iteration')
        return False

    def _make_chunk_indexing_dict(self, saved_info_path, target_image, cutoff):
        """Create a pickle file to save a dictionary.
        Keys are hash ids of chunks extracted from a target image for a
        given cutoff radius, and values are corresponding indices. """
        no_atoms = len(target_image)
        indexing_dict = {}
        if not os.path.exists(saved_info_path):
            os.mkdir(saved_info_path)
        file = f'{saved_info_path}/indexing_dict.pkl'
        for index in range(no_atoms):
            atomic_chunk, nft_id = extract_an_atomic_chunk(
                        target_image, index, cutoff=cutoff)
            hash_id = nft_id[0]
            indexing_dict[hash_id] = int(index)
        if os.path.exists(file):
            with open(file, 'rb') as pf:
                ori_indexing_dict = pickle.load(pf)
            indexing_dict.update(ori_indexing_dict)
        with open(file, 'wb') as pf:
            pickle.dump(indexing_dict, pf)

    def _prepare_dft_jobs_to_submit(self, indices, parent_calc):
        """Prepare folders and files to submit all dft jobs at
        the same time."""
        log = self.calc.log
        dftjobpath = "dft-jobs"
        dftnewimagespath = f"{dftjobpath}/dft-images"
        log(f'Prepare dft jobs to submit for iter {self._step}.')
        if not os.path.exists(dftjobpath):
            os.mkdir(dftjobpath)
        if not os.path.exists(dftnewimagespath):
            os.mkdir(dftnewimagespath)
        originalpath = os.getcwd()
        os.chdir(dftjobpath)
        if not os.path.exists('gpaw_params.pkl'):
            with open('gpaw_params.pkl', 'wb') as f:
                pickle.dump(parent_calc.parameters, f)
        pwd = os.getcwd()
        sleep = 1.
        for index in indices:
            if not os.path.exists('%i' % index):
                os.mkdir('%i' % index)
            os.chdir('%i' % index)
            time.sleep(sleep)
            os.chdir(pwd)
        os.chdir(originalpath)

    def _collect_saved_info(self, saved_info_path):
        """Collect existing nft info from saved files. This is helpful
         for restarting a terminated job."""
        nft_ids = []
        seen_atomic_chunks = {}
        seen_nft_ids = {}
        indexing_file = f'{saved_info_path}/indexing_dict.pkl'
        with open(indexing_file, 'rb') as pf:
            indexing_dict = pickle.load(pf)
        pwd = os.getcwd()
        os.chdir(saved_info_path)
        new_images_traj = 'atomic_chunks.traj'
        if os.path.exists(new_images_traj):
            new_images = read('atomic_chunks.traj', index=':')
            for atomic_chunk in new_images:
                hash_id_of_chunk = get_hash(atomic_chunk)
                nft_id = (hash_id_of_chunk, 0)
                nft_ids.append(nft_id)
                index = indexing_dict[hash_id_of_chunk]
                seen_nft_ids[index] = nft_id
                seen_atomic_chunks[index] = atomic_chunk
        os.chdir(pwd)
        return nft_ids, seen_atomic_chunks, seen_nft_ids

    def _insert_new_images(self, saved_info_path, atomic_chunks,
                           nft_ids):
        """Save new images at each NFT step to a separate trajectory file."""
        step = self._step
        trajectory = 'atomic_chunks.traj'
        _trajectory = f'atomic_chunks_{step}.traj'
        file = os.path.join(saved_info_path, trajectory)
        _file = os.path.join(saved_info_path, _trajectory)
        new_images = copy(atomic_chunks)
        if os.path.exists(file):
            old_new_images = read(file, index=':')
            new_images.extend(old_new_images)
        nft_pickle = 'nft_ids.pkl'
        nft_file = os.path.join(saved_info_path, nft_pickle)
        if os.path.exists(nft_file):
            with open(nft_file, 'rb') as pf:
                old_nft_ids = pickle.load(pf)
            for nft_id in nft_ids:
                if nft_id not in old_nft_ids:
                    old_nft_ids.append(nft_id)
            nft_ids = old_nft_ids
        with open(nft_file, 'wb') as pf:
            pickle.dump(nft_ids, pf)
        new_images = hash_images(new_images)
        new_images = list(new_images.values())
        traj = Trajectory(file, 'w')
        for image in new_images:
            traj.write(image)
        command = f'cp {file} {_file}'
        os.system(command)

    def _check_dft_jobs(self, indices, expired=600., dft_command=dft_command):
        """Check if all dft jobs are finished. This method is adapted from
        the `_manage_jobs` method in the `BootStrap` module."""
        def clean_and_restart():
            sleep = 1.
            for _ in os.listdir(os.getcwd()):
                if _ != 'run.py':
                    if os.path.isdir(_):
                        shutil.rmtree(_)
                    else:
                        os.remove(_)
            os.system(dft_command)
            time.sleep(sleep)
            log('  ---> restarted.')
        log = self.calc.log
        n = len(indices)
        n_unfinished = 0
        n_finished = 0
        n_notstarted = 0
        n_expired = 0
        pwd = os.getcwd()
        dftjobpath = "dft-jobs"
        os.chdir(dftjobpath)
        fulldftpath = os.getcwd()
        for index in indices:
            os.chdir('%i' % index)
            if not os.path.exists('completed'):
                if not os.path.exists('%i.txt' % index):
                    log('%i: Not started; no output file.' % index)
                    n_notstarted += 1
                else:
                    age = time.time() - os.path.getmtime('%i.txt' % index)
                    log('{:d}: Still running? No converged file. Age: '
                        '{:.1f} hr'.format(index, age / 3600.))
                    if age > expired:
                        log(' Assumed expired. Cleaning up directory and '
                            'restarting.')
                        n_expired += 1
                        clean_and_restart()
                    else:
                        n_unfinished += 1
                os.chdir(fulldftpath)
                continue
            with open('completed') as f:
                completed = f.read()

            if completed == 'True':
                log('%i: Completed.' % index)
                n_finished += 1
            else:
                os.chdir(fulldftpath)
                continue
            os.chdir(fulldftpath)
        log('')
        log('Stats:')
        log('%10i dft jobs completed' % n_finished)
        log('%10i dft jobs not yet started' % n_notstarted)
        log('%10i dft jobs expired, restarted' % n_expired)
        log('%10i dft jobs apparently still running' % n_unfinished)
        log('=' * 10)
        log('%10i total dft jobs' % n)
        log('\n')
        if n_finished < n:
            log('Not all dft jobs completed; not assigning energy '
                'and forces to uncertain atomic chunks.')
            os.chdir(pwd)
            return {'complete': False,
                    'n_finished': n_finished}
        log('All dft jobs completed.')
        os.chdir(pwd)
        return {'complete': True}

    def _assign_dft_energy_and_forces(self, indices, seen_atomic_chunks):
        """If all dft jobs are finished, assign energy and forces to
        uncertain atomic chunks."""
        log = self.calc.log
        pwd = os.getcwd()
        dftjobpath = "dft-jobs"
        os.chdir(dftjobpath)
        fulldftpath = os.getcwd()
        log("Assign dft energy and forces")
        for index in indices:
            os.chdir('%i' % index)
            with open("results", "rb") as f:
                results = pickle.load(f)
            energy = results['energy']
            forces = results['forces']
            atomic_chunk = seen_atomic_chunks[index]
            sp = SinglePointCalculator(atomic_chunk, energy=energy,
                                       forces=forces)
            atomic_chunk.set_calculator(sp)
            seen_atomic_chunks[index] = atomic_chunk
            os.chdir(fulldftpath)
        os.chdir(pwd)
        return seen_atomic_chunks
