import os
import sys
import shutil
import numpy as np
from ase.io import Trajectory
from scipy.spatial import distance_matrix

from ..utilities import hash_images, Logger
from ..descriptor.gaussian import Gaussian, make_default_symmetry_functions
from ..descriptor.zernike import Zernike

# Image selection using FurthestPointSampling


class FurthestPointSampling(object):
    """
    Hierarchical Furthest Point Sampling algorithm is a general selection
    technique.

    To search for informative points which are spread out and can largely
    represent the original dataset.

    Parameters
    ----------
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and
        forces in ASE format. This is the training set of data. This can
        also be the path to an ASE trajectory (.traj) or database (.db)
        file. Energies can be obtained from any reference, e.g. DFT
        calculations.
    k: int
        Number of images to be selected.
    encoder: str
        Method for encoding the atomic configurations. Available methods are
        'gaussian' and 'zernike'.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.

    Notes
    -----

    The following norms can be calculated:
    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    Returns
    --------
    Selected images saved in a trajectory file.
    """
    def __init__(self, images, k=None, encoder='gaussian', order=2, log=None,
                 cutoff=None):
        images = self.images = hash_images(images)
        if k is None:
            k = int(0.2*len(images))
        if k > len(images):
            raise RuntimeError('Not able to select more images than allowed!')
        self.k = k
        if log is None:
            log = Logger(sys.stdout)
        if isinstance(log, str):
            log = Logger(log)
        self._log = log
        if cutoff is None:
            cutoff = 3.5
        self.cutoff = cutoff
        log('Image selection')
        log('='*15)
        log('Furthest Point Sampling initiated...')
        log(f'Cutoff radius: {cutoff:.2f}')
        log(f'{k} images to be selected out of {len(images)}')
        self.ord = order
        self.encoder = encoder.lower()
        # Construct feature matrices with encoder
        self.uan, self.tan, self.fm = self._encoding()

    def _encoding(self):
        log = self._log
        images = self.images
        encoder = self.encoder
        cutoff = self.cutoff
        elements = list(set([atom.symbol for atoms in images.values()
                             for atom in atoms]))
        elements = sorted(elements)
        # 1st level search: unique atomic numbers
        uan = [sorted(list(set(atoms.numbers))) for atoms in images.values()]
        uan = self._add_pad(uan)
        # 2nd level search: total atomic numbers
        tan = [atoms.numbers for atoms in images.values()]
        tan = self._add_pad(tan)

        # 3rd level search: feature matrices of given descriptor
        if encoder == 'gaussian':
            log(' Encoding images with Gaussian descriptor')
            G = make_default_symmetry_functions(elements)
            gaussian = Gaussian(Gs=G, cutoff=cutoff, dblabel=encoder)
            gaussian.calculate_fingerprints(images)
            fps_data = gaussian.fingerprints
            fps_data.open()
            fps = list(fps_data.d.values())
            fps_data.close()
            fp_concatenated = []
            for fp in fps:
                fp_concatenated.append(np.hstack([_ for symbol, _ in fp]))
            fp_concatenated = self._add_pad(fp_concatenated)
            fm = np.array(fp_concatenated)
            fm = fm[:, fm.var(axis=0) > 1e-3]
        elif encoder == 'zernike':
            log(' Encoding images with Zernike descriptor')
            zernike = Zernike(cutoff=cutoff, nmax=3, dblabel=encoder)
            zernike.calculate_fingerprints(images)
            fps_data = zernike.fingerprints
            fps_data.open()
            fps = list(fps_data.d.values())
            fps_data.close()
            fp_concatenated = []
            for fp in fps:
                fp_concatenated.append(np.hstack([_ for symbol, _ in fp]))
            fp_concatenated = self._add_pad(fp_concatenated)
            fm = np.array(fp_concatenated)
            fm = fm[:, fm.var(axis=0) > 1e-3]
        else:
            raise NotImplementedError(f'Encoder {encoder} not supported.')

        self._cleanup()
        return uan, tan, fm

    def search(self, calculate_dev=False, save_traj=False):
        log = self._log
        order = self.ord
        fm = self.fm
        uan = self.uan
        tan = self.tan
        dm = distance_matrix(fm, fm, p=order)
        uanm = distance_matrix(uan, uan)
        tanm = distance_matrix(tan, tan)
        images = self.images.copy()
        k = self.k
        remaining_indices = list(range(len(images)))
        # Select the first point randomly
        chosen = []
        ind = np.random.randint(0, len(remaining_indices))
        chosen.append(ind)
        remaining_indices.pop(ind)
        count = 1
        while count < k:
            res, res_uan, res_tan = [], [], []
            for i in remaining_indices:
                uan_min = np.inf
                tan_min = np.inf
                d_min = np.inf
                for j in chosen:
                    uan_per = uanm[i, j]
                    tan_per = tanm[i, j]
                    d = dm[i, j]
                    if uan_per < uan_min:
                        uan_min = uan_per
                    if tan_per < tan_min:
                        tan_min = tan_per
                    if d < d_min:
                        d_min = d
                res_uan.append((i, uan_min))
                res_tan.append((i, tan_min))
                res.append((i, d_min))
            res_uan_max = max(res_uan, key=lambda x: x[1])
            res_tan_max = max(res_tan, key=lambda x: x[1])
            res_max = max(res, key=lambda x: x[1])
            if res_uan_max[1] > 0:
                res_ind = res_uan_max[0]
            elif res_tan_max[1] > 0:
                res_ind = res_tan_max[0]
            else:
                res_ind = res_max[0]
            chosen.append(res_ind)
            remaining_indices.remove(res_ind)
            count += 1
        chosen_images = images.copy()
        unchosen_images = images.copy()
        for i, key in enumerate(images.keys()):
            if i not in chosen:
                chosen_images.pop(key)
            if i in chosen:
                unchosen_images.pop(key)
        fm_chosen = fm[chosen, :]
        if calculate_dev:
            min_length = self.min_length
            dev = self._metric(fm[:, :min_length], fm_chosen[:, :min_length])
            log(' Relative norm deviation for feature matrices: '
                f'{100*dev:.2f} %.')
        if save_traj:
            if isinstance(save_traj, str):
                traj_file = save_traj
            else:
                traj_file = 'images_chosen.traj'
            unchosen = Trajectory('unchosen.traj', 'w')
            traj = Trajectory(traj_file, 'w')
            for image in chosen_images.values():
                traj.write(image)
            for image in unchosen_images.values():
                unchosen.write(image)
            log(f'Chosen images saved in file {traj_file}.')
            log('Images not chosen saved in file unchosen.traj.')
        return chosen_images

    def distance(self, x, y):
        order = self.ord
        x, y = np.array(x), np.array(y)
        if len(x) == len(y):
            return np.linalg.norm((x - y), ord=order)

    def _cleanup(self):
        encoder = self.encoder
        if encoder == 'naive':
            return
        nn_folder = encoder + '-neighborlists.ampdb'
        fp_folder = encoder + '-fingerprints.ampdb'
        if os.path.isdir(nn_folder):
            shutil.rmtree(nn_folder)
        if os.path.isdir(fp_folder):
            shutil.rmtree(fp_folder)

    def _add_pad(self, ll, constant=-5.):
        """Add constant paddings to the end of list of list."""
        max_length = 0
        min_length = np.inf
        for l in ll:
            if len(l) > max_length:
                max_length = len(l)
            if len(l) < min_length:
                min_length = len(l)
        self.min_length = min_length
        ll_padded = []
        for l in ll:
            l = list(l)
            l = l + (max_length - len(l)) * [constant]
            ll_padded.append(l)
        return ll_padded

    @staticmethod
    def _metric(fm, fm_chosen):
        """Evaluate if the selected images are good.
        Compute the F-norm relative deviation using selected
        images"""
        fm, fm_chosen = fm.T, fm_chosen.T
        # Compute the F-norm relative approximation error
        C = fm_chosen
        R = fm
        # compute the pseudo-inverse
        C_plus = np.linalg.pinv(C)
        R_plus = np.linalg.pinv(R)
        U = np.linalg.multi_dot([C_plus, fm, R_plus])
        fm_hat = np.linalg.multi_dot([C, U, R])
        dev = np.linalg.norm(fm_hat-fm)/np.linalg.norm(fm)
        return dev
