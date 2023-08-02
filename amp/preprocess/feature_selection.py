import os
import sys
import json
import shutil
import numpy as np
from scipy.stats.stats import pearsonr
from operator import itemgetter
from collections import defaultdict
from sklearn.preprocessing import RobustScaler

from ..utilities import hash_images, Logger
from ..descriptor.gaussian import Gaussian, make_default_symmetry_functions
from ..descriptor.gaussian import make_symmetry_functions
from ..descriptor.cutoffs import Cosine, Polynomial

# Symmetry function (feature) selection


class FTSel(object):
    """
    Feature selection module based on either following schemes.

    - CUR approximation
    - Correlation with atomic force magnitude

    Parameters
    ----------
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and
        forces in ASE format. This is the training set of data. This can
        also be the path to an ASE trajectory (.traj) or database (.db)
        file. Energies can be obtained from any reference, e.g. DFT
        calculations.
    k: int or dict
        Number of symmetry functions to be selected. If a single integer
        is given, the same number of SFs are selected for all elements.
        If it is a dictionary, element-wise number of SFs will be selected.
    log: str
        log file for feature selection.
    method: str
        - 'cur': feature selection based on the feature matrices, which only
        rely on the input values (atomic positions).
        - 'fcorr': force correlation method, which related the input to the
        output (atomic forces).
    encoder: dict
        For feature selection, only `Gaussian` decriptor is implemented.
        encoder['params'] stores the key arguments for setting up a `Gaussian`
        descriptor. If it is `None`, default Gaussian symmetry function
        parameters will be used.
    save_json: bool or str
        Save features in a json file ('Gs.json') if it is True. If a string is
        given, it will be taken as a filename.

    Returns
    --------
    Selected symmetry functions saved in a json file.

    """
    def __init__(self, images, k=None, log=None, method='cur',
                 encoder={'descriptor': 'gaussian',
                          'params': None},
                 save_json=True):
        images = self.images = hash_images(images, ordered=True)
        if k is None:
            k = int(0.2*len(images))
        self.k = k
        if log is None:
            log = Logger(sys.stdout)
        if isinstance(log, str):
            log = Logger(log)
        self._log = log
        log('Feature selection')
        log('='*17)
        method = self.method = method.lower()
        if method == 'cur':
            log('CUR approximation initiated...')
        elif method == 'fcorr':
            log('Force correlation method initiated...')
        self.descriptor = encoder['descriptor']
        self.params = encoder['params']
        # Construct feature matrices based off the encoder
        self.fm = self._encoding()
        self.save_json = save_json

    def _encoding(self):
        """Encode the images with the given parameters."""
        descriptor = self.descriptor
        params = self.params
        log = self._log
        images = self.images
        elements = list(set([atom.symbol for atoms in images.values()
                             for atom in atoms]))
        elements = self.elements = sorted(elements)

        if descriptor != 'gaussian':
            raise NotImplementedError('Feature selection for descriptor '
                                      f'{descriptor} not implemented.')
        if params is None:
            Gs = None
            Rc = 5.3  # default cutoff radius in angstrom
            cutoff_fxn = Cosine(Rc)
        else:
            if 'Gs' in params:
                Gs = self._make_Gs(params['Gs'])
            else:
                Gs = make_default_symmetry_functions(elements)
            if 'Rc' in params:
                Rc = params['Rc']
            else:
                Rc = 5.3
            if 'cutoff' in params:
                if params['cutoff'] == 'cosine':
                    cutoff_fxn = Cosine(Rc)
                elif params['cutoff'] == 'polynomial':
                    if 'gamma' in params:
                        gamma = params['gamma']
                    else:
                        gamma = 4.
                    cutoff_fxn = Polynomial(Rc, gamma=gamma)
            else:
                cutoff_fxn = Cosine(Rc)
        log(' Encoding images with Gaussian descriptor')
        gaussian = Gaussian(Gs=Gs, cutoff=cutoff_fxn, dblabel=descriptor)
        gaussian.calculate_fingerprints(images)
        self.Gs = gaussian.parameters.Gs
        fps_data = gaussian.fingerprints
        fps_data.open()
        fps = fps_data.d  # dictionary
        fps_data.close()
        fp_dict = defaultdict(list)
        for hash_id, _ in images.items():
            fp = fps[hash_id]
            # fpe: fingerprint element
            for symbol, fpe in fp:
                fp_dict[symbol].append(fpe)
        for symbol, fp_symbol in fp_dict.items():
            fp_dict[symbol] = np.array(fp_symbol)
        self._cleanup()
        return fp_dict

    def _make_Gs(self, Gs_params):
        """Make symmetry functions based on the parameters."""
        assert set(Gs_params.keys()).issubset(set(['G2', 'G4', 'G5']))
        G = []
        elements = self.elements
        for G_type, params in Gs_params.items():
            if G_type == 'G2':
                if 'offsets' in params.keys():
                    G += make_symmetry_functions(
                            elements=elements,
                            type=G_type,
                            etas=params['eta'],
                            offsets=params['offsets'])
                else:
                    G += make_symmetry_functions(
                            elements=elements,
                            type=G_type,
                            etas=params['eta'])
            else:
                G += make_symmetry_functions(
                        elements=elements,
                        type=G_type,
                        etas=params['eta'],
                        zetas=params['zeta'],
                        gammas=params['gamma'])
        return G

    def search(self, calculate_dev=False):
        """Run the feature selection alrorithm."""
        method = self.method
        log = self._log
        fm_dict = self.fm
        Gs = self.Gs
        k = self.k
        save_json = self.save_json
        chosen_Gs = {}
        fm_chosen_dict = {}
        indices_dict = {}
        forces_dict = self._get_atomic_force()
        self.forces_dict = forces_dict
        for element, fm_ori in fm_dict.items():
            fm = fm_ori.copy()
            if isinstance(k, int):
                ncols = k
            else:
                ncols = k[element]
            if ncols > fm_ori.shape[1]:
                raise RuntimeError('Dimension exceeds the allowed.')
            indices = -np.ones(ncols)
            nf = fm.shape[1]
            # mask features which have almost constant feature values
            masked = (np.var(fm, axis=0) > 1e-3)
            reduced_candidates = np.arange(nf)[masked]
            fm[:, reduced_candidates] = RobustScaler().fit_transform(
                                        fm[:, reduced_candidates])
            for i in range(ncols):
                if method == 'cur':
                    fm, index, reduced_candidates = \
                        self.col_sel(fm, reduced_candidates)
                elif method == 'fcorr':
                    forces = forces_dict[element]
                    forces /= np.linalg.norm(forces)
                    fm, index, reduced_candidates = \
                        self.col_sel_by_fcorr(fm, forces, reduced_candidates)
                indices[i] = index
            indices = indices.astype(int).tolist()
            fm_chosen = fm_ori[:, indices]
            fm_chosen_dict[element] = fm_chosen
            indices_dict[element] = indices
            if calculate_dev:
                dev = self._metric(fm_ori, fm_chosen, transpose=False)
                log(f'{len(indices)} symmetry functions selected out of'
                    f' {fm_ori.shape[1]} candidates')
                log(' Relative norm deviation for feature matrices '
                    f'of element {element} is {100*dev:.2f} %.')
            chosen_Gs[element] = itemgetter(*indices)(Gs[element])
        self.fm_chosen = fm_chosen_dict
        self.indices_chosen = indices_dict
        if save_json:
            if isinstance(save_json, str):
                json_file = save_json
            else:
                json_file = 'Gs.json'
            with open(json_file, 'w') as f:
                json.dump(chosen_Gs, f)
            log(f'Saved in json format as {json_file}.')
        return chosen_Gs

    def _cleanup(self):
        """Clean up the temp data bases for feature selection."""
        descriptor = self.descriptor
        nn_folder = descriptor + '-neighborlists.ampdb'
        fp_folder = descriptor + '-fingerprints.ampdb'
        if os.path.isdir(nn_folder):
            shutil.rmtree(nn_folder)
        if os.path.isdir(fp_folder):
            shutil.rmtree(fp_folder)

    def _get_atomic_force(self):
        """Return a dictionary in which the key is an element,
        and the value is corresponding atomic forces extracted
        from all configurations in a trajectory."""
        images = self.images
        f = defaultdict(list)
        for hash_id, atoms in images.items():
            forces = atoms.get_forces(apply_constraint=False)
            forces = np.linalg.norm(forces, axis=1)
            forces = np.round(forces, 3)
            symbols = atoms.symbols
            for j, symbol in enumerate(symbols):
                force = forces[j]
                f[symbol].append(force)
        return f

    @staticmethod
    def _add_pad(ll, constant=-5.):
        """Add constant paddings to the end of list of list."""
        max_length = 0
        for l in ll:
            if len(l) > max_length:
                max_length = len(l)
        ll_padded = []
        for l in ll:
            l = list(l)
            l = l + (max_length - len(l)) * [constant]
            ll_padded.append(l)
        return ll_padded

    @staticmethod
    def _metric(fm, fm_chosen, transpose=True):
        """Evaluate if the selected images are good.
        Compute the F-norm relative deviation using selected
        images"""
        if transpose:
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

    @staticmethod
    def col_sel(X, candidates, n=3):
        """Select single column (one feature) from feature matrix (X)"""
        fm = X[:, candidates]
        u, sigma, vh = np.linalg.svd(fm)
        prob_cols = np.square(vh)[0:n, :].sum(axis=0)
        index = np.argmax(prob_cols)
        selected_index = candidates[index]
        selected = X[:, selected_index]
        candidates = np.delete(candidates, index)
        for j in candidates:
            X[:, j] = X[:, j] - selected*(np.dot(selected, X[:, j])) /\
                     np.linalg.norm(selected)**2
        return X, selected_index, candidates

    @staticmethod
    def col_sel_by_fcorr(X, forces, candidates):
        """Select one feature which shows the highest force correlation."""
        corrs = np.zeros(len(candidates))
        for _, k in enumerate(candidates):
            corrs[_] = abs(pearsonr(X[:, k], forces)[0])
        index = np.argmax(corrs)
        index_selected = candidates[index]
        candidates = np.delete(candidates, index)
        selected = X[:, index_selected]
        # Orthogonalization other candidates for next step
        for j in candidates:
            X[:, j] = X[:, j] - selected*(np.dot(selected, X[:, j]))\
                / np.linalg.norm(selected)**2
        return X, index_selected, candidates
