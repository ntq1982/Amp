import os
import sys
import time
import numpy as np
from ase.spacegroup import crystal
from ase.calculators.emt import EMT
from ase.constraints import StrainFilter
from ase.optimize import BFGS as qn
from ase.io import read, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

from .dft_jobs import strain_filter_jobs, single_point_jobs
from ..utilities import Logger, enforce_magnetic_moments

# Generate initial training structures

# mapping crystal systems to space group numbers
sg_dict = {
    'triclinic': [1, 2],
    'monoclinic': list(range(3, 16)),
    'orthorhombic': list(range(16, 75)),
    'tetragonal': list(range(75, 143)),
    'trigonal': list(range(143, 168)),
    'hexagonal': list(range(168, 195)),
    'cubic': list(range(195, 230)),
}


class Initialization(object):
    """
    Sample small bulk images as the initial training set, in spirit to the
    ab initio random structure searching algorithm detailed in:
    Chris J Pickard and R J Needs. 2011 J. Phys.: Condens. Matter 23 053201
    (http://iopscience.iop.org/0953-8984/23/5/053201)

    We generate the initial training data by two steps:
        step I: For each space group, create the crystal system and optimize
        the cell size until the stress is zero. Cell optimization is performed
        using `ase.constraints.StrainFilter` module.
        step II: We double the size of the bulk, perturb and sample structures
        by rejection sampling to find more diverse structures.

    *Note: as implemented in the NFT work, EMT and GPAW-based
        DFT calculators are considered. For other calculators, it may need
        a different customization.

    Parameters
    ----------
    elements: str
        elements which constitute the model systems of interest.
    sgs: list of int
        space group numbers based on which bulk structures are constructed.
    a0: float
        initial lattice constant guess.
    minsep: float
        minimum separation for interatomic distances.
    posamp: float
        maximum position amplitude (in angstrom) to move all atoms in the
        cell in a different random direction.
    parent_calc: object
        parent calculator.
    log: str
        logging for the initialization.
    trajfile: str
        trajectory file to save the selected images.

    Returns
    -------
        A list of images saved in a trajectory file.
    """
    def __init__(self, elements, sgs=None, a0=4.0, minsep=1.6, posamp=0.15,
                 parent_calc=EMT(), log=None, trajfile=None):
        if log is None:
            log = Logger(sys.stdout)
        if isinstance(log, str):
            log = Logger(log)
        self._log = log
        self.elements = elements
        if sgs is None:
            # default space groups if none is specified.
            # monoclinic, tetragonal, cubic
            sgs = [10, 79, 225]
        self.sgs = sgs
        self.a0 = a0
        self.minsep = minsep
        self.posamp = posamp
        self.parent_calc = parent_calc
        self._log('Create initial images')
        self._log('='*21)
        no_selected = self.create(trajfile)
        self._log(f'Total number of images selected: {no_selected}')

    def create(self, filename=None):
        """Run the algorithm."""
        elements = self.elements
        sgs = self.sgs
        a0 = self.a0
        log = self._log
        if filename is None:
            filename = 'initial.traj'
        log(f"Images saved in {filename}")
        stable_structs = []
        traj_initial = Trajectory(filename, 'w')
        es_per_atom = []
        # step I
        self._log('Select images based on a cell optimization...')
        total_count = 0
        for sg in sgs:
            cellpar, basis = self.get_cell_params(a0, sg)
            atoms = crystal(elements, basis, spacegroup=sg,
                            cellpar=cellpar)
            # initial magnetic moments for certain elements
            atoms = enforce_magnetic_moments(atoms)
            calc = self._get_calc(atoms)
            if calc.name == 'gpaw':
                self._log(f" Submit cell optimization GPAW jobs for sg: {sg}")
                strain_filter_jobs(atoms, calc)
                # waiting for dft jobs to be finished.
                while not os.path.exists("sf_completed"):
                    time.sleep(30.)
                qn_atoms = read('strain_filter.traj', index=':')
                stable_structs.append(qn_atoms[-1])
            else:
                atoms.set_calculator(calc)
                sf = StrainFilter(atoms)
                temp_traj = 'temp.traj'
                temp_log = 'temp.log'
                dyn = qn(sf, trajectory=temp_traj, logfile=temp_log)
                dyn.run(fmax=0.05)
                stable_structs.append(atoms)
                qn_atoms = read(temp_traj, index=':')
                os.system(f'rm -f {temp_traj} {temp_log}')
            chosen_stepI, e_per_atom = self.energy_filter(qn_atoms)
            es_per_atom.append(e_per_atom)
            min_dist_from_traj = self.get_min_distance(chosen_stepI)
            if min_dist_from_traj < self.minsep:
                # if a unreasonable minsep is supplied in __init__,
                # it can be determined from the cell optimization step.
                self.minsep = min_dist_from_traj * 0.8
            for _ in chosen_stepI:
                traj_initial.write(_)
            self._log(f'Space group {sg}: '
                      f'{len(chosen_stepI)} images chosen out of '
                      f'{len(qn_atoms)} images')
            total_count += len(chosen_stepI)
            os.system("rm -rf sf_completed strain_filter.traj atoms.traj")
        self._log(f'...Number of images selected in step I: {total_count}')
        count_stepII = 0

        # step II
        self._log('Double cell size and random perturb atomic positions...')
        es_per_atom = np.array(es_per_atom).round(2)
        self._log(f' Maximum per atom energy: {str(es_per_atom)}')
        for sg, config in zip(sgs, stable_structs):
            atoms = config.repeat(2)
            chosen = self.shake(atoms, es_per_atom)
            count_stepII += len(chosen)
            total_count += len(chosen)
            self._log(f' Number of images for sg {sg}: {len(chosen)}')
            for _ in chosen:
                traj_initial.write(_)
        self._log(f'...Number of images selected in step II: {count_stepII}')
        return total_count

    def shake(self, atoms, es_per_atom, freq=10, e_per_atom_cutoff=0.4):
        """Randomly displace doubled cells and check if each time the
        displaced configuration satisfies the energy and geometry criteria.

        Parameters
        ------------
        freq : int
            number of images to be collected.
        e_per_atom_cutoff:
            per-atom energy cutoff.
        """
        posamp = self.posamp
        chosen = []
        count = 0
        randomseed = 0
        while count < freq:
            atoms_copy = atoms.copy()
            atoms_copy.rattle(stdev=posamp, seed=int(randomseed))
            randomseed += 1  # ensuring different random displacement each time
            is_minsep, min_dist = self.check_minsep(atoms_copy)
            if is_minsep:
                count += 1
                if self.parent_calc.name == 'gpaw':
                    self._log("Submit GPAW single-point jobs")
                    single_point_jobs(atoms_copy, self.parent_calc)
                    while not os.path.exists("sp_completed"):
                        time.sleep(30.)
                    atoms_copy = read('single_point.traj')
                    os.system('rm -rf single_point.traj atoms.traj' +
                              ' sp_completed')
                else:
                    atoms_copy = self._get_e_and_f(atoms_copy)
                e = atoms_copy.get_potential_energy()
                e_per_atom = e / len(atoms)
                self._log('  Minimum interatomic distance: '
                          f'{min_dist:.2f} \u212B. Average energy per atom: '
                          f'{e_per_atom:.2f} eV')
                es_per_atom = np.array(es_per_atom)
                e_diff = np.abs(es_per_atom - e_per_atom)
                if min(e_diff) <= e_per_atom_cutoff:
                    chosen.append(atoms_copy)
                else:
                    continue

        return chosen

    def check_minsep(self, atoms):
        """Calculate the minimum inter-atomic distance in a structure,
        and check if it satisfied the minimum separation requirement."""
        minsep = self.minsep
        no_of_atoms = len(atoms)
        min_dist = min(atoms.get_distance(i, j, mic=True)
                       for i in range(no_of_atoms)
                       for j in range(i+1, no_of_atoms))
        if min_dist >= minsep:
            return True, min_dist
        else:
            return False, min_dist

    def get_min_distance(self, traj):
        """Find the minimum inter-atomic distance from a list of atoms."""
        min_dists = []
        for atoms in traj:
            no_of_atoms = len(atoms)
            min_dist = min(atoms.get_distance(i, j, mic=True)
                           for i in range(no_of_atoms)
                           for j in range(i+1, no_of_atoms))
            min_dists.append(min_dist)
        return min(min_dists)

    def energy_filter(self, traj, e_cutoff=0.3):
        """Select images which are close to the stable/relaxed structure.
        Also return the maximum per-atom energy in the trajectory."""
        atoms_relaxed = traj[-1]
        e_min = atoms_relaxed.get_potential_energy(apply_constraint=False)
        e_max_per_atom = -np.inf
        chosen = []
        for atoms in traj:
            e = atoms.get_potential_energy(apply_constraint=False)
            e_per_atom = e / len(atoms)
            if abs(e - e_min) < e_cutoff:
                chosen.append(atoms)
                if e_per_atom > e_max_per_atom:
                    e_max_per_atom = e_per_atom
        return chosen, e_max_per_atom

    def _get_calc(self, atoms):
        """Create a new calculator, which is useful for a DFT calculator."""
        calc = self.parent_calc
        if calc.name == 'gpaw':
            from gpaw import GPAW
            cell = atoms.cell.lengths()
            kpts = np.int32(30 // cell) + 1
            params_dict = calc.parameters
            calc = GPAW(**params_dict)
            calc.set(kpts=kpts)
        return calc

    def _get_e_and_f(self, atoms):
        calc = self.parent_calc
        if calc.name == 'gpaw':
            from gpaw import GPAW
            cell = atoms.cell.lengths()
            kpts = np.int32(30 // cell) + 1
            params_dict = calc.parameters
            calc = GPAW(**params_dict)
            calc.set(kpts=kpts)
        atoms.calc = calc
        e = atoms.get_potential_energy(apply_constraint=False)
        f = atoms.get_forces(apply_constraint=False)
        sp = SinglePointCalculator(atoms, energy=e, forces=f)
        atoms.set_calculator(sp)
        return atoms

    def get_cell_params(self, a0, sg):
        """Return cell parameters for different crystal systems."""
        elements = self.elements
        bravais_target = None
        a = a0
        for bravais, sg_list in sg_dict.items():
            if sg in sg_list:
                bravais_target = bravais
        if bravais_target is None:
            raise RuntimeError("Not valid space group number")
        if len(elements) == 1:
            np.random.seed(42)
            if bravais_target == 'monoclinic':
                # create a geometry-restricted monoclinic cell
                # 0.2 is the amplitude for cell length fluctuation w.r.t a0
                # 20 is the amplitude for cell angle fluctuation w.r.t 90
                b = a0*(0.8 + 2*0.2*np.random.rand())
                c = a0*(0.8 + 2*0.2*np.random.rand())
                alpha, beta, gamma = 90 + 20*(2*np.random.rand(3) - 1)
                cellpar = [a, b, c, alpha, beta, gamma]
                basis = [(0, 0, 0)]
            elif bravais_target == 'orthorhombic':
                b = a0*(0.8 + 2*0.2*np.random.rand())
                c = a0*(0.8 + 2*0.2*np.random.rand())
                cellpar = [a, b, c, 90, 90, 90]
                basis = [(0, 0, 0)]
            elif bravais_target == 'tetragonal':
                c = a0*(0.8 + 2*0.2*np.random.rand())
                cellpar = [a, a, c, 90, 90, 90]
                basis = [(0, 0, 0)]
            elif bravais_target == 'cubic':
                cellpar = [a, a, a, 90, 90, 90]
                basis = [(0, 0, 0)]
            elif bravais_target in ['hexagonal', 'trigonal']:
                c = 1.633*a
                cellpar = [a, a, c, 90, 90, 120]
                basis = [(1./3., 2./3., 3./4.)]
        elif len(elements) == 2:
            if bravais_target == 'tetragonal':
                # Rutile template
                c = a0/(1.4 + 2*0.2*np.random.rand())
                cellpar = [a, a, c, 90, 90, 90]
                basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
            elif bravais_target == 'cubic':
                cellpar = [a, a, a, 90, 90, 90]
                basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
            else:
                raise RuntimeError('This space group is not implemented'
                                   ' for two-element systems.')
        else:
            raise RuntimeError('Current implementation is only for'
                               ' one or two-element systems!'
                               ' We recommend using random doping to create'
                               ' systems with more than two elements.')
        return cellpar, basis
