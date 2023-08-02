import os
import time
import pickle
from string import Template
from ase.io import Trajectory
from ase.parallel import paropen

# Template script for GPAW DFT jobs during nearsighted force training.
script = """#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
from ase.io import read, Trajectory
from ase.parallel import paropen
from gpaw import GPAW, PW, FermiDirac, Mixer

with open("../gpaw_params.pkl", "rb") as f:
    gpaw_params = pickle.load(f)
gpaw_params.pop('txt', None)
def get_calc(text=None, atoms=None):
    if atoms is None:
        kpts = (1, 1, 1)
    else:
        cell = atoms.cell.lengths()
        kpts = np.int32(30 // cell) + 1
    gpaw_params.update(dict(kpts=kpts))
    calc = GPAW(txt=text, **gpaw_params)
    return calc

index = int(os.path.split(os.getcwd())[-1])
traj = '../dft-images/%i.traj' % index
atoms = read(traj)

calc = get_calc(text='%s.txt' % str(index), atoms=atoms)
atoms.calc = calc

completed = True
results = {}
try:
    e = atoms.get_potential_energy()
    f = atoms.get_forces(apply_constraint=False)
    results.update(energy=e)
    results.update(forces=f)
except:
    completed = False

f = paropen('completed', 'w')
f.write(str(completed))
f.close()

with paropen('results', 'wb') as f:
    pickle.dump(results, f)
"""

# Template script for cell optimization based on strain fileter
sf_script = """#!/usr/bin/env python3
import os
from ase.io import read, Trajectory
from ase.parallel import paropen
from ase.optimize import BFGS as qn
from ase.constraints import StrainFilter
from ase.calculators.singlepoint import SinglePointCalculator
from gpaw import GPAW
import pickle
import numpy as np

atoms = read("atoms.traj")

with open('gpaw_params.pkl', 'rb') as pf:
    gpaw_params = pickle.load(pf)
def get_calc(text=None, atoms=None):
    if atoms is None:
        kpts = (1, 1, 1)
    else:
        cell = atoms.cell.lengths()
        kpts = np.int32(30 // cell) + 1
    gpaw_params.update(dict(kpts=kpts))
    calc = GPAW(txt=text, **gpaw_params)
    return calc

calc = get_calc(atoms=atoms, text='sf.txt')

atoms.set_calculator(calc)
sf = StrainFilter(atoms)
temp_traj = 'temp.traj'
temp_logfile = 'temp.log'
dyn = qn(sf, trajectory=temp_traj, logfile=temp_logfile)
dyn.run(fmax=0.05)

relax_traj = read(temp_traj, index=':')
sp_relax_traj = []
traj = Trajectory('strain_filter.traj', mode='w')
for _ in relax_traj:
    e = _.get_potential_energy(apply_constraint=False)
    f = np.zeros(_.positions.shape)
    sp = SinglePointCalculator(_, energy=e, forces=f)
    _.calc = sp
    traj.write(_)

command = "rm -rf temp.traj temp.log"
os.system(command)

completed = True
f = paropen('sf_completed', 'w')
f.write(str(completed))
f.close()
"""

# Template script for single point calculation used in Initialization
sp_script = """#!/usr/bin/env python3
import os
from ase.io import read, Trajectory
from ase.parallel import paropen
from ase.calculators.singlepoint import SinglePointCalculator
from gpaw import GPAW
import pickle
import numpy as np

atoms = read("atoms.traj")
with open('gpaw_params.pkl', 'rb') as pf:
    gpaw_params = pickle.load(pf)
def get_calc(text=None, atoms=None):
    if atoms is None:
        kpts = (1, 1, 1)
    else:
        cell = atoms.cell.lengths()
        kpts = np.int32(30 // cell) + 1
    gpaw_params.update(dict(kpts=kpts))
    calc = GPAW(txt=text, **gpaw_params)
    return calc

calc = get_calc(atoms=atoms, text='sp.txt')
atoms.calc = calc
traj_file = 'single_point.traj'
if os.path.exists(traj_file):
    traj = Trajectory(traj_file, mode='a')
else:
    traj = Trajectory(traj_file, mode='w')
e = atoms.get_potential_energy(apply_constraint=False)
f = atoms.get_forces(apply_constraint=False)
sp = SinglePointCalculator(atoms, energy=e, forces=f)
atoms.set_calculator(sp)
traj.write(atoms)

completed = True
f = paropen('sp_completed', 'w')
f.write(str(completed))
f.close()
"""


def submit_dft_jobs(index, cores=24, memory='40G', dftjobpath='dft-jobs'):
    template = Template(script)
    # params_dict = parent_calc.parameters
    # params_dict.pop('txt', None)
    script_dict = {}
    originalpath = os.getcwd()
    os.chdir(dftjobpath)
    os.chdir('%i' % index)
    with open('run.py', 'w') as f:
        f.write(template.substitute(script_dict))
    reminder = cores % 24
    if reminder == 0:
        nodes = int(cores / 24)
        cores = 24
    else:
        nodes = int(cores / 24 + 1)
        cores = int(cores // nodes)
    gpaw_submit_command = 'gpaw-submit'
    start_command = f'{gpaw_submit_command} -n {nodes} -c {int(cores)} ' + \
                    f'-t 48:00:00 -m  {memory} run.py gpaw'
    if os.path.exists('completed') and os.path.exists('results'):
        pass
    else:
        os.system(start_command)
    os.chdir(originalpath)


def strain_filter_jobs(atoms, calc, cores=24, memory='60G'):
    """Independent GPAW jobs for cell optimization using a strain filter."""
    traj = Trajectory('atoms.traj', mode='w')
    traj.write(atoms)
    with open('gpaw_params.pkl', 'wb') as f:
        pickle.dump(calc.parameters, f)
    template = Template(sf_script)
    with paropen('sf.py', 'w') as f:
        f.write(template.substitute({}))
    gpaw_submit_command = 'gpaw-submit'
    start_command = f'{gpaw_submit_command} -n 1 -c {int(cores)} ' + \
                    f'-t 48:00:00 -m  {memory} sf.py gpaw'
    os.system(start_command)


def single_point_jobs(atoms, calc, cores=24, memory='60G'):
    """Independent GPAW jobs for single point calculations."""
    traj = Trajectory('atoms.traj', mode='w')
    traj.write(atoms)
    with open('gpaw_params.pkl', 'wb') as f:
        pickle.dump(calc.parameters, f)
    template = Template(sp_script)
    with paropen('sp.py', 'w') as f:
        f.write(template.substitute({}))
    gpaw_submit_command = 'gpaw-submit'
    start_command = f'{gpaw_submit_command} -n 1 -c {int(cores)} ' + \
                    f'-t 48:00:00 -m  {memory} sp.py gpaw'
    os.system(start_command)


def relaxation_dft_jobs(atoms, parent_calc=None, cores=24,
                        memory='60G', dftjobpath='dft-jobs'):
    dftnewimagespath = f"{dftjobpath}/dft-images"
    originalpath = os.getcwd()
    if not os.path.exists(dftjobpath):
        os.mkdir(dftjobpath)
    os.chdir(dftjobpath)
    if not os.path.exists('gpaw_params.pkl'):
        with open('gpaw_params.pkl', 'wb') as f:
            pickle.dump(parent_calc.parameters, f)
    pwd = os.getcwd()
    sleep = 1.
    indices = range(len(atoms))
    for index in indices:
        if not os.path.exists('%i' % index):
            os.mkdir('%i' % index)
        os.chdir('%i' % index)
        time.sleep(sleep)
        os.chdir(pwd)
    os.chdir(originalpath)
    if not os.path.exists(dftnewimagespath):
        os.mkdir(dftnewimagespath)
    for index in indices:
        trajfile = f"{dftnewimagespath}/{index}.traj"
        if not os.path.exists(trajfile):
            traj = Trajectory(trajfile, 'w')
            traj.write(atoms[index])
        submit_dft_jobs(index=index, cores=cores, memory=memory,
                        dftjobpath=dftjobpath)
