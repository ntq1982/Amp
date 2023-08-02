import numpy as np


def atom_electronegativity(elements, supplied_electronegativities=None):
    electronegativities_dict = {
     'H': 2.20,
     'He': np.nan,
     'Li': 0.98,
     'Be': 1.57,
     'B': 2.04,
     'C': 2.55,
     'N': 3.04,
     'O': 3.44,
     'F': 3.98,
     'Ne': np.nan,
     'Na': 0.93,
     'Mg': 1.31,
     'Al': 1.61,
     'Si': 1.90,
     'P': 2.19,
     'S': 2.58,
     'Cl': 3.16,
     'Ar': np.nan,
     'K': 0.82,
     'Ca': 1.0,
     'Sc': 1.36,
     'Ti': 1.54,
     'V': 1.63,
     'Cr': 1.66,
     'Mn': 1.55,
     'Fe': 1.83,
     'Co': 1.88,
     'Ni': 1.91,
     'Cu': 1.90,
     'Zn': 1.65,
     'Ga': 1.81,
     'Ge': 2.01,
     'As': 2.18,
     'Se': 2.55,
     'Br': 2.96,
     'Kr': 3.00,
     'Rb': 0.82,
     'Sr': 0.95,
     'Y': 1.22,
     'Zr': 1.33,
     'Nb': 1.6,
     'Mo': 2.16,
     'Tc': 1.9,
     'Ru': 2.2,
     'Rh': 2.28,
     'Pd': 2.2,
     'Ag': 1.93,
     'Cd': 1.69,
     'In': 1.78,
     'Sn': 1.96,
     'Sb': 2.05,
     'Te': 2.1,
     'I': 2.66,
     'Xe': 2.6,
     'Cs': 0.79,
     'Ba': 0.89,
     'La': 1.1,
     'Ce': 1.12,
     'Pr': 1.13,
     'Nd': 1.14,
     'Pm': 1.13,
     'Sm': 1.17,
     'Eu': 1.2,
     'Gd': 1.2,
     'Tb': 1.1,
     'Dy': 1.22,
     'Ho': 1.23,
     'Er': 1.24,
     'Tm': 1.25,
     'Yb': 1.1,
     'Lu': 1.27,
     'Hf': 1.3,
     'Ta': 1.5,
     'W': 2.36,
     'Re': 1.9,
     'Os': 2.2,
     'Ir': 2.2,
     'Pt': 2.28,
     'Au': 2.54,
     'Hg': 2.,
     'Tl': 1.62,
     'Pb': 2.33,
     'Bi': 2.02,
     'Po': 2.,
     'At': 2.2,
     'Rn': 2.2,
     'Fr': 0.79,
     'Ra': 0.9,
     'Ac': 1.1,
     'Th': 1.3,
     'Pa': 1.5,
     'U': 1.38,
     'Np': 1.36,
     'Pu': 1.28,
     'Am': 1.13,
     'Cm': 1.28,
     'Bk': 1.3,
     'Cf': 1.3,
     'Es': 1.3,
     'Fm': 1.3,
     'Md': 1.3,
     'No': 1.3,
     'Lr': 1.3, }
    electronegativities_guess = {}
    if supplied_electronegativities is not None:
        electronegativities_dict.update(supplied_electronegativities)
    for element in elements:
        if element in electronegativities_dict.keys():
            electronegativities_guess[element] = \
                electronegativities_dict[element]
    return electronegativities_guess


def atom_charges(elements, supplied_charges=None):
    charges = {
     'H': 1.,
     'He': 0.,
     'Li': 1.,
     'Be': 2.,
     'B': 3.,
     'C': 4.,
     'N': -3.,
     'O': -2.,
     'F': -1.,
     'Ne': 0.,
     'Na': 1.,
     'Mg': 2.,
     'Al': 3.,
     'Si': 4.,
     'P': -3.,
     'S': -2.,
     'Cl': -1.,
     'Ar': 0.,
     'K': 1.,
     'Ca': 2.0,
     'Sc': 0.,
     'Ti': 0.,
     'V': 0.,
     'Cr': 0.,
     'Mn': 0.,
     'Fe': 0.,
     'Co': 0.,
     'Ni': 0.,
     'Cu': 0.,
     'Zn': 0.,
     'Ga': 3.,
     'Ge': 4.,
     'As': -3.,
     'Se': -2.,
     'Br': -1.,
     'Kr': 0.,
     'Rb': 1.,
     'Sr': 2.,
     'Y': 0.,
     'Zr': 0.,
     'Nb': 0.,
     'Mo': 0.,
     'Tc': 0.,
     'Ru': 0.,
     'Rh': 0.,
     'Pd': 0.,
     'Ag': 0.,
     'Cd': 0.,
     'In': 3.,
     'Sn': 4.,
     'Sb': -3.,
     'Te': -2.,
     'I': -1.,
     'Xe': 0.,
     'Cs': 1.,
     'Ba': 2.,
     'La': 0.,
     'Ce': 0.,
     'Pr': 0.,
     'Nd': 0.,
     'Pm': 0.,
     'Sm': 0.,
     'Eu': 0.,
     'Gd': 0.,
     'Tb': 0.,
     'Dy': 0.,
     'Ho': 0.,
     'Er': 0.,
     'Tm': 0.,
     'Yb': 0.,
     'Lu': 0.,
     'Hf': 0.,
     'Ta': 0.,
     'W': 0.,
     'Re': 0.,
     'Os': 0.,
     'Ir': 0.,
     'Pt': 0.,
     'Au': 0.,
     'Hg': 0.,
     'Tl': 3.,
     'Pb': 4.,
     'Bi': -3.,
     'Po': -2.,
     'At': -1.,
     'Rn': 0.,
     'Fr': 1.,
     'Ra': 2.,
     'Ac': 0.,
     'Th': 0.,
     'Pa': 0.,
     'U': 0.,
     'Np': 0.,
     'Pu': 0.,
     'Am': 0.,
     'Cm': 0.,
     'Bk': 0.,
     'Cf': 0.,
     'Es': 0.,
     'Fm': 0.,
     'Md': 0.,
     'No': 0.,
     'Lr': 0., }
    if supplied_charges is not None:
        charges.update(supplied_charges)
    atom_charges = {}
    for element in elements:
        if element in charges.keys():
            atom_charges[element] = charges[element]
    return atom_charges
