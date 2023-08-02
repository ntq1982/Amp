import os
import sys
import shutil
import tempfile
import numpy as np
from copy import deepcopy
from collections import Counter
from ase import Atoms
from ase.data import atomic_numbers
from ase.calculators.lammps.unitconvert import convert
from .utilities import Logger

def save_to_n2p2(desc_pars, model_pars, images=None, log=None,
                 reference_method='PBE'):
    """Generate input files for n2p2 to use its fast force inference.
    By default,three or four types of files will be generated:

    - scaling.data: scaling information for symmetry functions.
    - input.nn: a settings file, including NN topology and SF setup.
    - weights.xxx.data: NN weights and biases, xxx being the atomic number.
    - input.data (optional): a configuration file includes atomic structures.

    Parameters
    ----------
    desc_pars: dict
        AMP descriptor parameters.
    model_pars: dict
        AMP neural network model parameters.
    images: list
        List of ASE atoms objects.
    log: str or Logger instance
        Logging file.
    reference_method : str
        Method used to generate the reference data.
    """
    if log is None:
        log = Logger(sys.stdout)
    elif isinstance(log, str):
        log = Logger(log)
    # desc_pars = calc.descriptor.parameters
    # model_pars = calc.model.parameters
    reference_method = reference_method.upper()
    if (desc_pars['mode'] != 'atom-centered' or
       model_pars['mode'] != 'atom-centered'):
        raise NotImplementedError(
            'N2P2 requires atom-centered symmetry functions.')
    els = desc_pars['elements']
    n_els = len(els)
    cutoff_name = desc_pars['cutoff']['name'].lower()
    if cutoff_name == 'cosine':
        cutoff_type = 1
    else:
        raise NotImplementedError('Only cosine cutoff can be matched between'
            ' AMP and N2P2')
    cutoff = desc_pars['cutoff']['kwargs']['Rc']
    hl_dict = deepcopy(model_pars['hiddenlayers'])
    hls = list(hl_dict.values())
    for hl in hls:
        if hl == hls[0]:
            continue
        else:
            raise NotImplementedError('N2P2 requires the same NN for all'
             ' elements.')
    hl = list(hls[0])
    hl.append(1)
    activation = model_pars['activation']
    activation = [activation[0]]*(len(hl)) + ['l']

    # Write the input.nn file
    filename = 'input.nn'
    overwrite_file(filename, suffix='.nn', log=log)
    f = open(filename, 'w')

    ## Write header.
    f.write('#'*79 + '\n')
    f.write('# Length unit     : Angstrom\n')
    f.write('# Energy unit     : eV\n')
    f.write(f'# Reference method: {reference_method}\n')
    f.write('#'*79 + '\n')
    f.write('\n')

    ## General NNP settings
    f.write('#'*79 + '\n')
    f.write('# GENERAL NNP SETTINGS\n')
    f.write('#'*79 + '\n')
    f.write('{0: <32}{1:<15d}{2}\n'.format(
        'number_of_elements', n_els, '# Number of elements.'))
    f.write('{0: <32}{1:<15}{2}\n'.format(
        'elements', ' '.join(els), '# Specification of elements.'))
    f.write('{0: <32}{1:<15d}{2}\n'.format(
        'cutoff_type', cutoff_type, '# Cutoff type.'))
    f.write('{0: <32}{1:<15}{2}\n'.format(
        'scale_symmetry_functions', ' ',
        '# Scale all symmetry functions with min/max values.'))
    f.write('{0: <32}{1:<15}{2}\n'.format(
        '# scale_symmetry_functions_sigma', ' ',
        '# Scale all symmetry functions with sigma.'))
    f.write('{0: <32}{1:<15}{2}\n'.format(
        '# atom_energy', 'S 0.0',
        '# Free atom reference energy (S).'))
    f.write('{0: <32}{1:<15.1f}{2}\n'.format(
        'scale_min_short', -1., '# Minimum value for scaling.'))
    f.write('{0: <32}{1:<15.1f}{2}\n'.format(
        'scale_max_short', 1., '# Maximum value for scaling.'))
    f.write('{0: <32}{1:<15d}{2}\n'.format(
        'global_hidden_layers_short', len(hl), '# Number of hiddenlayers'))
    hl_str = [str(_) for _ in hl]
    f.write('{0: <32}{1:<15}{2}\n'.format(
        'global_nodes_short', ' '.join(hl_str),
        '# Number of nodes in each hidden layer'))
    f.write('{0: <32}{1:<15}{2}\n'.format(
        'global_activation_short', ' '.join(activation),
        '# Activation function for each hidden layer and output layer.\n'))

    f.write('#'*79 + '\n')
    f.write('# ADDITIONAL SETTINGS FOR DATASET TOOLS\n')
    f.write('#'*79 + '\n')
    f.write('# These keywords are used only by tools handling data sets:\n')
    f.write('{0: <32}{1:<15}{2}\n'.format(
        'use_short_forces', '', '# Use forces.'))
    f.write('{0: <32}{1:<15d}{2}\n'.format(
        'random_seed', 1234567, '# Random number generator seed.\n'))

    f.write('#'*79 + '\n')
    f.write('# SYMMETRY FUNCTIONS\n')
    f.write('#'*79 + '\n')
    f.write('# Radial symmetry function (type 2):\n')
    f.write('#symfunction_short <element-central> 2'
     ' <element-neighbor> <eta> <rshift> <rcutoff> \n\n')
    f.write('# Narrow Angular symmetry function (type 3):\n')
    f.write('#symfunction_short <element-central> 3 '
        ' <element-neighbor1> <element-neighbor2> '
        '<eta> <lambda> <zeta> <rcutoff> <<rshift> \n\n')
    f.write('# Wide Angular symmetry function (type 9):\n')
    f.write('#symfunction_short <element-central> 9 '
        ' <element-neighbor1> <element-neighbor2> '
        '<eta> <lambda> <zeta> <rcutoff> <<rshift> \n\n')

    ## Write symmetry functions
    leading = 'symfunction_short'
    Gs_indices_dict = {}
    for el in els:
        f.write(f'### Element {el}\n' )
        Gs = desc_pars['Gs'][el]
        # Gs = sorted(Gs, key=lambda x: x['type'])
        Gs_indices, Gs = get_Gs_indices(Gs)
        Gs_indices_dict[el] = Gs_indices
        for G in Gs:
            if G['type'] == 'G2':
                neighb0 = G['element']
                eta = G['eta'] / cutoff ** 2
                offset = G['offset']
                f.write('{0} {1:<3}{2} {3:<3}{4:.15E} {5:.3E} {6:.3E}\n'.\
                    format(leading, el, 2, neighb0, eta, offset, cutoff))
            elif G['type'] == 'G4':
                neighbs = G['elements']
                neighbs = '  '.join(neighbs)
                eta = G['eta'] / cutoff ** 2
                gamma = G['gamma']
                zeta = G['zeta']
                f.write('{0} {1:<3}{2} {3:<8}{4:.15E} {5:d} {6:.3E} '
                    '{7:.3E} {8:.3E}\n'.\
                    format(leading, el, 3, neighbs, eta, gamma, zeta,
                        cutoff, 0.))
            elif G['type'] == 'G5':
                neighbs = G['elements']
                neighbs = '  '.join(neighbs)
                eta = G['eta'] / cutoff ** 2
                gamma = G['gamma']
                zeta = G['zeta']
                f.write('{0} {1:<3}{2} {3:<8}{4:.15E} {5:d} {6:.3E} '
                    '{7:.3E} {8:.3E}\n'.\
                    format(leading, el, 9, neighbs, eta, gamma, zeta,
                        cutoff, 0.))
            else:
                raise NotImplementedError('Symmetry {0}'
                    ' not known.'.format(G['type']))
        f.write('\n')
    f.close()

    # Write the scaling.data file
    scaling_file = 'scaling.data'
    overwrite_file(scaling_file, suffix='.data', log=log)
    f = open(scaling_file, 'w')

    fprange = model_pars.fprange

    ## write headerlines
    f.write('#'*79 + '\n')
    f.write('# Symmetry function scaling data.\n')
    f.write('#'*79 + '\n')
    f.write('# Col  Name     Description\n')
    f.write('#'*79 + '\n')
    f.write('# 1    e_index  Element index.\n')
    f.write('# 2    sf_index Symmetry function index.\n')
    f.write('# 3    sf_min   Symmetry function minimum.\n')
    f.write('# 4    sf_max   Symmetry function maximum.\n')
    f.write('# 5    sf_mean  Symmetry function mean.\n')
    f.write('# 6    sf_sigma Symmetry function sigma.\n')
    f.write('#'*121 + '\n')
    f.write('{0}{1:>9d}{2:>11d}{3:>25d}{4:>25d}{5:>25d}{6:>25d}\n'.format(
        '#', 1, 2, 3, 4, 5, 6))
    f.write('{0}{1:>9}{2:>11}{3:>25}{4:>25}{5:>25}{6:>25}\n'.format(
        '#', 'e_index', 'sf_index', 'sf_min',
        'sf_max', 'sf_mean', 'sf_sigma'))
    f.write('#'*121 + '\n')
    for e_i, el in enumerate(els):
        e_index = e_i + 1
        fprange_el = fprange[el]
        sym_indices = Gs_indices_dict[el]
        fprange_el = list(np.array(fprange_el)[sym_indices])
        for sf_i, each_row in enumerate(fprange_el):
            sf_index = sf_i + 1
            sf_min, sf_max = each_row
            sf_mean, sf_sigma = np.mean(each_row), np.std(each_row)
            f.write('{0}{1:>10d}{2:>11d}{3:>25.16E}{4:>25.16E}'
                '{5:>25.16E}{6:>25.16E}\n'.format(
        '', e_index, sf_index, sf_min, sf_max, sf_mean, sf_sigma))
    f.close()

    # write NN weights and biases
    ann_weights = model_pars.weights
    scalings = model_pars.scalings
    for el in els:
        atomic_number = atomic_numbers[el]
        nn_file = f'weights.{atomic_number:03d}.data'
        overwrite_file(nn_file, suffix='.data', log=log)
        f = open(nn_file, 'w')
        f.write('#'*79 + '\n')
        f.write('# Neural network connection values'
        ' (weights and biases).\n')
        f.write('#'*79 + '\n')
        f.write('# Col  Name       Description\n')
        f.write('#'*79 + '\n')
        f.write('# 1    connection Neural network connection value.\n')
        f.write('# 2    t          Connection type'
        ' (a = weight, b = bias).\n')
        f.write('# 3    index      Index enumerating weights.\n')
        f.write('# 4    l_s        Starting point layer'
         ' (end point layer for biases).\n')
        f.write('# 5    n_s        Starting point neuron in starting'
            ' layer (end point neuron for biases).\n')
        f.write('# 6    l_e        End point layer.\n')
        f.write('# 7    n_e        End point neuron in end layer.\n')
        f.write('#'*79 + '\n')
        f.write('#                      1 2         3     4     5     6'
            '     7\n')
        f.write('#             connection t     index   l_s   n_s   l_e'
            '   n_e\n')
        f.write('#'*79 + '\n')
        ann_weights_el = ann_weights[el]
        scalings_el = scalings[el]
        intercept = scalings_el['intercept']
        slope = scalings_el['slope']
        index = 0
        ll = max(ann_weights_el.keys()) # number of last layer
        syms_indices = Gs_indices_dict[el]
        for key2 in sorted(ann_weights_el.keys()):
            l_s, l_e = key2 - 1, key2
            connections = np.array(ann_weights_el[key2])
            if l_s < 1:
                connections[:-1, :] = connections[syms_indices, :]
            rows, cols = connections.shape
            for ind, val in np.ndenumerate(connections):
                n_s, n_e = ind[0] + 1, ind[1] + 1
                index += 1
                if n_s < rows:
                    t = 'a' # weight
                    f.write(f'{val:>24.16E} {t}{index:>10d}{l_s:>6d}'
                        f'{n_s:>6d}{l_e:>6d}{n_e:>6d}\n')
                else:
                    t = 'b' # bias
                    f.write(f'{val:>24.16E} {t}{index:>10d}{l_e:>6d}'
                        f'{n_e:>6d}\n')
        # coordinate different ll structures in amp and n2p2
        l_s, l_e = ll, ll + 1
        n_s, n_e = 1, 1
        index += 1
        t = 'a'
        f.write(f"{slope:>24.16E} {t}{index:>10d}{l_s:>6d}"
            f"{n_s:>6d}{l_e:>6d}{n_e:>6d}\n")
        index += 1
        t = 'b'
        f.write(f"{intercept:>24.16E} {t}{index:>10d}{l_e:>6d}"
            f"{n_e:>6d}\n")
        f.close()

    if not images:
        return
    save_to_n2p2_images_only(images, log=log)

def save_to_n2p2_images_only(images, log=None):
    """Write the input.data;i.e., the atomic configurations. This is usually
    used together with the `save_to_n2p2` function at above."""
    if log is None:
        log = Logger(sys.stdout)
    elif isinstance(log, str):
        log = Logger(log)
    if not isinstance(images, list):
        if isinstance(images, Atoms):
            images = [images]
        else:
            raise ValueError(f'Type {type(images)} not supported.')
    data_file = 'input.data'
    overwrite_file(data_file, log=log, suffix='.data')
    f = open(data_file, 'w')
    for atoms in images:
        positions = atoms.positions
        symbols = atoms.symbols
        count_elements = Counter(symbols)
        count_str = ' '.join([' '.join([key, str(val)]) for key, val
                      in count_elements.items()])
        pbc = atoms.pbc
        cell = atoms.cell.array
        try:
            e = atoms.get_potential_energy()
        except:
            e = 0.
        try:
            forces = atoms.get_forces(apply_constraint=False)
        except:
            forces = np.zeros(positions.shape)
        try:
            charges = atoms.charges
        except:
            charges = np.zeros(len(atoms))
        total_charge = 0.
        f.write('begin\n')
        if not any(pbc):
            f.write('comment This non-periodic structure contains'
                f' {count_str}\n')
        else:
            f.write('comment This periodic structure contains'
                f' {count_str}\n')
            f.write(f'lattice {cell[0, 0]:.9f} {cell[0, 1]:.9f}'
                f' {cell[0, 2]:.9f}\n')
            f.write(f'lattice {cell[1, 0]:.9f} {cell[1, 1]:.9f}'
                f' {cell[1, 2]:.9f}\n')
            f.write(f'lattice {cell[2, 0]:.9f} {cell[2, 1]:.9f}'
                f' {cell[2, 2]:.9f}\n')
        for i in range(len(atoms)):
            position = positions[i]
            symbol = symbols[i]
            charge = charges[i]
            n_i = 0. # not used
            force = forces[i]
            f.write(f'atom {position[0]:.7f} {position[1]:.7f}'
                f' {position[2]:.7f} {symbol} {charge:.1f}'
                f' {n_i:.1f} {force[0]:.6f} {force[1]:.6f}'
                f' {force[2]:.6f}\n')
        f.write(f'energy {e:.6f}\n')
        f.write(f'charge {total_charge:.1f}\n')
        f.write('end\n')
    f.close()


def overwrite_file(filename, log, suffix='.data'):
    """Overwrite files if existent."""
    if os.path.exists(filename):
        oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                  delete=False,
                                                  suffix=suffix).name

        log('Overwriting file: "%s". Moving original to "%s".'
                  % (filename, oldfilename))
        shutil.move(filename, oldfilename)


def get_Gs_indices(Gs):
    """Handy function to deal with symmetry function orders."""
    from operator import itemgetter
    if not isinstance(Gs, list):
        raise ValueError('Input not known!')
    list_to_sort = []
    for i, G in enumerate(Gs):
        t = int(G['type'][-1])
        if t == 2:
            e1 = atomic_numbers[G['element']]
            e2 = -9999
        elif t == 4 or t == 5:
            e1 = atomic_numbers[G['elements'][0]]
            e2 = atomic_numbers[G['elements'][1]]
        eta = G['eta']
        rs = G['offset'] if 'offset' in G else 0.
        zeta = G['zeta'] if 'zeta' in G else -9999.
        la = G['gamma'] if 'gamma' in G else -9999
        list_to_sort.append((t, eta, rs, zeta, la, e1, e2, i))
    list_to_sort = sorted(list_to_sort, key=itemgetter(0, 1, 2, 3, 4, 5, 6),
        reverse=False)
    indices = [_[-1] for _ in list_to_sort]
    Gs_sorted = list(np.array(Gs)[indices])
    return indices, Gs_sorted

def save_to_prophet(calc, filename='potential_', overwrite=False,
                    units="metal"):
    """Saves the calculator in a way that it can be used with PROPhet.

    Parameters
    ----------
    calc : obj
        A trained Amp calculator object.
    filename : str
        File object or path to the file to write to.
    overwrite : bool
        If an output file with the same name exists, overwrite it.
    units : str
        LAMMPS units style to be used with the outfile file.
    """

    if os.path.exists(filename):
        if overwrite is False:
            oldfilename = filename
            filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                   suffix='.amp').name
            calc._log('File "%s" exists. Instead saving to "%s".' %
                      (oldfilename, filename))
        else:
            oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                      delete=False,
                                                      suffix='.amp').name

            calc._log('Overwriting file: "%s". Moving original to "%s".'
                      % (filename, oldfilename))
            shutil.move(filename, oldfilename)

    desc_pars = calc.descriptor.parameters
    model_pars = calc.model.parameters
    if (desc_pars['mode'] != 'atom-centered' or
       model_pars['mode'] != 'atom-centered'):
        raise NotImplementedError(
            'PROPhet requires atom-centered symmetry functions.')
    if desc_pars['cutoff']['name'] != 'Cosine':
        raise NotImplementedError(
            'PROPhet requires cosine cutoff functions.')
    if model_pars['activation'] != 'tanh':
        raise NotImplementedError(
            'PROPhet requires tanh activation functions.')
    els = desc_pars['elements']
    n_els = len(els)
    length_G2 = int(n_els)
    length_G4 = int(n_els*(n_els+1)/2)
    cutoff = convert(desc_pars['cutoff']['kwargs']['Rc'], 'distance', 'ASE', units)

    # Get correct order of elements listed in the Amp object
    el = desc_pars['elements'][0]
    n_G2 = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G2')
    n_G4 = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G4')
    els_ordered = []
    if n_G2 > 0:
        for Gs in range(n_els):
            els_ordered.append(desc_pars['Gs'][el][Gs]['element'])
    elif n_G4 > 0:
        for Gs in range(n_els):
            els_ordered.append(desc_pars['Gs'][el][Gs]['elements'][0])
    else:
        raise RuntimeError('There must be at least one G2 or G4 symmetry '
                           'function.')
    # Write each element's PROPhet input file
    for el in desc_pars['elements']:
        f = open(filename + el, 'w')
        # Write header.
        f.write('nn\n')
        f.write('structure\n')
        # Write elements.
        f.write(el + ':  ')
        for el_i in els_ordered:
            f.write(el_i+' ')
        f.write('\n')
        n_G2_el = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G2')
        n_G4_el = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G4')
        if n_G2_el != n_G2 or n_G4_el != n_G4:
            raise NotImplementedError(
                'PROPhet requires each element to have the same number of '
                'symmetry functions.')
        f.write(str(int(n_G2/length_G2+n_G4/length_G4))+'\n')
        # Write G2s.
        for Gs in range(0, n_G2, length_G2):
            eta = desc_pars['Gs'][el][Gs]['eta']
            for i in range(length_G2):
                eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                if eta != eta_2:
                    raise NotImplementedError(
                        'PROPhet requires each G2 function to have the '
                        'same eta value for all element pairs.')
            f.write('G2 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                    ' {}\n'.format(0.0)) #August 10/2-2020: Center of radial Gaussian. This should be changed if one wants to allow for non-centered Gaussians.
        # Write G4s (G3s in PROPhet).
        for Gs in range(n_G2, n_G2+n_G4, length_G4):
            eta = desc_pars['Gs'][el][Gs]['eta']
            gamma = desc_pars['Gs'][el][Gs]['gamma']
            zeta = desc_pars['Gs'][el][Gs]['zeta']
            for i in range(length_G4):
                eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                gamma_2 = desc_pars['Gs'][el][Gs+i]['gamma']
                zeta_2 = desc_pars['Gs'][el][Gs+i]['zeta']
                if eta != eta_2 or gamma != gamma_2 or zeta != zeta_2:
                    raise NotImplementedError(
                        'PROPhet requires each G4 function to have the '
                        'same eta, gamma, and zeta values for all '
                        'element pairs.')
            f.write('G3 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                    ' ' + str(zeta) + ' ' + str(gamma) + '\n')
        # Write input means for G2.
        for i in range(n_els):
            for Gs in range(0, n_G2, length_G2):
                # For debugging, to see the order of the PROPhet file
                # if el==desc_pars['elements'][0]:
                #    print(desc_pars['Gs'][el][Gs+i])
                mean = (model_pars['fprange'][el][Gs+i][1] +
                        model_pars['fprange'][el][Gs+i][0]) / 2.
                #mean = model_pars['fprange'][el][Gs+i][1]
                f.write(str(mean) + ' ')
        # Write input means for G4.
        for i in range(n_els):
            for j in range(n_els-i):
                for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                    # For debugging, to see the order of the PROPhet file
                    # if el==desc_pars['elements'][0]:
                    #    print(desc_pars['Gs'][el][Gs+j+n_els*i+int((i-i**2)/2)])
                    mean = (model_pars['fprange'][el][Gs + j + n_els * i +
                                                      int((i - i**2) / 2)][1] +
                            model_pars['fprange'][el][Gs + j + n_els * i +
                                                      int((i - i**2) / 2)][0])/2 #August added divide by 2
                    # NB the G4 mean is doubled to correct for PROPhet
                    # counting each neighbor pair twice as much as Amp
                    f.write(str(mean) + ' ')
        f.write('\n')
        # Write input variances for G2.
        for i in range(n_els):
            for Gs in range(0, n_G2, length_G2):
                variance = (model_pars['fprange'][el][Gs+i][1] -
                            model_pars['fprange'][el][Gs+i][0]) / 2.
                #variance = model_pars['fprange'][el][Gs+i][0]
                f.write(str(variance) + ' ')
        # Write input variances for G4.
        for i in range(n_els):
            for j in range(n_els-i):
                for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                    variance = (model_pars['fprange'][el][Gs + j + n_els * i +
                                                          int((i - i**2) /
                                                              2)][1] -
                                model_pars['fprange'][el][Gs + j + n_els * i +
                                                          int((i - i**2) /
                                                              2)][0])/2 #August added divide by 2
                    # NB the G4 variance is doubled to correct for PROPhet
                    # counting each neighbor pair twice as much as Amp
                    f.write(str(variance) + ' ')
        f.write('\n')
        f.write('energy\n')
        # Write output mean.
        f.write('0\n')
        # Write output variance.
        f.write('1\n')
        curr_node = 0
        # Write NN layer architecture.
        for nodes in model_pars['hiddenlayers'][el]:
            f.write(str(nodes)+' ')
        f.write('1\n')
        # Write first hidden layer of the NN for the symmetry functions.
        layer = 0
        f.write('[[ layer ' + str(layer) + ' ]]\n')
        for node in range(model_pars['hiddenlayers'][el][layer]):
            # Write each node of the layer.
            f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
            f.write('   ')
            # G2
            for i in range(n_els):
                for Gs in range(0, n_G2, length_G2):
                    f.write(str(model_pars['weights'][el]
                                [layer + 1][Gs + i][node]))
                    f.write('     ')
            # G4
            for i in range(n_els):
                for j in range(n_els-i):
                    for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                        f.write(str(model_pars['weights'][el]
                                    [layer + 1][Gs + j + n_els * i +
                                                int((i - i**2) / 2)][node]))
                        f.write('     ')
            f.write('\n')
            f.write('   ')
            f.write(str(model_pars['weights'][el][layer+1][-1][node]))
            f.write('\n')
            curr_node += 1
        # Write remaining hidden layers of the NN.
        for layer in range(1, len(model_pars['hiddenlayers'][el])):
            f.write('[[ layer ' + str(layer) + ' ]]\n')
            for node in range(model_pars['hiddenlayers'][el][layer]):
                f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
                f.write('   ')
                for i in range(len(model_pars['weights'][el][layer+1])-1):
                    f.write(str(model_pars['weights'][el][layer+1][i][node]))
                    f.write('     ')
                f.write('\n')
                f.write('   ')
                f.write(str(model_pars['weights'][el][layer+1][-1][node]))
                f.write('\n')
                curr_node += 1
        # Write output layer of the NN, consisting of an activated node.
        f.write('[[ layer ' + str(layer+1) + ' ]]\n')
        f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
        f.write('   ')
        for i in range(len(model_pars['weights'][el][layer+2])-1):
            f.write(str(model_pars['weights'][el][layer+2][i][0]))
            f.write('     ')
        f.write('\n')
        f.write('   ')
        f.write(str(model_pars['weights'][el][layer+2][-1][0]))
        f.write('\n')
        curr_node += 1
        # Write output layer of the NN, consisting of a linear node,
        # representing Amp's scaling.
        f.write('[[ layer ' + str(layer+2) + ' ]]\n')
        f.write('  [ node ' + str(curr_node) + ' ]  linear\n')
        f.write('   ')
        f.write(str(convert(model_pars['scalings'][el]['slope'], 'energy', 'ASE', units)))
        f.write('\n')
        f.write('   ')
        f.write(str(convert(model_pars['scalings'][el]['intercept'], 'energy', 'ASE', units)))
        f.write('\n')
        f.close()


def save_to_openkim(calc, filename='amp.params', overwrite=False,
                    units="metal"):
    """Saves the calculator in a way that it can be used with OpenKIM.

    Parameters
    ----------
    calc : obj
        A trained Amp calculator object.
    filename : str
        File object or path to the file to write to.
    overwrite : bool
        If an output file with the same name exists, overwrite it.
    units : str
        LAMMPS units style to be used with the outfile file.
    """

    if os.path.exists(filename):
        if overwrite is False:
            oldfilename = filename
            filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                   suffix='.params')
            calc._log('File "%s" exists. Instead saving to "%s".' %
                      (oldfilename, filename))
        else:
            oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                      delete=False,
                                                      suffix='.params')

            calc._log('Overwriting file: "%s". Moving original to "%s".'
                      % (filename, oldfilename))
            shutil.move(filename, oldfilename)

    desc_pars = calc.descriptor.parameters
    model_pars = calc.model.parameters
    if (desc_pars['mode'] != 'atom-centered' or
       model_pars['mode'] != 'atom-centered'):
        raise NotImplementedError(
            'KIM model requires atom-centered symmetry functions.')
    if desc_pars['cutoff']['name'] != 'Cosine':
        raise NotImplementedError(
            'KIM model requires cosine cutoff functions.')
    elements = desc_pars['elements']
#    path = os.path.dirname(__file__)
    elements = sorted(elements)
#    f = open(path + '/../tools/amp-kim/amp_parameterized_model/' +
#             filename, 'w')
    f = open(filename, 'w')
    f.write(str(len(elements)) + '  # number of chemical species')
    f.write('\n')
    f.write(' '.join(elements) + '  # chemical species')
    f.write('\n')
    f.write(' '.join(str(len(desc_pars['Gs'][element])) for element in
            elements) +
            '  # number of fingerprints of each chemical species')
    f.write('\n')
    for element in elements:
        count = 0
        # writing symmetry functions
        for G in desc_pars['Gs'][element]:
            if G['type'] == 'G2':
                f.write(element + ' ' + 'g2' + '  # fingerprint of %s' %
                        element)
                f.write('\n')
                f.write(G['element'] + ' ' + str(G['eta']) + '  # eta')
            elif G['type'] == 'G4':
                f.write(element + ' ' + 'g4' +
                        '  # fingerprint of %s' % element)
                f.write('\n')
                f.write(G['elements'][0] + ' ' + G['elements'][1] + ' ' +
                        str(G['eta']) + ' ' + str(G['gamma']) + ' ' +
                        str(G['zeta']) + '  # eta, gamma, zeta')
            f.write('\n')
            # writing fingerprint range
            f.write(str(model_pars['fprange'][element][count][0]) + ' ' +
                    str(model_pars['fprange'][element][count][1]) +
                    '  # range of fingerprint %i of %s' % (count, element))
            f.write('\n')
            count += 1

    # writing the cutoff
    cutoff = convert(desc_pars['cutoff']['kwargs']['Rc'], 'distance', 'ASE', units)

    f.write(str(cutoff) + '  # cutoff radius')
    f.write('\n')
    f.write(model_pars['activation'] + '  # activation function')
    f.write('\n')
    # writing the neural network structures
    for element in elements:
        f.write(str(len(model_pars['hiddenlayers'][element])) +
                '  # number of hidden-layers of %s neural network' % element)
        f.write('\n')
        f.write(' '.join(str(_) for _ in model_pars['hiddenlayers'][element]) +
                '  # number of nodes of hidden-layers of %s neural network' %
                element)
        f.write('\n')

    # writing parameters of the neural network
    f.write(' '.join(str(_) for _ in \
                     # calc.model.ravel.to_vector(model_pars.weights,
                     # model_pars.scalings)
                     calc.model.vector) +
            '  # weights, biases, and scalings of neural networks')
    f.write('\n')
    f.close()
