import numpy as np
import torch
from torch_geometric.data import Data

from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.utils import one_of_k_encoding_unk, one_of_k_encoding


def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Si', 'P', 'S', 'Cl', 'Ge', 'Se', 'Br', 'Sn', 'Te', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms) # 17
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) # 4
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1]) # 2
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) # 7
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) # 3
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D]) # 5    
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]
    
    atom_features += one_of_k_encoding(atom.GetIsAromatic(), [0, 1])
    atom_features += one_of_k_encoding(atom.IsInRing(), [0, 1])

    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)


def get_graph_from_smile(molecule, idx):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype = torch.float)
    edge_index = torch.tensor(bonds, dtype = torch.long).T
    edge_feats = torch.tensor(edge_features, dtype = torch.float)

    return Data(x = atom_feats, edge_index = edge_index, edge_attr = edge_feats, idx = idx)


def get_3d_from_smile_GEOM(molecule):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    conf = molecule.GetConformers()[0]
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    centroid = pos.mean(dim = 0)
    distances = torch.linalg.norm(pos - centroid, dim = 1)
    radius = distances.max()

    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype = torch.float)
    edge_index = torch.tensor(bonds, dtype = torch.long).T
    edge_feats = torch.tensor(edge_features, dtype = torch.float)

    return Data(x = atom_feats, edge_index = edge_index, edge_attr = edge_feats, pos = pos, radius = radius)


def get_3d_from_smile(molecule):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    conf = molecule.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    centroid = pos.mean(dim = 0)
    distances = torch.linalg.norm(pos - centroid, dim = 1)
    radius = distances.max()

    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype = torch.float)
    edge_index = torch.tensor(bonds, dtype = torch.long).T
    edge_feats = torch.tensor(edge_features, dtype = torch.float)

    return Data(x = atom_feats, edge_index = edge_index, edge_attr = edge_feats, pos = pos, radius = radius)


def get_virtual_environment(solute, solvent, idx):

    atomic_number = []

    for i in range(solute.GetNumAtoms()):
        atom_i = solute.GetAtomWithIdx(i)
        atomic_number.append(atom_i.GetAtomicNum())
    solute_z = torch.tensor(atomic_number, dtype=torch.long)

    AllChem.EmbedMolecule(solute, AllChem.ETKDG())
    solute_conf = solute.GetConformer()
    solute_pos = solute_conf.GetPositions()
    solute_pos = torch.tensor(solute_pos, dtype=torch.float)

    solute = Data(z = solute_z, pos = solute_pos, idx = idx)

    atomic_number = []

    for i in range(solvent.GetNumAtoms()):
        atom_i = solvent.GetAtomWithIdx(i)
        atomic_number.append(atom_i.GetAtomicNum())
    solvent_z = torch.tensor(atomic_number, dtype=torch.long)

    AllChem.EmbedMolecule(solvent, AllChem.ETKDG())
    solvent_conf = solvent.GetConformer()
    solvent_pos = solvent_conf.GetPositions()
    solvent_pos = torch.tensor(solvent_pos, dtype=torch.float)

    gaussian_noise = np.random.normal(size = solute_pos.shape)
    transformation = solute_pos + gaussian_noise

    transformed_solvent_pos = transformation.unsqueeze(1) + solvent_pos

    solvent_list = list()
    for i in range(transformed_solvent_pos.shape[0]):
        solvent_list.append(Data(z = solvent_z, pos = transformed_solvent_pos[i], idx = idx))
    
    system_list = [solute] + solvent_list
    system_data = Batch.from_data_list(system_list)

    return Data(z = system_data.z, pos = system_data.pos, solute = system_data.batch == 0, idx = idx)


def create_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = axis
    
    rotation_matrix = np.array([
        [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle) - uz*sin_angle, ux*uz*(1-cos_angle) + uy*sin_angle],
        [uy*ux*(1-cos_angle) + uz*sin_angle, cos_angle + uy*uy*(1-cos_angle), uy*uz*(1-cos_angle) - ux*sin_angle],
        [uz*ux*(1-cos_angle) - uy*sin_angle, uz*uy*(1-cos_angle) + ux*sin_angle, cos_angle + uz*uz*(1-cos_angle)]
    ])
    
    return rotation_matrix


def generate_random_axis_angle():
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, 2 * np.pi)
    return axis, angle


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles