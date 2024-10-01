from typing import List, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01] + \
           [1 if atom.IsInRing() else 0]
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond



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

        atom_i_features = atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = bond_features(bond_ij)
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

        atom_i_features = atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = bond_features(bond_ij)
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

        atom_i_features = atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = bond_features(bond_ij)
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