import sys
sys.path.append('.')

import glob
import os
import itertools
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm

from collections.abc import Sequence

import torch
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

from utils.chem import get_3d_from_smile, create_rotation_matrix, generate_random_axis_angle
from datasets_pretrain.VRdata import VRData


EPSILON = 1e-6

def get_molecular_radius(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    centroid = pos.mean(dim = 0)
    distances = torch.linalg.norm(pos - centroid, dim = 1)
    radius = distances.max()
    
    return radius


class DDI(InMemoryDataset):
    def __init__(self, root, rotation = False, radius = False, fixed_direction = False, sample = None, no_solvent = False, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        
        self.rotation = rotation
        self.radius = radius
        self.fixed_direction = fixed_direction
        self.sample = sample
        self.no_solvent = no_solvent
        if self.no_solvent == False:
            assert self.radius != self.fixed_direction
        
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.raw_folder_path = os.path.join(self.root, "DDI", "raw")

        super(DDI, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_dataset()
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, "DDI", "processed")

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):

        csv_files = glob.glob(os.path.join(self.raw_folder_path, '**', '*.csv'), recursive=True)

        smiles1_list = list()
        smiles2_list = list()
        label_list = list()

        for file in csv_files:
            raw_df = pd.read_csv(file)
            smiles1_list.append(raw_df["smiles_1"])
            smiles2_list.append(raw_df["smiles_2"])
            label_list.append(raw_df["label"])
            print(file)
        
        smiles1 = list(pd.concat(smiles1_list, ignore_index=True))
        smiles2 = list(pd.concat(smiles2_list, ignore_index=True))
        labels = list(pd.concat(label_list, ignore_index=True))

        df = pd.DataFrame({'smiles1': smiles1, 'smiles2': smiles2, 'label': labels})
        df = df[df["label"] == 1]
        df = df.drop_duplicates().reset_index(drop=True)

        unique_smiles = np.unique(list(df["smiles1"]) + list(df["smiles2"]))
        smiles2graph = {}
        
        for smiles in tqdm(unique_smiles):
            try:
                molecule = Chem.MolFromSmiles(smiles)
                molecule = Chem.AddHs(molecule)
                smiles2graph[smiles] = get_3d_from_smile(molecule)
            except:
                print("Error for SMILES: {}".format(smiles))
                smiles2graph[smiles] = "N/A"

        solute_list, solvent_list = list(), list()

        for i in range(len(df)):

            try:
                radius1 = smiles2graph[df.iloc[i]["smiles1"]].radius
                radius2 = smiles2graph[df.iloc[i]["smiles2"]].radius
                if radius1 > radius2:
                    solute_list.append(df.iloc[i]["smiles1"])
                    solvent_list.append(df.iloc[i]["smiles2"])
                elif radius1 < radius2:
                    solute_list.append(df.iloc[i]["smiles2"])
                    solvent_list.append(df.iloc[i]["smiles1"])
            except:
                print("Error Index {}".format(i))
                pass

        df = pd.DataFrame({'Solute': solute_list, 'Solvent': solvent_list})

        torch.save((smiles2graph, df), self.processed_paths[0])

    def load_dataset(self):

        self.smiles2graph, self.dataframe = torch.load(self.processed_paths[0])
        
        return

    def get_only_solute(self, solute):

        return Data(pos = solute.pos, z = torch.where(solute.x[:, :17])[1])

    def get_virtual_environment(self, solute, solvent):

        solute_temp = Data(pos = solute.pos, pos_target = torch.zeros_like(solute.pos, dtype = float), 
                           z = torch.where(solute.x[:, :100])[1], assigned_solute = torch.zeros(solute.pos.shape[0]), assigned_solvent = torch.zeros(solute.pos.shape[0]))
        solvent_temp = Data(pos = solvent.pos, z = torch.where(solvent.x[:, :100])[1])

        # Solvent only for atoms not in the ring
        not_ring = solute.x[:, -1] == 0
        not_ring_idx = torch.where(not_ring)[0]
        if self.sample is not None:
            if len(not_ring_idx) > self.sample: # Sample only when the number of not ring atom is larger than sample
                perm = torch.randperm(not_ring_idx.size(0))
                not_ring_idx = perm[:self.sample]
            else:
                pass

        gaussian_noise = torch.randn(size = solute_temp.pos[not_ring_idx].shape)
        # gaussian_noise = torch.zeros_like(solute_temp.pos[not_ring_idx])
        if self.radius:
            # Scale the noise as radius of solvent
            noise_norm = torch.norm(gaussian_noise, dim = 1)
            normalized_noise = gaussian_noise / noise_norm.reshape(-1, 1)
            gaussian_noise = normalized_noise * solvent.radius
            
        translation = solute_temp.pos[not_ring_idx] + gaussian_noise
        
        if self.fixed_direction:
            position_norm = torch.norm(solute_temp.pos[not_ring_idx], dim = 1) + EPSILON
            direction = solute_temp.pos[not_ring_idx] / position_norm.reshape(-1, 1)
            direction = direction * solvent.radius
            translation += direction
        
        solvent_list = list()
        for i in range(len(not_ring_idx)):
            
            if self.rotation:
                axis, angle = generate_random_axis_angle()
                rotation_matrix = create_rotation_matrix(axis, angle)
                rotated_solvent = np.dot(solvent.pos, rotation_matrix)
                transformed_solvent_pos = translation[i].unsqueeze(0) + rotated_solvent
            else:
                transformed_solvent_pos = translation[i].unsqueeze(0) + solvent.pos

            assert transformed_solvent_pos.isnan().sum() == 0
            
            pos_target = transformed_solvent_pos - solute_temp.pos[not_ring_idx][i]
            pos_target = pos_target / (pos_target.norm(dim = 1).reshape(-1, 1) + EPSILON)
            
            assert pos_target.isnan().sum() == 0
            
            solute_atom_idx = not_ring_idx[i]
            
            solvent_list.append(Data(pos = transformed_solvent_pos, pos_target = pos_target, 
                                     z = solvent_temp.z, assigned_solute = solute_atom_idx.expand(solvent.pos.shape[0]),
                                     assigned_solvent = torch.tensor(range(solvent.pos.shape[0])),
                                     edge_index = solvent.edge_index))
        
        # For Solvent 2D Edge Index
        system_data_ = Batch.from_data_list(solvent_list)

        system_list = [solute_temp] + solvent_list
        system_data = Batch.from_data_list(system_list)

        return VRData(z = system_data.z, pos = system_data.pos.float(), pos_target = system_data.pos_target.float(), solute = system_data.batch == 0, 
                      solute_2d_idx = system_data.assigned_solute.long(), solvent_2d_idx = system_data.assigned_solvent.long(),
                      solute_size = solute.pos.shape[0], solvent_size = solvent.pos.shape[0], mol_idx = system_data.batch, num_mol = system_data.batch.max() + 1,
                      num_solvent_atoms = system_data_.pos.shape[0], edge_index = system_data_.edge_index)

    def get(self, idx):

        solute = self.dataframe.iloc[idx]["Solute"]
        solvent = self.dataframe.iloc[idx]["Solvent"]

        solute_data = self.smiles2graph[solute]
        solvent_data = self.smiles2graph[solvent]
        
        if self.no_solvent:
            vr_data = self.get_only_solute(solute_data)

        else:
            vr_data = self.get_virtual_environment(solute_data, solvent_data)
            assert (vr_data.solute_2d_idx[vr_data.solute == 1] == 0).sum() == solute_data.x.shape[0]
        
        return solute_data, solvent_data, vr_data
    
    def __len__(self):
        return len(self.dataframe)
    
    def indices(self) -> Sequence:
        return range(len(self.dataframe))


class DDISubset(DDI):
    def __init__(self, root, size, rotation = False, radius = False, fixed_direction = False, sample = None, no_solvent = False, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.rotation = rotation
        self.radius = radius
        self.fixed_direction = fixed_direction
        self.sample = sample
        self.no_solvent = no_solvent
        if self.no_solvent == False:
            assert self.radius != self.fixed_direction
        
        self.size = size
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        
        super(DDI, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_dataset()
        self.dataframe = self.dataframe.sample(frac=1)

        return

    def __len__(self):
        return self.size


if __name__ == "__main__":
    
    DATA_PATH = "./data_pretrain"
    batch_size = 32
    num_workers = 8
    
    # Chr : (Chromophore, Solvent, Absorption max (nm)/ Emission max (nm)/ Lifetime (ns))
    dataset = DDI(DATA_PATH, rotation = True, radius = False, fixed_direction = True, sample = 2, no_solvent = True)
    subdataset = DDISubset(DATA_PATH, size = 20, rotation = True, fixed_direction = True, sample = 5, no_solvent = True)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(subdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")