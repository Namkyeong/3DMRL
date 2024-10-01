import sys
sys.path.append('.')

import copy
import os
import itertools
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm

from collections.abc import Sequence
from typing import Optional

import torch
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import get_3d_from_smile, create_rotation_matrix, generate_random_axis_angle
from datasets_pretrain.VRdata import VRData

EPSILON = 1e-6

class SolvationAll(InMemoryDataset):
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
        self.solv_file_path = os.path.join(self.root, "raw", "combisolv_exp.csv")

        super(SolvationAll, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_dataset()
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_solvation')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        
        combisolv_df = pd.read_csv(self.solv_file_path, sep = ",")
        combisolv_df = combisolv_df[["mol solute", "mol solvent"]]
        combisolv_df = combisolv_df.rename(columns={"mol solute": "Solute", "mol solvent": "Solvent"})
        combisolv_df = combisolv_df[combisolv_df["Solvent"] != "gas"]

        # Canonicalize SMILES
        print("Start Canonicalization of SMILES")
        canon_solute = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in combisolv_df["Solute"]]
        canon_solvent = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in combisolv_df["Solvent"]]
        
        # Get unique SMILES
        canon_solute = np.unique(canon_solute)
        print("Num. Unique Solute: {}".format(len(canon_solute)))
        canon_solvent = np.unique(canon_solvent)
        print("Num. Unique Solvent: {}".format(len(canon_solvent)))

        solute_list, solvent_list= list(), list()
        
        for idx, solute in enumerate(tqdm(canon_solute)):
            try:
                solute = Chem.MolFromSmiles(solute)
                solute = Chem.AddHs(solute)
                solute_graph = get_3d_from_smile(solute)
                solute_graph.idx = idx
                solute_list.append(solute_graph)
            except:
                pass
        
        for idx, solvent in enumerate(tqdm(canon_solvent)):
            try:
                solvent = Chem.MolFromSmiles(solvent)
                solvent = Chem.AddHs(solvent)
                solvent_graph = get_3d_from_smile(solvent)
                solvent_graph.idx = idx
                solvent_list.append(solvent_graph)
            except:
                pass
        
        print("Converted Solute Data: {}/{}".format(len(solute_list), len(canon_solute)))
        print("Converted Solvent Data: {}/{}".format(len(solvent_list), len(canon_solvent)))
        
        solute_graphs, solute_slices = self.collate(solute_list)
        solvent_graphs, solvent_slices = self.collate(solvent_list)        

        torch.save((solute_graphs, solute_slices, solvent_graphs, solvent_slices, canon_solute, canon_solvent), self.processed_paths[0])

    def load_dataset(self):

        self.solute_graphs, self.solute_slices, self.solvent_graphs, self.solvent_slices, self.solute_list, self.solvent_list = torch.load(self.processed_paths[0])
        self.num_solutes = len(self.solute_slices["x"]) - 1
        self.num_solvents = len(self.solvent_slices["x"]) - 1
        self.data_list = np.asarray(list(itertools.product([i for i in range(self.num_solutes)], [i for i in range(self.num_solvents)])))
        
        return

    def get_only_solute(self, solute):

        return Data(pos = solute.pos, z = torch.where(solute.x[:, :17])[1])

    def get_virtual_environment(self, solute, solvent):

        solute_temp = Data(pos = solute.pos, pos_target = torch.zeros_like(solute.pos, dtype = float), 
                           z = torch.where(solute.x[:, :17])[1], assigned_solute = torch.zeros(solute.pos.shape[0]), assigned_solvent = torch.zeros(solute.pos.shape[0]))
        solvent_temp = Data(pos = solvent.pos, z = torch.where(solvent.x[:, :17])[1])

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
            pos_target = pos_target / pos_target.norm(dim = 1).reshape(-1, 1)
            
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

        solute_idx, solvent_idx = self.data_list[idx]

        solute_data = Data()
        for key in self.solute_graphs.keys:
            item, slices = self.solute_graphs[key], self.solute_slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[solute_data.__cat_dim__(key, item)] = slice(slices[solute_idx], slices[solute_idx + 1])
            solute_data[key] = item[s]
        
        solvent_data = Data()
        for key in self.solvent_graphs.keys:
            item, slices = self.solvent_graphs[key], self.solvent_slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[solvent_data.__cat_dim__(key, item)] = slice(slices[solvent_idx], slices[solvent_idx + 1])
            solvent_data[key] = item[s]
        
        if self.no_solvent:
            vr_data = self.get_only_solute(solute_data)

        else:
            vr_data = self.get_virtual_environment(solute_data, solvent_data)
            assert (vr_data.solute_2d_idx[vr_data.solute == 1] == 0).sum() == solute_data.x.shape[0]
        
        return solute_data, solvent_data, vr_data
    
    def __len__(self):
        return len(self.data_list)
    
    def indices(self) -> Sequence:
        return range(len(self.data_list))


class SolvationSubset(SolvationAll):
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
        
        super(SolvationAll, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_dataset()
        np.random.shuffle(self.data_list)

        return

    def __len__(self):
        return self.size


if __name__ == "__main__":
    
    DATA_PATH = "./data_pretrain"
    batch_size = 4
    num_workers = 6
    
    # Chr : (Chromophore, Solvent, Absorption max (nm)/ Emission max (nm)/ Lifetime (ns))
    dataset = SolvationAll(DATA_PATH, rotation = True, radius = False, fixed_direction = True, sample = 2, no_solvent = True)
    subdataset = SolvationSubset(DATA_PATH, size = 1000, rotation = True, fixed_direction = True, no_solvent = True)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(subdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")