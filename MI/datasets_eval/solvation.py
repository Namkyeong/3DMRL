import sys
sys.path.append('.')

import copy
import os
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import IndexType

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import get_graph_from_smile


class Solvation(InMemoryDataset):
    def __init__(self, root, name, log_target = False, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log_target = log_target

        if self.name == "Abraham":
            self.solute_col = "smiles_solute"
            self.solvent_col = "smiles_solvent"
            self.target = "dGsolv_avg [kcal/mol]"
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/abraham.csv")
        
        elif self.name == "CombiSolv":
            self.solute_col = "mol solute"
            self.solvent_col = "mol solvent"
            self.target = "target Gsolv kcal"
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/combisolv_exp.csv")
        
        elif self.name == "CompSol":
            self.solute_col = "smiles_solute"
            self.solvent_col = "smiles_solvent"
            self.target = "dGsolv_avg [kcal/mol]"
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/compsol.csv")

        elif self.name == "FreeSol":
            self.solute_col = "smiles_solute"
            self.solvent_col = "smiles_solvent"
            self.target = "dGsolv_avg [kcal/mol]"
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/freesol.csv")
        
        elif self.name == "MNSol":
            self.solute_col = "smiles_solute"
            self.solvent_col = "smiles_solvent"
            self.target = "dGsolv_avg [kcal/mol]"
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/mnsol.csv")

        else:
            raise Exception

        super(Solvation, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_dataset()
        return
    

    @property
    def processed_dir(self):
        return os.path.join(self.root, '{}'.format(self.name), 'processed')
    

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
    

    def process(self):
        
        raw_df = pd.read_csv(self.raw_file_path, sep = ",")

        solute_list, solvent_list, targets = list(), list(), list()
        
        for idx in tqdm(range(len(raw_df))):

            solute = raw_df.loc[idx][self.solute_col]
            solute = Chem.MolFromSmiles(solute)
            solute = Chem.AddHs(solute)
            solute_graph = get_graph_from_smile(solute, idx)
            solute_list.append(solute_graph)

            solvent = raw_df.loc[idx][self.solvent_col]
            solvent = Chem.MolFromSmiles(solvent)
            solvent = Chem.AddHs(solvent)
            solvent_graph = get_graph_from_smile(solvent, idx)
            solvent_list.append(solvent_graph)

            delta_g = raw_df.loc[idx][self.target]
            targets.append(delta_g)
        
        solute_graphs, solute_slices = self.collate(solute_list)
        solvent_graphs, solvent_slices = self.collate(solvent_list)

        torch.save((solute_graphs, solute_slices, solvent_graphs, solvent_slices, targets), self.processed_paths[0])


    def load_dataset(self):

        self.graphs, self.slices, self.solvent_graphs, self.solvent_slices, self.targets = torch.load(self.processed_paths[0])
        
        if self.log_target:
            self.targets = np.log(self.targets)
        else:
            pass

        return
    

    def get(self, idx):

        target = self.targets[idx]

        solute_data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[solute_data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            solute_data[key] = item[s]
        
        solvent_data = Data()
        for key in self.solvent_graphs.keys:
            item, slices = self.solvent_graphs[key], self.solvent_slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[solvent_data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            solvent_data[key] = item[s]
        
        return solute_data, solvent_data, target
    

    def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if idx is None:
            data_list = [self.get(i) for i in self.indices()]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = None

        solute_list, solvent_list, targets = list(), list(), list()

        for data in data_list:
            solute_list.append(data[0])
            solvent_list.append(data[1])
            targets.append(data[2])

        dataset.graphs, dataset.slices = self.collate(solute_list)
        dataset.solvent_graphs, dataset.solvent_slices = self.collate(solvent_list)
        dataset.targets = targets

        return dataset
    
    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":

    from torch_geometric.loader import DataLoader as pyg_DataLoader
    
    DATA_PATH = "./data_eval"
    batch_size = 32
    num_workers = 6
    
    dataset = Solvation(DATA_PATH, "Abraham")
    dataset = Solvation(DATA_PATH, "CombiSolv")
    dataset = Solvation(DATA_PATH, "CompSol")
    dataset = Solvation(DATA_PATH, "FreeSol")
    dataset = Solvation(DATA_PATH, "MNSol")
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")