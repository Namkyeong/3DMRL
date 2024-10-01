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

from utils.chem import get_graph_from_smile, get_scaffold
from sklearn.model_selection import train_test_split


class Chromophore(InMemoryDataset):
    def __init__(self, root, name, log_target = False, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log_target = log_target

        if self.name == "Absorption max (nm)":
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/chr_abs.csv")
        
        elif self.name == "Emission max (nm)":
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/chr_emi.csv")
        
        elif self.name == "Lifetime (ns)":
            self.raw_file_path = os.path.join(self.root, '{}'.format(self.name), "raw/chr_lif.csv")

        else:
            raise Exception

        super(Chromophore, self).__init__(root, transform, pre_transform, pre_filter)

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

            solute = raw_df.loc[idx]["Chromophore"]
            solute = Chem.MolFromSmiles(solute)
            # solute = Chem.AddHs(solute)
            solute_graph = get_graph_from_smile(solute, idx)
            solute_list.append(solute_graph)

            solvent = raw_df.loc[idx]["Solvent"]
            solvent = Chem.MolFromSmiles(solvent)
            # solvent = Chem.AddHs(solvent)
            solvent_graph = get_graph_from_smile(solvent, idx)
            solvent_list.append(solvent_graph)

            delta_g = raw_df.loc[idx][self.name]
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
    
    def get_molecule_split(self):
        
        df = pd.read_csv(self.raw_file_path, sep = ",")
        mol_ids = df["Chromophore"].unique()
        _train_ids, _test_ids = train_test_split(mol_ids, test_size=0.2, random_state=42)
        train_ids = np.where(df['Chromophore'].isin(_train_ids))[0]
        test_ids = np.where(df['Chromophore'].isin(_test_ids))[0]

        assert len(train_ids) + len(test_ids) == len(df)
        assert len(set(train_ids).intersection(set(test_ids))) == 0

        return train_ids, test_ids
    
    def get_scaffold_split(self):
        
        df = pd.read_csv(self.raw_file_path, sep = ",")
        df['scaffold'] = df['Chromophore'].apply(get_scaffold)

        # Sort Scaffold by counting the numbers
        scaffold_counts = df['scaffold'].value_counts().reset_index()
        scaffold_counts.columns = ['scaffold', 'count']
        scaffold_counts = scaffold_counts.sort_values(by='count', ascending=False)

        train_fraction = 0.8
        cumulative_count = scaffold_counts['count'].cumsum()
        total_count = scaffold_counts['count'].sum()
        cutoff = cumulative_count.searchsorted(train_fraction * total_count)

        train_scaffolds = scaffold_counts.iloc[:cutoff+1]['scaffold']
        test_scaffolds = scaffold_counts.iloc[cutoff+1:]['scaffold']

        train_ids = np.where(df['scaffold'].isin(train_scaffolds))[0]
        test_ids = np.where(df['scaffold'].isin(test_scaffolds))[0]

        assert len(train_ids) + len(test_ids) == len(df)
        assert len(set(train_ids).intersection(set(test_ids))) == 0

        return train_ids, test_ids


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
    
    # Chr : (Chromophore, Solvent, Absorption max (nm)/ Emission max (nm)/ Lifetime (ns))
    dataset = Chromophore(DATA_PATH, "Absorption max (nm)")
    dataset = Chromophore(DATA_PATH, "Emission max (nm)")
    dataset = Chromophore(DATA_PATH, "Lifetime (ns)", log_target=True)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")