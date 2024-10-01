import sys
sys.path.append('.')

import glob
import copy
import os
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import random

from typing import Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import IndexType

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import get_graph_from_smile, get_scaffold
from sklearn.model_selection import train_test_split


class DDI(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.raw_folder_path = os.path.join(self.root, self.name, "raw")

        super(DDI, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_dataset()
        return
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):

        print("Start Preprocessing {} dataset".format(self.name))

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

        df = pd.DataFrame({'Solute': smiles1, 'Solvent': smiles2, 'label': labels})
        df = df[df["label"] == 1]
        df = df.drop_duplicates().reset_index(drop=True)

        solute_smiles_list, solvent_smiles_list, solute_list, solvent_list, targets = list(), list(), list(), list(), list()

        print("Start Creating Positive Samples ...")
        
        for idx in tqdm(range(len(df))):
        # for idx in tqdm(range(200)):

            try:
                solute = df.loc[idx]["Solute"]
                solute_smiles = solute
                solute = Chem.MolFromSmiles(solute)
                solute = Chem.AddHs(solute)
                solute_graph = get_graph_from_smile(solute, idx)

                solvent = df.loc[idx]["Solvent"]
                solvent_smiles = solvent
                solvent = Chem.MolFromSmiles(solvent)
                solvent = Chem.AddHs(solvent)
                solvent_graph = get_graph_from_smile(solvent, idx)

                target = torch.tensor(1, dtype = torch.long)

                solute_list.append(solute_graph)
                solute_smiles_list.append(solute_smiles)
                solvent_list.append(solvent_graph)
                solvent_smiles_list.append(solvent_smiles)
                targets.append(target)

            except Exception as e:
                
                import traceback
                print(f"An error occurred: {e}")
                traceback.print_exc()

        num_positive_samples = len(targets)
        print("Number of Created Positive Samples: {}".format(num_positive_samples))

        # Build Negative Pairs
        print("Start Creating Negative Samples ...")
        molecules = np.unique(list(df["Solute"]) + list(df["Solvent"]))

        all_pairs = list(itertools.product(molecules, molecules))
        original_pairs = set(zip(df['Solute'], df['Solvent'])) | set(zip(df['Solvent'], df['Solute']))
        negative_samples = [pair for pair in all_pairs if pair not in original_pairs]
        
        random_negative_samples = random.sample(negative_samples, num_positive_samples)
        
        assert original_pairs.isdisjoint(set(random_negative_samples)), "Negative samples overlap with original pairs!"

        negative_df = pd.DataFrame(random_negative_samples, columns=['Solute', 'Solvent'])

        for idx in tqdm(range(len(negative_df))):

            try:
                solute = negative_df.loc[idx]["Solute"]
                solute_smiles = solute
                solute = Chem.MolFromSmiles(solute)
                solute = Chem.AddHs(solute)
                solute_graph = get_graph_from_smile(solute, idx)

                solvent = negative_df.loc[idx]["Solvent"]
                solvent_smiles = solvent
                solvent = Chem.MolFromSmiles(solvent)
                solvent = Chem.AddHs(solvent)
                solvent_graph = get_graph_from_smile(solvent, idx)

                target = torch.tensor(0, dtype = torch.long)
                
                solute_list.append(solute_graph)
                solute_smiles_list.append(solute_smiles)
                solvent_list.append(solvent_graph)
                solvent_smiles_list.append(solvent_smiles)
                targets.append(target)
            
            except Exception as e:
                
                import traceback
                print(f"An error occurred: {e}")
                traceback.print_exc()

        print("Number of Created Negative Samples: {}".format(len(targets) - num_positive_samples))
        
        df = pd.DataFrame([solute_smiles_list, solvent_smiles_list], index=['Solute', 'Solvent']).transpose()
        solute_graphs, solute_slices = self.collate(solute_list)
        solvent_graphs, solvent_slices = self.collate(solvent_list)

        torch.save((solute_graphs, solute_slices, solvent_graphs, solvent_slices, targets, df), self.processed_paths[0])


    def load_dataset(self):

        self.graphs, self.slices, self.solvent_graphs, self.solvent_slices, self.targets, self.smiles_df = torch.load(self.processed_paths[0])
        
        return
    

    def get_random_split(self, seed):

        train_ids, test_ids = train_test_split(self.smiles_df.index, test_size=0.4, random_state=seed)
        
        assert len(train_ids) + len(test_ids) == len(self.smiles_df)
        assert len(set(train_ids).intersection(set(test_ids))) == 0

        valid_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=seed)

        return np.asarray(train_ids), np.asarray(valid_ids), np.asarray(test_ids)
    

    def get_molecule_split(self, seed):

        all_drugs = list(set(self.smiles_df['Solute']).union(set(self.smiles_df['Solvent'])))

        test_drug_ratio = 0.4
        test_drugs = set(np.random.choice(all_drugs, size=int(len(all_drugs) * test_drug_ratio), replace=False))

        test_ids = self.smiles_df[(self.smiles_df['Solute'].isin(test_drugs)) | (self.smiles_df['Solvent'].isin(test_drugs))].index

        train_ids = self.smiles_df[~((self.smiles_df['Solute'].isin(test_drugs)) | (self.smiles_df['Solvent'].isin(test_drugs)))].index

        assert len(train_ids) + len(test_ids) == len(self.smiles_df)
        assert len(set(train_ids).intersection(set(test_ids))) == 0

        valid_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=seed)

        return np.asarray(train_ids), np.asarray(valid_ids), np.asarray(test_ids)
    

    def get_scaffold_split(self, seed):

        self.smiles_df['Scaffold1'] = self.smiles_df['Solute'].apply(get_scaffold)
        self.smiles_df['Scaffold2'] = self.smiles_df['Solvent'].apply(get_scaffold)
        
        self.smiles_df['scaffold'] = self.smiles_df['Solute'].apply(get_scaffold)

        all_scaffolds = set(self.smiles_df['Scaffold1']).union(set(self.smiles_df['Scaffold2']))

        test_scaffold_ratio = 0.4
        test_scaffolds = set(np.random.choice(list(all_scaffolds), size=int(len(all_scaffolds) * test_scaffold_ratio), replace=False))

        test_ids = self.smiles_df[(self.smiles_df['Scaffold1'].isin(test_scaffolds)) | (self.smiles_df['Scaffold2'].isin(test_scaffolds))].index
        
        train_ids = self.smiles_df[~((self.smiles_df['Scaffold1'].isin(test_scaffolds)) | (self.smiles_df['Scaffold2'].isin(test_scaffolds)))].index

        assert len(train_ids) + len(test_ids) == len(self.smiles_df)
        assert len(set(train_ids).intersection(set(test_ids))) == 0

        valid_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=seed)

        return np.asarray(train_ids), np.asarray(valid_ids), np.asarray(test_ids)


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
    
    dataset = DDI(DATA_PATH, "ChChMiner")
    dataset = DDI(DATA_PATH, "ZhangDDI")
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")