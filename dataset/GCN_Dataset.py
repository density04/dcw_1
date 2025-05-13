from torch.utils.data import Dataset
import numpy as np
import torch
import random
from rdkit import Chem

import data_utils as DATA_UTILS

MAX_ATOM = 100

class GraphDataset(Dataset):
    def __init__(self, data):
        self.feature_list = []
        self.num_atom_list = []
        self.adj_list = []
        smiles_list, label_list = data
        self.label_list = torch.from_numpy(label_list)
        self.__process(smiles_list)

    def __process(self, smiles_list):
        num_atom_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                n = mol.GetNumAtoms()
                adj = DATA_UTILS.get_adj(mol)
                feature = DATA_UTILS.get_atom_feature(mol)
                num_atom_list.append(n)
                self.adj_list.append(adj)
                self.feature_list.append(feature)
        self.num_atom_list = np.array(num_atom_list)

    def __getitem__(self,idx):
        sample = dict()
        sample['N'] = self.num_atom_list[idx]
        sample['A'] = self.adj_list[idx]
        sample['Y'] = self.label_list[idx]
        sample['F'] = self.feature_list[idx]
        return sample

    def __len__(self):
        return len(self.feature_list)

    @staticmethod 
    def collate_fn(batch):
        sample = dict()
        num_atom_list = []
        for b in batch:
            num_atom_list.append(b['N'])
        num_atom_list = np.array(num_atom_list)
        max_atom = MAX_ATOM
        adj_list = []
        af_list = []
        y_list = []
        for b in batch:
            k = b['N']
            padded_adj = np.zeros((max_atom,max_atom))
            padded_adj[:k,:k] = b['A']
            padded_feature = np.zeros((max_atom,31))
            padded_feature[:k,:31] = b['F']
            af_list.append(padded_feature)
            adj_list.append(padded_adj)
            y_list.append(b['Y'])
        sample['N'] = torch.Tensor(num_atom_list)
        sample['A'] = torch.Tensor(adj_list)
        sample['F'] = torch.Tensor(af_list)
        sample['Y'] = torch.Tensor(y_list)
        return sample

#===================================#
