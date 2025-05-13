from torch.utils.data import Dataset
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import data_utils as DATA_UTILS

MAX_ATOM = 100

class SmilesDataset(Dataset):
    def __init__(self, smiles, stereo=True):
        self.smiles = smiles
        self.c_to_i = DATA_UTILS.C_TO_I
        self.stereo = stereo
        self.n_char = DATA_UTILS.N_CHAR

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        keydata=self.smiles[idx]
        if self.stereo :
            isomers=list(EnumerateStereoisomers(Chem.MolFromSmiles(keydata)))
            keydata = Chem.MolToSmiles(isomers[random.randint(0, len(isomers)-1)], isomericSmiles=True)
        else :
            keydata=Chem.MolToSmiles(Chem.MolFromSmiles(keydata))
        keydata += 'Q'
        sample = dict()
        sample['X'] = torch.from_numpy(np.array([self.c_to_i[c] for c in keydata]))
        sample['L'] = len(keydata)-1
        sample['n_char'] = self.n_char
        sample['smiles'] = self.smiles[idx] 
        return sample

    @staticmethod 
    def collate_fn(batch) :
        sample = dict()
        n_char = batch[0]['n_char']
        X = torch.nn.utils.rnn.pad_sequence([b['X'] for b in batch], batch_first=True, padding_value = n_char-1)
        L = torch.Tensor([b['L'] for b in batch])
        S = [b['smiles'] for b in batch]
        sample['X'] = X
        sample['L'] = L
        sample['S'] = S
        return sample