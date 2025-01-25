import logging
import torch
import torch.utils.data as data
import numpy as np
import json, utils
import os
import gzip
import tqdm

from collections import OrderedDict
import os
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sentence_transformers import models, losses, util, datasets, evaluation
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors


class HerbDataset(data.Dataset):
    def __init__(self, formula_list, label_list, id_dict):
        self.formula_list = formula_list
        self.label_list = label_list
        self.len = len(self.formula_list)
        self.id_dict = id_dict

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        formula = self.formula_list[index]
        formula_id = self.id_dict[formula]
        return [formula_id, self.label_list[index]]
