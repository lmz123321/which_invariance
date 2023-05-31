import torch
import pandas as pd

def dfTotensor(df):
    """
    convert a DataFrame to a torch Tensor
    caution: this can be quite *slow*, so try to use it sparsely
    """
    return (torch.from_numpy(df.values)).float()

def concat(dic, keys):
    r"""
    Functs: - return torch.cat([dic[key1],dic[key2],...],dim=1)
    """
    values = list()
    for key in keys:
        values.append(dic[key])
    return torch.cat(values,dim=1)