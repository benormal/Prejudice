import torch
import numpy as np
import time
import os
import pickle
import copy

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, x, y, e):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.e = torch.Tensor(e).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sample = {"x": self.x[idx], "y": self.y[idx], "e": self.e[idx]}
        
        return sample
    
    
class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
    
def load_dataset(data_dir, batch_size):
    data = {}
    for category in ['train', 'valid']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['e_' + category] = cat_data['e']
        
    scaler = StandardScaler(data["x_train"][:, :, 0].mean(), data["x_train"][:, :, 0].std())
    data["scaler"] = scaler
    data["x_train"][:, :, 0] = scaler.transform(data["x_train"][:, :, 0])
    data["x_valid"][:, :, 0] = scaler.transform(data["x_valid"][:, :, 0])
    
    train_dataset = MyDataset(data['x_train'], data['y_train'], data['e_train'])
    val_dataset = MyDataset(data['x_valid'], data['y_valid'], data['e_valid'])
    
    data["train_loader"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    data["val_loader"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_smape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels) / (torch.abs(labels) + torch.abs(preds)) * 100
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)