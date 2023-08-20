import torch
import torch.nn as nn
import numpy as np
import time
import util
import torch.optim as optim
import os
import pickle
import argparse
import copy

from tcn import *
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='gpu device name boram have to revise mps')
parser.add_argument('--data', type=str, default="sample_data", help="data path")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=30, help='epoch num')
parser.add_argument('--print_every', type=int, default=300, help='')
parser.add_argument('--exp', type=str, default='2013', help='')
parser.add_argument('--in_dim', type=int, default=5, help='value feature num')
parser.add_argument('--embedd_dim', type=int, default=64, help='embedd dimension size')
parser.add_argument('--residual_channels', type=int, default=128, help='residual channel size')
parser.add_argument('--dilation_channels', type=int, default=128, help='dilation channel size')
parser.add_argument('--skip_channels', type=int, default=256, help='skip channel size')
parser.add_argument('--end_channels', type=int, default=512, help='end channel size')
parser.add_argument('--blocks', type=int, default=48, help='control receptive field size')
parser.add_argument('--layers', type=int, default=3, help='control receptive field size')

args = parser.parse_args()
args = vars(args)


def train(argument):
    folder = argument["data"]
    device = torch.device(argument["device"])
    dataloader = util.load_dataset(argument["data"], argument["batch_size"])
    
    model = TCN(in_dim=argument["in_dim"], out_dim=168, embedd_dim=argument["embedd_dim"], residual_channels=argument["residual_channels"],
               dilation_channels=argument["dilation_channels"], skip_channels=argument["skip_channels"], end_channels=argument["end_channels"], 
                kernel_size=2, blocks=argument["blocks"], layers=argument["layers"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    mae_loss = util.masked_mae
    rmse_loss = util.masked_rmse
    mape_loss = util.masked_mape
    smape_loss = util.masked_smape
    
    clip = 5
    
    train_loader = dataloader["train_loader"]
    val_loader = dataloader["val_loader"]
    
    his_loss = []
    val_time = []
    train_time = []
    best_loss = 100000000000000000
    exp = argument["exp"]
    
    for i in range(1, argument["epoch"] + 1, 1):
        train_loss = []
        train_rmse = []
        train_mape = []
        train_smape = []
        
        t1 = time.time()
        for iter, item in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            trainx = item["x"].to(device).transpose(1, 2)
            trainy = item["y"].to(device)
            traine = item["e"].to(device)
            
            output = model(trainx, traine)
            output = dataloader["scaler"].inverse_transform(output)
            
            loss = mae_loss(output, trainy, 0.0)
            loss.backward()
            
            optimizer.step()
            
            rmse = rmse_loss(output, trainy, 0.0)
            mape = mape_loss(output, trainy, 0.0)
            smape = smape_loss(output, trainy, 0.0)
            
            train_loss.append(loss.item())
            train_rmse.append(rmse.item())
            train_mape.append(mape.item())
            train_smape.append(smape.item())
            
            if iter % argument["print_every"] == 0:
                print(f"Iter {iter} Train loss: {train_loss[-1]} RMSE Loss: {train_rmse[-1]} MAPE Loss: {train_mape[-1]} SMAPE Loss: {train_smape[-1]}", flush=True)
                
        t2 = time.time()
        train_time.append(t2 - t1)
        
        valid_loss = []
        valid_rmse = []
        valid_mape = []
        valid_smape = []
        
        s1 = time.time()
        for iter, item in enumerate(val_loader):
            model.eval()
            
            valx = item["x"].to(device).transpose(1, 2)
            valy = item["y"].to(device)
            vale = item["e"].to(device)
            
            output = model(valx, vale)
            output = dataloader["scaler"].inverse_transform(output)
            
            loss = mae_loss(output, valy, 0.0)
            rmse = rmse_loss(output, valy, 0.0)
            mape = mape_loss(output, valy, 0.0)
            smape = smape_loss(output, valy, 0.0)
            
            valid_loss.append(loss.item())
            valid_rmse.append(rmse.item())
            valid_mape.append(mape.item())
            valid_smape.append(smape.item())
            
        s2 = time.time()
        log = "Epoch : {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mape = np.mean(train_mape)
        mtrain_smape = np.mean(train_smape)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mape = np.mean(valid_mape)
        mvalid_smape = np.mean(valid_smape)
        
        his_loss.append(mvalid_smape)
        
        if mvalid_smape < best_loss:
            best_loss = mvalid_smape
            
            torch.save(model.state_dict(), f"tcn_best{exp}.pth")
            
        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {.4f} Training Time: {:.4f}/epoch"
        # print(mtrain_loss, mtrain_mae, mvalid_loss, mvalid_mae)
        print("EPoch:", i, "Train Loss:", mtrain_loss, "Train RMSE:", mtrain_rmse, "Train MAPE:", mtrain_mape, "Train SMAPE:", mtrain_smape, "Valid Loss", mvalid_loss, "Valid RMSE:", mvalid_rmse, "Valid MAPE:", mvalid_mape, "Valid SMAPE:", mvalid_smape, "Train Time:", t2 - t1)
        
    bestid = np.argmin(his_loss)
    print("load parameter:", "epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)))
    model.load_state_dict(torch.load(f"tcn_best{exp}.pth"))
    
    
    with open(f"{folder}/test.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    output_dict = dict()
    
    for num in range(1, 101, 1):
        building_data = test_data[num]
        x = torch.Tensor(building_data["x"]).to(device).transpose(1, 2)
        e = torch.LongTensor(building_data["e"]).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(x, e)
            output = dataloader["scaler"].inverse_transform(output)
            output = output.squeeze(0).squeeze(1).detach().cpu().numpy().tolist()
            
        output_dict[num] = output
    
    with open(f"tcn_best{exp}_result.pkl", "wb") as f:
        pickle.dump(output_dict, f)
        
if __name__ == "__main__":
    t1 = time.time()
    train(args)
    t2 = time.time()
    print("Total time spent : {:.4f}".format(t2 - t1))