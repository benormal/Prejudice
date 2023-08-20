import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import numpy as np
import random


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()
        week_num_size = 6
        dow_size = 7
        type_size = 12
        hour_size = 25
        building_size = 101
        
        self.week_num_embedd = nn.Embedding(week_num_size, d_model)
        self.dow_embedd = nn.Embedding(dow_size, d_model)
        self.type_embedd = nn.Embedding(type_size, d_model)
        self.hour_embedd = nn.Embedding(hour_size, d_model)
        self.building_embedd = nn.Embedding(building_size, d_model)
        
        
    def forward(self, x):
        week_num_x = self.week_num_embedd(x[:, :, 0])
        dow_x = self.dow_embedd(x[:, :, 1])
        type_x = self.type_embedd(x[:, :, 2])
        hour_x = self.hour_embedd(x[:, :, 3])
        building_x = self.building_embedd(x[:, :, 4])
        
        return week_num_x + dow_x + type_x + hour_x + building_x
    
    
class TCN(nn.Module):
    def __init__(self, in_dim=5, out_dim=168, embedd_dim=64, residual_channels=128, dilation_channels=128, 
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=48, layers=3):
        super(TCN, self).__init__()
        self.blocks = blocks
        self.layers = layers
        
        self.temporal_embedding = TemporalEmbedding(embedd_dim)
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        self.start_conv = nn.Conv1d(in_channels=in_dim + embedd_dim,
                                    out_channels=residual_channels,
                                    kernel_size=1)
        receptive_field = 1
        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size, dilation=new_dilation))
                
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size, dilation=new_dilation))
                
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1))
                
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1))
                
                self.bn.append(nn.BatchNorm1d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=1,
                                    bias=True)

        self.receptive_field = receptive_field


    def forward(self, input, embedd):
        in_len = input.size(2)
        
        embedd = self.temporal_embedding(embedd)
        embedd = embedd.transpose(1, 2)
        
        input = torch.cat([input, embedd], dim=1)
        
        if in_len < self.receptive_field:
            x = nn.functional.pad(input,(0, self.receptive_field - in_len))
        else:
            x = input
        
        x = self.start_conv(x)
        skip = 0
        
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            
            # parametrized skip connection
            
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
                
            skip = s + skip
            
            x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.size(2):]
            
            x = self.bn[i](x)
            
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        return x