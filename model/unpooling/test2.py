import torch
import os
from torch import nn, optim
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader
import numpy as np
from gcn_model_sim_summ_qm import UnpoolGeneratorQ
from ugcn_model_summ_2 import pre_GCNModel_edge_3eos
from gcn_model_sim_summ_2 import *
from ugcn_model_summ_2 import pre_GCNModel_edge_3eos
import torch


device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

# Build discriminators

gcn_model = pre_GCNModel_edge_3eos(in_dim=10, \
                                   hidden_dim=128, \
                                   edge_dim=3, \
                                   edge_hidden_dim=32, \
                                   lin_hidden_dim=128, \
                                   out_hidden_dim=256, \
                                   device=device, \
                                    check_batch=None, 
                                    useBN=True, \
                                    droprate=0.3, 
                                    pool_method='trivial', \
                                    add_edge_link=False, 
                                    add_conv_link=True, \
                                    outBN=False, out_drop=0.3, 
                                    out_divide=4.0, 
                                    add_edge_agg=False, 
                                    real_trivial=False, 
                                    final_layers=2, 
                                    add_trivial_feature=False).to(device)


model_path = "model/unpooling/ul_gan_qm9.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

added_d = pre_GCNModel_edge_3eos(in_dim=10, \
                                   hidden_dim=128, \
                                   edge_dim=3, \
                                   edge_hidden_dim=32, \
                                   lin_hidden_dim=128, \
                                   out_hidden_dim=256, \
                                   device=device, \
                                    check_batch=None, 
                                    useBN=False, \
                                    droprate=0.3, 
                                    pool_method='trivial', \
                                    add_edge_link=False, \
                                    add_conv_link=False, \
                                    outBN=False, out_drop=0.3, 
                                    out_divide=4.0, 
                                    add_edge_agg=False, 
                                    real_trivial=True, 
                                    final_layers=2, 
                                    add_trivial_feature=True).to(device)
added_d.e1_ind = 0.0 # Not consider the mean of edges, just the sum


# Build generator

generator = UnpoolGeneratorQ(in_dim=256, \
                            edge_dim=3, 
                            node_dim=10, 
                            node_hidden_dim=[128, 128, 128, 64, 64], 
                            edge_hidden_dim=64, \
                            use_x_bn=True, 
                            use_e_bn=True, 
                            unpool_bn=True, 
                            link_bn=False, 
                            attr_bn=True, 
                            skip_z=True, 
                            skip_zdim=None, 
                            conv_type='nn', 
                            device=device, 
                            last_act='leaky', 
                            link_act='leaky', \
                            unpool_para=dict(add_perference=True, roll_bn=False, roll_simple=True, \
                                            add_additional_link=True), \
                             without_aroma=True).to(device)
