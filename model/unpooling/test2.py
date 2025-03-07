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
from trainer import GANTrainer


### Load data
batch_size = 64
mol_data = torch.load('qm9_smiles_noar.pt')
data_list = torch.load('qm9_data_noar.pt')
data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['edge_index', 'y'])

device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")

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

train = GANTrainer(d=gcn_model, g=generator, \
                   rand_dim=256, train_folder='ULGAN_QM9', \
                   tot_epoch_num=200, eval_iter_num=1000, \
                   batch_size=64, \
                   device=device, d_add=added_d, \
                   learning_rate_g=2e-4, learning_rate_d=1e-4, \
                   lambda_g=10.0, \
                   max_train_G=2, \
                   tresh_add_trainG=0.2, \
                   use_loss='wgan', \
                   g_out_prob=True, \
                   lambda_rl=5e-3, lambda_nonodes = 0., 
                   lambda_noedges = 0., \
                   qm9=True, \
                   without_ar=True, \
                  )


train.train(data_loader, verbose=False, use_data_x = 10, use_data_edgeattr=3, \
        evaluate_num=1000, mol_data=mol_data, alter_trainer=True, NN=200, reinforce_acclerate=True)
