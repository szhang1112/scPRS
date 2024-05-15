"""
Example training script for 5-fold cross-validation.
"""
from model import scPRS
import pickle as pkl
import os
import gc
import networkx as nx
import numpy as np
import pandas as pd
import argparse
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")

data = pkl.load(open(f'{data_path}', 'rb'))
X = data[0] #cell-level PRSs
y = data[1] #labels
edge= data[2] #cell-cell similarity network

means = np.nanmean(X, 0, keepdims=True)
stds = np.nanstd(X, 0, keepdims=True)
stds[np.isnan(stds)] = 1
X = (X - means) / (stds + 1e-7)
X[np.isnan(X)] = 0

L = torch.FloatTensor(edge)
graph = nx.Graph()
for i in range(X.shape[1]):
    graph.add_edge(i, i)
for i in range(len(edge)):
    graph.add_edge(edge[i,0], edge[i,1])
device = torch.device('cuda')
normalized_laplacian_matrix = torch.FloatTensor(nx.normalized_laplacian_matrix(graph).toarray()).to(device)

X = torch.FloatTensor(X).to(device)
y = torch.FloatTensor(y).to(device)
L = torch.FloatTensor(L).long().to(device).T
edge_weight = torch.ones(L.shape[1]).to(device)

###set hyperparameters based on your choice!
l1 = 1
l2 = 10
lgraph = 10
layer = 1

print('!!!!', X.shape)
perf = {}
for score in ['auc', 'aupr']:
    perf[score] = {}
    for split in ['train', 'val', 'test']:
        perf[score][split] = []
        for kf in range(5):
            perf[score][split].append([])

kf = KFold(5, shuffle=True)
for i, (train_ind, test_ind) in (enumerate(kf.split(X, y))):
    train_ind, val_ind = train_test_split(train_ind, test_size=0.25)
    X_train = (X[train_ind])
    y_train = (y[train_ind])
    y_train_np= y_train.detach().cpu().numpy()
    X_val = (X[val_ind])
    y_val = (y[val_ind])
    y_val_np= y_val.detach().cpu().numpy()
    X_test = (X[test_ind])
    y_test = (y[test_ind])
    y_test_np= y_test.detach().cpu().numpy()

    model = scPRS(X.shape[-1], X.shape[1], layer).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_epoch = 200
    for epoch in tqdm(range(n_epoch)):
        y_id = pd.DataFrame(y_train_np).reset_index()
        y_id_pos = y_id[y_id[0]==1].sample(n=len(y_id[0])-int(sum(y_id[0])), replace=True, random_state=epoch)
        select_idn = np.array(list(y_id_pos['index']) + list(y_id[y_id[0] == 0]['index']))
        dataset = TensorDataset(X_train[select_idn], y_train[select_idn])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        
        for xx, yy in dataloader:
            optim.zero_grad()
            out = model(xx, L, edge_weight)[:, 0]
            loss = F.binary_cross_entropy_with_logits(out, yy)
            loss2 = abs(model.parameter).mean()*l1 + ((model.pred.weight**2).mean())*l2
            loss3 = model.pred.weight @ normalized_laplacian_matrix @ model.pred.weight.T * lgraph \
                    / len(model.pred.weight)
            loss = loss + loss2 + loss3
            loss.backward()
            optim.step()

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                outv = F.sigmoid(model(X_val, L, edge_weight)[:, 0])
                outt = F.sigmoid(model(X_test, L, edge_weight)[:, 0])
                outtr = model(X_train[:len(X_train)//2], L, edge_weight)[:, 0]
                outtr2 = model(X_train[len(X_train)//2:], L, edge_weight)[:, 0]

                aucv = roc_auc_score(y_val_np, outv.cpu().detach().numpy())
                auprv = average_precision_score(y_val_np, outv.cpu().detach().numpy())
                auct = roc_auc_score(y_test_np, outt.cpu().detach().numpy())
                auprt = average_precision_score(y_test_np, outt.cpu().detach().numpy())
                auctr = roc_auc_score(y_train.cpu().detach().numpy(), list(outtr.cpu().detach().numpy())
                                                      +list(outtr2.cpu().detach().numpy()))
                auprtr = average_precision_score(y_train.cpu().detach().numpy(), list(outtr.cpu().detach().numpy())
                                                      +list(outtr2.cpu().detach().numpy()))

                perf['auc']['train'][i].append(auctr)
                perf['auc']['val'][i].append(aucv)
                perf['auc']['test'][i].append(auct)
                perf['aupr']['train'][i].append(auprtr)
                perf['aupr']['val'][i].append(auprv)
                perf['aupr']['test'][i].append(auprt)


#save perf for tracking and model selection
np.save(f'{perf_file_path}', perf)


