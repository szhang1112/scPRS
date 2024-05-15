import torch
import random
import argparse

import torch.nn.functional as F
import pickle as pkl
import networkx as nx
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model.model import scPRS


def data_process(data_path):
    data = pkl.load(open(data_path, 'rb'))
    X = data[0]
    y = data[1]
    edge = data[2]
    means = X.mean(0, keepdims=True)
    stds = X.std(0, keepdims=True)
    stds[np.isnan(stds)] = 1
    X = (X - means) / (stds + 1e-7)
    print(np.isnan(X).sum())

    graph = nx.Graph()
    for i in range(X.shape[1]):
        graph.add_edge(i, i)
    for i in range(len(edge)):
        graph.add_edge(edge[i, 0], edge[i, 1])
    normalized_laplacian_matrix = nx.normalized_laplacian_matrix(graph).toarray()
    return X, y, edge, normalized_laplacian_matrix

def run_model(args, X, y, edge, normalized_laplacian_matrix):
    device = torch.device('cuda:0')

    normalized_laplacian_matrix = torch.FloatTensor(normalized_laplacian_matrix).to(device)
    y = torch.FloatTensor(y).to(device)
    L = torch.FloatTensor(edge).long().to(device).T
    weights = []
    for random_state in tqdm(range(100)):
        torch.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        X = torch.FloatTensor(X).to(device)
        y_train_np = y.detach().cpu().numpy()
        model = scPRS(X.shape[-1], X.shape[1], args.layer).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in (range(args.n_epoch)):
            y_id = pd.DataFrame(y_train_np).reset_index()
            y_id_pos = y_id[y_id[0]==1].sample(n=len(y_id[0])-int(sum(y_id[0])), replace=True, random_state=epoch)
            select_idn = np.array(list(y_id_pos['index']) + list(y_id[y_id[0] == 0]['index']))
            dataset = TensorDataset(X[select_idn], y[select_idn])
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
            for xx, yy in dataloader:
                optim.zero_grad()
                out = model(xx, L)[:, 0]
                loss = F.binary_cross_entropy_with_logits(out, yy)
                loss2 = abs(model.pred.weight).mean()*args.l1 + (model.pred.weight**2).mean()*args.l2
                loss3 = model.pred.weight @ normalized_laplacian_matrix @ model.pred.weight.T * args.lgraph / len(
                    model.pred.weight)
                loss = loss + loss2 + loss3
                loss.backward()
                optim.step()
        torch.save(model, f'{args.model_save_path}_{random_state}.pt')
        weights.append(model.pred.weight)
        weights = [x.detach().cpu().numpy() for x in weights]
    weights = np.array(weights)[:, 0, :]
    return weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--l1", type=float, default=0)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--lgraph", type=float, default=0)
    parser.add_argument("--n_epoch", type=int, default=350)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()

    X, y, edge, normalized_laplacian_matrix = data_process(args.data_path)
    disease_cell_association = run_model(args, X, y, edge, normalized_laplacian_matrix)
    pd.DataFrame(disease_cell_association).reset_index().rename(columns={"index":"repeat_id"}).to_csv(args.result_path)
    