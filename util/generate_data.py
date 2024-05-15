import os
import argparse
import pickle as pkl
import anndata as ann
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm


def step1_process_prs_data(data_path):
    res = []
    for item in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for filename in tqdm(os.listdir(f'./{data_path}/{item}')):
            if 'score' in filename:
                data = pd.read_csv(f'./{data_path}/{item}/{filename}', sep='\t')
                data['sim_cutoff'] = item
                data['filename'] = filename
                res.append(data)

    res = pd.concat(res)
    res['type2'] = res['filename'].apply(lambda x: '.'.join(x.split('.')[1:-1]))
    d_type2 = {item: i for i, item in enumerate(set(res['type2']))}
    d_sim_cutoff = {item: i for i, item in enumerate(set(res['sim_cutoff']))}
    res['cell'] = res['filename'].apply(lambda x: x.split('.')[0])
    res = res.drop(['filename'], 1)
    d_sample = {item: i for i, item in enumerate(set(res['IID']))}
    d_cell = {item: i for i, item in enumerate(set(res['cell']))}
    X = np.zeros([len(d_sample), len(d_cell), len(d_type2) * len(d_sim_cutoff)])
    datasets = res.to_dict('records')
    for item in tqdm(datasets):
        x = d_sample[item['IID']]
        y = 10 * d_sim_cutoff[item['sim_cutoff']] + d_type2[item['type2']]
        z = d_cell[item['cell']]
        X[x, z, y] = item['SCORE1_AVG']
    return X, d_sample, d_cell, d_type2, d_sim_cutoff


def step2_process_atac_data(data_path, k):
    summary = ann.read_h5ad(data_path)
    atac_embed = np.concatenate([summary.obsm['5kATAC_lsi']], 1)
    d_barcode = {i: item for i, item in enumerate(summary.obs_names)}
    neighbor = kneighbors_graph(atac_embed, k, mode='distance', metric='euclidean', include_self=False)
    neighbor = neighbor.toarray().T
    x, y = np.nonzero(neighbor)
    edge = pd.DataFrame([x, y]).T
    edge[0] = edge[0].map(d_barcode)
    edge[1] = edge[1].map(d_barcode)
    edge[2] = 1
    return edge


def step3_process_covar_bulkprs_data(covar_data_path, bulk_prs_data_path):
    pcs = pd.read_csv(covar_data_path, sep=' ', header=None).toarray()
    d_pcs = {}
    for i in range(len(pcs)):
        d_pcs[pcs[i, 0]] = pcs[i, 1:]

    d_prs = {}
    for i, item in enumerate([x for x in os.listdir(f'./{bulk_prs_data_path}/prs') if 'sscore' in x if
                              '0.2.ss' not in x and '0.3.ss' not in x and '0.4.ss' not in x]):
        data = pd.read_csv(f'./{bulk_prs_data_path}/prs/' + item, sep='\t')
        for item2 in data.to_dict('records'):
            if item2['IID'] not in d_prs:
                d_prs[item2['IID']] = np.zeros(21)
            d_prs[item2['IID']][i] = item2['SCORE1_AVG']
    return d_pcs, d_prs


def step4_merge_data(ATAC_data,d_pcs, d_prs, edge, label_data_path, save_path):
    X_ATAC, d_sample_ATAC, d_cell_ATAC, d_type2_ATAC, d_sim_cutoff_ATAC = ATAC_data

    label = pd.read_csv(label_data_path)
    d_id2label = {item['IID']: item['label'] for item in label.to_dict('records')}
    sample_id = list(d_sample_ATAC.keys())

    select_cell = ([y for x, y in d_cell_ATAC.items()])
    d_cell_inv = {v: k for k, v in d_cell_ATAC.items()}
    X_ATAC = X_ATAC[:, [d_cell_ATAC[d_cell_inv[x]] for x in select_cell], :]
    X_ATAC = X_ATAC[[d_sample_ATAC[x] for x in sample_id]]
    used_prs = np.array([d_prs[x] for x in sample_id])
    used_pcs = np.array([d_pcs[x] for x in sample_id])
    features = []
    for item in ['0.1', '0.05', '0.01', '0.5', '1e-3', '1e-4', '1e-5']:
        for item2 in [0.1, 0.3, 0.5]:
            features.append(d_type2_ATAC[item] + d_sim_cutoff_ATAC[item2] * 10)
    X_ATAC = X_ATAC[:, :, features]
    y = np.array([d_id2label[x] for x in sample_id])
    X = np.concatenate([X_ATAC], 2)
    summary = sc.read_h5ad('.//cell_81501_HT_summary.h5ad')
    d_celltype = {item['cell_id']: item['celltype'] for item in summary.obs.reset_index().to_dict('records')}
    celltype = [d_celltype[d_cell_inv[x]] for x in select_cell]
    cell_barcode = [d_cell_inv[x] for x in select_cell]
    meta = pd.DataFrame([celltype, cell_barcode]).T
    meta = meta.rename(columns={0: 'celltype', 1: 'barcode'})

    d_barcode = {item: i for i, item in enumerate(meta['barcode'])}

    edge['x'] = edge[0].map(d_barcode)
    edge['y'] = edge[1].map(d_barcode)
    edge['celltype_0'] = edge[0].map(d_celltype)
    edge['celltype_1'] = edge[1].map(d_celltype)
    d_sum = {item[1]: item[2] for item in edge.groupby(1).sum().reset_index().to_dict('records')}
    edge[3] = edge[1].map(d_sum)
    edge[2] = edge[2] / edge[3]
    L = (np.array([edge['x'], edge['y']]).T)
    pkl.dump([X, y, used_prs, used_pcs, L, np.array(edge[2]), meta[['celltype', 'barcode']], sample_id], open(save_path,
                                                                                                              'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prs_path", type=str)
    parser.add_argument("--atac_path", type=str)
    parser.add_argument("--covar_data_path", type=str)
    parser.add_argument("--bulk_prs_data_path", type=str)
    parser.add_argument("--label_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--k", type=int, default=25)
    args = parser.parse_args()

    ATAC_data = step1_process_prs_data(args.prs_path)
    edge = step2_process_atac_data(args.atac_path,args.k)
    d_pcs,d_prs = step3_process_covar_bulkprs_data(args.covar_data_path,args.bulk_prs_data_path)
    step4_merge_data(ATAC_data,d_pcs, d_prs, edge, args.label_data_path, args.save_path)
