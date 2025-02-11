# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:dataset.py
# @Software:PyCharm
# @Created Time:2023/12/28 5:56 PM
import hdf5plugin
import anndata
from numpy import float32
import os,sys
from typing import Dict, Iterable, List, Optional, Tuple, Union
import scanpy as sc
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import scvi
import lmdb
import dgl
import torch
import os.path as osp
import json
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse
from tqdm import tqdm
import random
import copy
sys.path.insert(0, "../")
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch  #,random_mask_value
from .dataset import cell_annotation_test_dataset

def Pretraining_dataloader(dataset,args,shuffle,drop_last,num_workers,**kwargs):
    # if args.distributed:
    #     sampler=DistributedSampler(dataset)
    # else:
    #     sampler=SequentialSampler(dataset)
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
    )
    return data_loader

def seed_all(seed_value, cuda_deterministic=False):
    """
    设置所有的随机种子
    """
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def Load_Data(data_path,args,**kwargs):

    if args.task=='Cell_annotation_test':
        # return test_dataset(data_path,args=args, **kwargs)
        return cell_annotation_test_dataset(testdata_path=data_path,args=args,only_test=True,**kwargs)
    if args.task=='Cell_annotation_inference':
        return Inference_dataset(data_path,args=args, **kwargs)
    # if args.task=='Cell_annotation_testonly':
    #     return Inference_dataset(data_path,args=args, **kwargs)


def Inference_dataset(data_path,args, **kwargs):
    if args.data_name=="scplantdb_root506007":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "PRJNA506007.h5ad")
        # adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False

        adata.var["gene_name"] = adata.var.index.tolist()
        celltype_id_labels = adata.obs["Celltype"].astype("category").cat.codes.values
        celltypes=['Columella root cap',  'G1/G0 phase', 'Lateral root cap',  'Non-hair',  'Phloem', 'Root cap','Root cortex',  'Root endodermis', 'Root hair',  'Root stele', 'S phase', 'Unknow', 'Xylem']
        num_types = 13
        id2type={0: 'Columella root cap', 1: 'G1/G0 phase', 2: 'Lateral root cap', 3: 'Non-hair', 4: 'Phloem', 5: 'Root cap', 6: 'Root cortex', 7: 'Root endodermis', 8: 'Root hair', 9: 'Root stele', 10: 'S phase', 11: 'Unknow', 12: 'Xylem'}
        # adata.obs["celltype_id"] = celltype_id_labels
        adata = adata.copy()
        adata_raw=adata.copy()
        adata.obs["batch_id"] = 0
    if args.data_name=="scplantdb_root471914":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "PRJNA471914.h5ad")
        # adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        # celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        # print('celltype_id_labels',celltype_id_labels)
        celltypes1=[ 'G2/M phase'  'Root stele']
        celltypes=['Columella root cap',  'G1/G0 phase', 'Lateral root cap',  'Non-hair',  'Phloem', 'Root cap','Root cortex',  'Root endodermis', 'Root hair',  'G2/M phase', 'S phase', 'Unknow', 'Xylem']
        num_types = 13
        id2type={0: 'Columella root cap', 1: 'G1/G0 phase', 2: 'Lateral root cap', 3: 'Non-hair', 4: 'Phloem', 5: 'Root cap', 6: 'Root cortex', 7: 'Root endodermis', 8: 'Root hair', 9: 'G2/M phase', 10: 'S phase', 11: 'Unknow', 12: 'Xylem'}
        # id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        # adata.obs["celltype_id"] = celltype_id_labels
        adata = adata.copy()
        adata_raw=adata.copy()
        adata.obs["batch_id"] = 0    
    
    logger=kwargs['logger']
    vocab=kwargs['vocab']
    is_master=kwargs['is_master']
    mask_value=kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master,logger=logger)  # only retain the gene that appears in vocab

    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  #  false step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)

    test_data_pt,num_batch_types=prepare_cell_data_inference(adata=adata,args=args,vocab=vocab,is_master=is_master,mask_value=mask_value,pad_value=pad_value,logger=logger,pad_token=pad_token)

    return test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_raw










def test_dataset(data_path,args, **kwargs):
    if args.data_name=="scplantdb_root497883":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "PRJNA497883.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        print('celltype_id_labels',celltype_id_labels)
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        adata = adata.copy()
        adata_raw=adata.copy()
        adata.obs["batch_id"] = 1
    if args.data_name=="scplantdb_root471914":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "PRJNA471914.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        celltypes1=[ 'G2/M phase'  'Root stele']
        celltypes=['Columella root cap', 'G1/G0 phase', 'Lateral root cap', 'Non-hair', 'Phloem', 'Root cap','Root cortex',  'Root endodermis', 'Root hair',  'G2/M phase', 'S phase', 'Unknow', 'Xylem']
        num_types = 13
        id2type={0: 'Columella root cap', 1: 'G1/G0 phase', 2: 'Lateral root cap', 3: 'Non-hair', 4: 'Phloem', 5: 'Root cap', 6: 'Root cortex', 7: 'Root endodermis', 8: 'Root hair', 9: 'G2/M phase', 10: 'S phase', 11: 'Unknow', 12: 'Xylem'}
       
        # celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        # celltypes = adata.obs["celltype"].unique()
        # num_types = len(np.unique(celltype_id_labels))
        # id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        adata = adata.copy()
        adata_raw=adata.copy()
        adata.obs["batch_id"] = 1
    if args.data_name=="scplantdb_root506007":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "PRJNA506007.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        print('celltypes\t',celltypes)
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        # print('id2type',id2type)
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        adata = adata.copy()
        adata_raw=adata.copy()
        adata.obs["batch_id"] = 0
    logger=kwargs['logger']
    vocab=kwargs['vocab']
    is_master=kwargs['is_master']
    mask_value=kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master,logger=logger)  # only retain the gene that appears in vocab

    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  #  false step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)

    test_data_pt,num_batch_types=prepare_cell_data(adata=adata,args=args,vocab=vocab,is_master=is_master,mask_value=mask_value,pad_value=pad_value,logger=logger,pad_token=pad_token)

    return test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_raw



    

  
def random_mask_value1(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()
    row = values
    non_zero_idx = np.nonzero(row)[0]
    # print('non_zero_idx',non_zero_idx,len(non_zero_idx))
    n_mask = int(len(non_zero_idx) * mask_ratio)
    # print('n_mask',n_mask)
    mask_idx = np.random.choice(non_zero_idx, n_mask, replace=False)
    # print('mask_idx',mask_idx,len(mask_idx))
    row[mask_idx] = mask_value
    # print('row',row,row.shape)
    return row



def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    print('mask_ratio',mask_ratio)
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()
    # print('values',values)
    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        # print('np.nonzero(row - pad_value)',np.nonzero(row - pad_value))
        n_mask = int(len(non_padding_idx) * mask_ratio)
        # print('n_mask',n_mask)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        # print('mask_idx',mask_idx,len(mask_idx))
        row[mask_idx] = mask_value
        # print('row',row)
    return values   
    # row = values


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def filter_gene(vocab,adata,is_master,logger):
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var_names.tolist()
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    if is_master:
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )   
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    return adata,gene_ids_in_vocab


def prepare_cell_data_inference(adata,args,vocab,is_master,mask_value,pad_value,logger,pad_token='<pad>'):
    '''
    Args:
        adata: adata used for training
        adata_test: adata used for testing
        args:
        vocab:
        is_master: does the current GPU act as the master
        mask_value: specify certain values used as mask value (default: -1)
        pad_value: specify certain values used as padding value (default: -2)
        logger:
        sort_seq_batch:
        pad_token:
    Returns:
    '''
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]
   #选择的X_binned
    batch_ids = adata.obs["batch_id"].tolist()
    print('Total set number of inference:',len(batch_ids))
    num_batch_types = len(set(batch_ids))
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    if adata is not None:
        all_counts_test = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        # celltypes_labels_test = adata.obs["celltype_id"].tolist()  # make sure count from 0
        # celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        tokenized_test = tokenize_and_pad_batch(
            all_counts_test,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=args.append_cls,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
            # graph=graph
        )
        input_values_test = torch.from_numpy(random_mask_value(
            tokenized_test["values"],
            mask_ratio=args.mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )).float()
        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids_test).long(),
            # "celltype_labels": torch.from_numpy(celltypes_labels_test).long(),
            # "sorted_layer_idx": tokenized_test["sorted_layer_idx"]
        }
    else:
        print('数据为空')
        test_data_pt=None
        # celltypes_labels_test=None

    if is_master:
        print(
            f"Ratio of masked values in train: ",
            f"{(input_values_test == mask_value).sum() / (input_values_test - pad_value).count_nonzero():.4f}",
        )

    
    return test_data_pt,num_batch_types



def prepare_cell_data(adata,args,vocab,is_master,mask_value,pad_value,logger,pad_token='<pad>'):
    '''
    Args:
        adata: adata used for training
        adata_test: adata used for testing
        args:
        vocab:
        is_master: does the current GPU act as the master
        mask_value: specify certain values used as mask value (default: -1)
        pad_value: specify certain values used as padding value (default: -2)
        logger:
        sort_seq_batch:
        pad_token:
    Returns:
    '''
    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]
   #选择的X_binned
    batch_ids = adata.obs["batch_id"].tolist()
    print('Total set number of test:',len(batch_ids))
    num_batch_types = len(set(batch_ids))
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    if adata is not None:
        all_counts_test = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        celltypes_labels_test = adata.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        tokenized_test = tokenize_and_pad_batch(
            all_counts_test,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=args.append_cls,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
            # graph=graph
        )
        input_values_test = torch.from_numpy(random_mask_value(
            tokenized_test["values"],
            mask_ratio=args.mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )).float()
        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids_test).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels_test).long(),
            # "sorted_layer_idx": tokenized_test["sorted_layer_idx"]
        }
    else:
        test_data_pt=None
        celltypes_labels_test=None

    if is_master:
        print(
            f"Ratio of masked values in train: ",
            f"{(input_values_test == mask_value).sum() / (input_values_test - pad_value).count_nonzero():.4f}",
        )

    
    return test_data_pt,num_batch_types