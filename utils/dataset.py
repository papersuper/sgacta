# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:dataset.py
# @Software:PyCharm
# @Created Time:2023/12/28 5:56 PM
import hdf5plugin
import anndata
from anndata import AnnData
from scgpt import logger
import pandas as pd
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
from .preprocess_bin import Preprocessor_only_bin,Preprocessor_annimal_normalized
sys.path.insert(0, "../")
from scgpt.preprocess import Preprocessor
# from scgpt import Preprocessor_only_bin
from scgpt.tokenizer import tokenize_and_pad_batch  #,random_mask_value
 

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
def lmdbanno_dataloader(dataset,args,shuffle,drop_last,num_workers,**kwargs):
    # sampler = SequentialSampler(dataset)
    if args.distributed:
        sampler=DistributedSampler(dataset)
    else:
        # sampler=SequentialSampler(dataset)
        sampler=RandomSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size if 'batch_size' in args else None,
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

def Load_Data(data_path,testdata_path,args,**kwargs):

    if args.task=='Cell_annotation':
        if args.lmdb_celldata:
            train_path=osp.join(args.data_path,'train.db')
            valid_path=osp.join(args.data_path,'val.db')
            # test_path=osp.join(args.data_path,'test.db')
            train_data_pt = lmdb_cell_annotation_datapoint(db_path=train_path,bin_num=args.n_bins,lmdb_type='train', args=args,**kwargs)
            valid_data_pt = lmdb_cell_annotation_datapoint(db_path=valid_path,bin_num=args.n_bins,lmdb_type='valid', args=args,**kwargs)
            # test_data_pt = lmdb_cell_annotation_datapoint(db_path=test_path,bin_num=args.n_bins,lmdb_type='test', args=args,**kwargs)
            # print('\n\ntrain_data_pt',train_data_pt[2])
            num_batch_types = 2
            celltypes,id2type,num_types = lmdb_cellanno_otherinf(args=args)
            # adata_test_raw = cell_annotation_test_dataset(testdata_path=testdata_path,args=args,**kwargs)
            test_data_pt,adata_test_raw = cell_annotation_test_dataset(testdata_path=testdata_path,args=args,only_test=False,**kwargs)
            # test_data_pt,adata_test_raw = cell_annotation_test_dataset_nobin(testdata_path=testdata_path,args=args,only_test=False,**kwargs)
            
            test_len = len(test_data_pt)
            print(f"Test path: {testdata_path}")
            print(f"Test data length: {test_len}")
            return train_data_pt, valid_data_pt, test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_test_raw
        
        else:
            # data_point=cell_annotation_dataset(data_path=data_path,args=args,**kwargs)
            # print('data_point',data_point)
            return cell_annotation_dataset(data_path=data_path,args=args,**kwargs)
    
    elif args.task=='Cell_annotation_test':
        return cell_annotation_test_dataset(testdata_path=testdata_path,args=args,only_test=True,**kwargs)
    elif args.task=='Cell_annotation_testonly':
        if args.lmdb_celldata:
            print('-------------使用lmdb------------------')
            train_path=osp.join(args.data_path,'train.db')
            train_data_pt = lmdb_cell_annotation_datapoint(db_path=train_path,bin_num=args.n_bins,lmdb_type='train', args=args,**kwargs)
            num_batch_types = 2
            celltypes,id2type,num_types = lmdb_cellanno_otherinf(args=args)
            return train_data_pt,num_batch_types,celltypes,id2type,num_types
        else: 
            print('-------------使用h5ad------------------')
            test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_test_raw = cell_annotation_test_dataset(testdata_path=data_path,args=args,only_test=True,**kwargs)
            return test_data_pt,num_batch_types,celltypes,id2type,num_types
    elif args.task=='Integration':
        return Integration_dataset(data_path=data_path,args=args,**kwargs)
    elif args.task=='Pretraining':
        return Pretraining_dataset(data_path=data_path,args=args,**kwargs)
    elif args.task=='GRN_inference':
        return GRN_dataset(data_path=data_path, args=args, **kwargs)




def GRN_dataset(data_path,args,**kwargs):
    if args.data_name=='adamson':
        from gears import PertData
        data_dir = Path(data_path)
        pert_data = PertData(data_dir)
        pert_data.load(data_name="adamson")
        adata = sc.read(data_dir / "adamson/perturb_processed.h5ad")
        ori_batch_col = "control"
        adata.obs["celltype"] = adata.obs["condition"].astype("category")
        adata.obs["str_batch"] = adata.obs["control"].astype(str)
        data_is_raw = False
        filter_gene_by_counts=3
    else:
        raise ValueError(f'Invalid dataset{args.data_name} for task {args.task}')
    return adata,data_is_raw,ori_batch_col,filter_gene_by_counts




'''
使用lmdb加载注释的celldata
'''
class lmdb_cell_annotation_datapoint(Dataset):
    def __init__(self, db_path,n_bins=51,args=None,pad_value=-2,lmdb_type=None,**kwargs):
    # def __init__(self, db_path, n_bins=51, args=None, pad_value=-2, mask_ratio=0.15,lmdb_type='train',data_name='root',
                #  append_cls=False, include_zero_gene=False, max_seq_len=512, normalize_total=1e4, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.lmdb_type=lmdb_type
        self.args = args
        self.data_name=args.data_name
        self.invalid_datapoint_count = 0
        self.pad_value = pad_value
        self.n_bins = n_bins
        self.mask_ratio = args.mask_ratio
        self.append_cls = args.append_cls
        self.include_zero_gene = args.include_zero_gene
        self.max_seq_len = args.max_seq_len
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'__len__').decode("utf-8"))




    def _pad_and_mask(self, values, gene_ids,return_pt: bool = True):
        if self.include_zero_gene:
            values = values
            gene_ids = gene_ids
        else:
            idx = np.nonzero(values)[-1]
            # print('idx',idx)
            # print(' np.nonzero(values)', np.nonzero(values))
            values = np.array(values[idx])
            # print('values',values)
            gene_ids = np.array(gene_ids[idx])
            # print('gene_ids', gene_ids)

        if len(gene_ids) > self.max_seq_len:
            # if not self.append_cls:
            #     idx=np.random.choice(len(gene_ids), self.max_seq_len, replace=False)
            # else:
            #     idx = np.random.choice(len(gene_ids) - 1, self.max_len - 1, replace=False)
            #     idx=idx+1
            #     idx=np.insert(idx,0,0)
            idx = np.random.choice(len(gene_ids), self.max_seq_len, replace=False)
            gene_ids = gene_ids[idx]
            values = values[idx]
            # masked_values=masked_values[idx]
        # print('value_pretrain',values,len(values))
        masked_values = random_mask_value1(values, self.mask_ratio)

        ## padding
        if len(gene_ids) < self.max_seq_len:
            pad_id = self.vocab['<pad>']
            gene_ids = np.concatenate(
                [gene_ids, np.full(self.max_seq_len - len(gene_ids), pad_id, dtype=gene_ids.dtype)])
            values = np.concatenate(
                [values, np.full(self.max_seq_len - len(values), self.pad_value, dtype=values.dtype)])
            masked_values = np.concatenate(
                [masked_values, np.full(self.max_seq_len - len(masked_values), self.pad_value, dtype=masked_values.dtype)])

        if self.append_cls:
            values = np.insert(values, 0, 0)
            gene_ids = np.insert(gene_ids, 0, self.vocab['<cls>'])



        return torch.tensor(values).float(), torch.tensor(gene_ids).int(), torch.tensor(masked_values).float()
            # masked_values).float(), sorted_gene_ids, masked_sorted_gene_ids, sorted_layer_idx
    def __getitem__(self, index):
        # values = self.txn.get(u'{}'.format(index).encode())
        # expression_value = self.txn.get(u'{}'.format(index).encode('ascii'))#, dtype=np.float16
        # celltype_labels = self.txn.get(u'{}_celltype'.format(index).encode('ascii')).decode('utf-8')
        expression_key = u'{}'.format(index).encode('ascii')
        celltype_key = u'{}_celltype'.format(index).encode('ascii')

        expression_data = self.txn.get(expression_key)
        celltype = self.txn.get(celltype_key)
        # if expression_data is None:
        #     print('\nexpression_data is none\n')
        expression_data = self.txn.get(expression_key)
        # print('----------------expression_data--------------',expression_data,
        # '\n----------------celltype_key---------------------',celltype_key)
        expression_value = np.frombuffer(expression_data, dtype=np.float16)
        # expression_value = expression_value1[:33322]
        # try:
        #     expression_value = np.frombuffer(expression_data, dtype=np.float16)
        # except TypeError as e:
        #     print(f"Error converting expression_data to numpy array: {e}")
        #     print(f"expression_data: {expression_data[0:11]}")
        celltype_labels = celltype.decode('utf-8')
        # print('\n\ncelltype_labels',celltype_labels[0:2])
        # try:
        # expression_value = np.frombuffer(expression_value, dtype=np.float16)  # np.array([gene_num,])
        # print('\n\nexpression_value', len(expression_value))

        gene_ids = np.array(range(0, len(expression_value)))  # 同时gene——id也是32401长度，即每个value的索引为gene_id
        # print("len(gene_ids)",len(gene_ids))
        # celltype_labels = celltype_labels.decode('utf-8')
        # except Exception as e:
        #     print("error:", e)

        assert len(expression_value) == len(gene_ids)

        if self.data_name == 'root':
            #scplantdb
            # celltypes = ['G2/M phase', 'Root cortex', 'Root cap', 'S phase', 'Root hair', 
            # 'Lateral root cap', 'Non-hair', 'Columella root cap', 'Phloem', 'Root endodermis', 
            # 'Xylem', 'G1/G0 phase', 'Xylem pole pericycle', 'Root procambium', 'Phloem pole pericycle', 
            # 'Companion cell', 'Metaxylem', 'Sieve element', 'Protoxylem', 'Root stele', 'Pericycle', 
            # 'Meristematic cell', 'Stem cell niche', 'Root epidermis', 'Phloem/Pericycle', 
            # 'Phloem parenchyma', 'Lateral root primordia', 'Unknow']
            celltypes = [
            "Root hair", "Pericycle", "Lateral root cap", "Non-hair", "Root procambium",
            "Root endodermis", "Root cortex", "Xylem", "Columella", "Phloem",
            "Quiescent Center", "Unknow", "Phloem pole pericycle", "Xylem pole",
            "Root epidermis", "Root stele", "Xylem pole pericycle",
            "Lateral root endodermis", "Lateral root primordia", "G2/M phase",
            "Root cap", "S phase", "Columella root cap", "G1/G0 phase",
            "Companion cell", "Metaxylem", "Sieve element", "Protoxylem",
            "Meristematic cell", "Stem cell niche", "Phloem/Pericycle",
            "Phloem parenchyma"
            ]
            # adata.obs["celltype"] = adata.obs["celltype"].astype("category")
            data_is_raw = False
            filter_gene_by_counts = False
            num_types = len(np.unique(celltypes))
            id2type = dict(enumerate(celltypes))
            celltype_id = self.celltype_to_id(id2type, celltype_labels)
        elif self.data_name == 'root_152766':

            celltypes = [
            "Phloem pole pericycle", "Root cortex", "Unknow", "Root hair", "Columella", 
            "Non-hair", "Xylem pole", "Root endodermis", "Quiescent Center", "Phloem", "Xylem"
            ]

            data_is_raw = False
            filter_gene_by_counts = False
            num_types = len(np.unique(celltypes))
            id2type = dict(enumerate(celltypes))
            celltype_id = self.celltype_to_id(id2type, celltype_labels)
        elif self.data_name == 'leaf':
            #scplantdb
            # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
            # 'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
            # 'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
            # 'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
            # 'Leaf epidermis', 'Stress response', 'Meristematic cell']  
            celltypes = [
            "Leaf epidermis", "Palisade mesophyll cell", "Spongy mesophyll cell", 
            "Vascular tissue", "Leaf guard cell", "Mesophyll", "Phloem parenchyma", 
            "Bundle sheath", "Companion cell", "Hydathodes", "Unknow", "Xylem", 
            "Leaf pavement cell", "S phase", "Shoot system epidermis", "Shoot apical meristem", 
            "G2/M phase", "Sieve element", "Phloem", "Stress response", "Meristematic cell"
            ]

            data_is_raw = False
            filter_gene_by_counts = False
            num_types = len(np.unique(celltypes))
            id2type = dict(enumerate(celltypes))
            celltype_id = self.celltype_to_id(id2type, celltype_labels)
        elif self.data_name == 'flower':
            celltypes=['Shoot system epidermis', 'Cortex', 'G2/M phase', 'Vascular cambium', 
            'Flower meristem', 'S phase', 'Unknow', 'Mesophyll', 'Xylem', 'Phloem', 
            'Vegetative nuclei', 'Microspore nuclei', 'Sperm nuclei', 'Generative nuclei', 
            'Contaminating nuclei', 'Transitory']
            data_is_raw = False
            filter_gene_by_counts = False
            num_types = len(np.unique(celltypes))
            id2type = dict(enumerate(celltypes))
            celltype_id = self.celltype_to_id(id2type, celltype_labels)
        elif self.data_name == 'cotyledons':
            celltypes=['Spongy mesophyll', 'Palisade mesophyll', 'Leaf guard cell', 
            'Stress response', 'Leaf pavement cell', 'Phloem', 'Xylem', 'Bundle sheath', 
            'S phase', 'Companion cell', 'Phloem parenchyma', 'Unknow']
            data_is_raw = False
            filter_gene_by_counts = False
            num_types = len(np.unique(celltypes))
            id2type = dict(enumerate(celltypes))
            celltype_id = self.celltype_to_id(id2type, celltype_labels)
        elif self.data_name == 'seed':
            celltypes=[
            "Seed.coat", "Cotyledons", "Embryo", "Unknown", "Chalazal.seed.coat._CZSC_",
            "Inner.integument._ii_", "Cotyledons._abaxial.side_", "Provasculature,.QC",
            "Outer.integument._oi_", "Endothelium._ii1_", "Endosperm._MCE_", "Vascular.tissue",
            "Endosperm", "Endosperm._PEN_", "Cotyledon.tip", "Chalazal.endosperm._CZE_",
            "Cotyledon.tip.and.vascular.initials", "Cotyledons.epidermis", "Provascular.tissue",
            "Seed.coat.epidermis", "Central.cell", "Chalazal.seed.coat._and.vasculature_",
            "Chalazal.region._chalazal.seed.coat,vascular_", "Procambium,.Stele,.RAM",
            "Unknown_Embryo._SAM_.Seed.Coat", "Cotyledons._cells.around.SAM_", "Endosperm_MCE_",
            "Endosperm_PEN_", "Sperm.cell.and.vegetative.cell", "Chalazal.region",
            "Synergid.cell", "Suspensor", "Egg.cell", "Zygote,.Basal.cell"
            ]
            data_is_raw = False
            filter_gene_by_counts = False
            num_types = len(np.unique(celltypes))
            id2type = dict(enumerate(celltypes))
            celltype_id = self.celltype_to_id(id2type, celltype_labels)

        else:
            print('请在选择范围内给出data_name')

        '''归一化处理'''
        # normalize_total = 1e4
        # if expression_value.sum() == 0:
        #     print("expression_values.sum()=0")
        # normalized_values = expression_value / expression_value.sum() * normalize_total
        # print('expression_values_normalize', normalized_values[:11],len(normalized_values))
        # # Log1p 转换
        # # 避免对值为0的数据应用 log1p，可以添加一个小的常数以避免负无穷大
        # epsilon = 1e-10
        # log_values = np.log1p(normalized_values + epsilon)
        # log2_values = log_values / np.log(2)
        # print('expression_values_log',log_values[:11],len(log_values))
        # binned_values, bin_edge = self._binning(log2_values)

        '''bin'''
        binned_values, bin_edge = self._binning(expression_value)
        # print('binned_values', binned_values[:11])

        # values, gene_ids,masked_values = self._pad_and_mask(
        #     binned_values, gene_ids=gene_ids)

        values, gene_ids,masked_values = self._pad_and_mask(
            binned_values, gene_ids=gene_ids)

        # , masked_values, sorted_gene_ids, masked_sorted_gene_ids, sorted_layer_idx
        if self.lmdb_type=='train':
            batch_labels = 0
        if self.lmdb_type == 'valid':
            batch_labels = 0
        if self.lmdb_type == 'test':
            batch_labels = 1
        datapoint={"gene_ids": gene_ids,
            "values": masked_values,
            "target_values": values,
            "batch_labels": batch_labels,
            "celltype_labels": celltype_id
            }
        # datapoint = {"gene_ids": gene_ids, "masked_values": masked_values, "target_values": values,
        #              "sorted_gene_ids": sorted_gene_ids, "masked_sorted_gene_ids": masked_sorted_gene_ids,
        #              "sorted_layer_idx": sorted_layer_idx}
        # print('datapoint',datapoint)
        # cell_id_num=[celltypes, id2type, num_types]
        # print('cell_id_num',cell_id_num)
        return datapoint

    def _binning(self, values):
        non_zero_ids = values.nonzero()
        # print('non_zero_ids',non_zero_ids)
        non_zero_row = values[non_zero_ids]
        # print('non_zero_row',non_zero_row)
        
        matrix_bins=np.array([0.01,0.1,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.8,1.9,2.0,
        2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,
        4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,6,6.5,7])
        # bins=matrix_bins
        
        bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
        

        # print('bins',bins)
        # bins = np.sort(np.unique(bins))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        # print('non_zero_row',non_zero_row,'\n','bins',bins)
        non_zero_digits = self._digitize(non_zero_row, bins)
        # assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= self.n_bins - 1
        binned_row = np.zeros_like(values, dtype=np.int64).copy()
        binned_row[non_zero_ids] = non_zero_digits
        bin_edge = np.concatenate([[0], bins])
        return binned_row, bin_edge

    # @staticmethod
    def _digitize(self, x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.
        side (:class:`str`, optional):
            The side to use for digitization. If "one", the left side is used. If
            "both", the left and right side are used. Default to "one".

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        if side == "one":
            return left_digits

        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

        # 分箱处理

    def __len__(self):
        return self.length

    def celltype_to_id(self, id2type, celltype_label):
        # Assuming id2type is a dictionary mapping cell type strings to integers
        type2id = {v: k for k, v in id2type.items()}
        return type2id.get(celltype_label, -1)
        # set up the preproces

def lmdb_cellanno_otherinf(args):
    if args.data_name == 'root':
        '''scplantdb'''
        # celltypes = ['G2/M phase', 'Root cortex', 'Root cap', 'S phase', 'Root hair', 
        # 'Lateral root cap', 'Non-hair', 'Columella root cap', 'Phloem', 'Root endodermis', 
        # 'Xylem', 'G1/G0 phase', 'Xylem pole pericycle', 'Root procambium', 'Phloem pole pericycle', 
        # 'Companion cell', 'Metaxylem', 'Sieve element', 'Protoxylem', 'Root stele', 'Pericycle', 
        # 'Meristematic cell', 'Stem cell niche', 'Root epidermis', 'Phloem/Pericycle', 
        # 'Phloem parenchyma', 'Lateral root primordia', 'Unknow']
        # adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        '''all'''
        celltypes = [
        "Root hair", "Pericycle", "Lateral root cap", "Non-hair", "Root procambium",
        "Root endodermis", "Root cortex", "Xylem", "Columella", "Phloem",
        "Quiescent Center", "Unknow", "Phloem pole pericycle", "Xylem pole",
        "Root epidermis", "Root stele", "Xylem pole pericycle",
        "Lateral root endodermis", "Lateral root primordia", "G2/M phase",
        "Root cap", "S phase", "Columella root cap", "G1/G0 phase",
        "Companion cell", "Metaxylem", "Sieve element", "Protoxylem",
        "Meristematic cell", "Stem cell niche", "Phloem/Pericycle",
        "Phloem parenchyma"
        ]
        data_is_raw = False
        filter_gene_by_counts = False
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
    elif args.data_name == 'root_152766':
        celltypes = [
        "Phloem pole pericycle", "Root cortex", "Unknow", "Root hair", "Columella", 
        "Non-hair", "Xylem pole", "Root endodermis", "Quiescent Center", "Phloem", "Xylem"
        ]
        data_is_raw = False
        filter_gene_by_counts = False
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
    elif args.data_name == 'leaf':
        # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
        # 'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
        # 'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
        # 'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
        # 'Leaf epidermis', 'Stress response', 'Meristematic cell']
        celltypes = [
        "Leaf epidermis", "Palisade mesophyll cell", "Spongy mesophyll cell", 
        "Vascular tissue", "Leaf guard cell", "Mesophyll", "Phloem parenchyma", 
        "Bundle sheath", "Companion cell", "Hydathodes", "Unknow", "Xylem", 
        "Leaf pavement cell", "S phase", "Shoot system epidermis", "Shoot apical meristem", 
        "G2/M phase", "Sieve element", "Phloem", "Stress response", "Meristematic cell"
        ]
        data_is_raw = False
        filter_gene_by_counts = False
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
    elif args.data_name == 'flower':
        celltypes=['Shoot system epidermis', 'Cortex', 'G2/M phase', 'Vascular cambium', 
        'Flower meristem', 'S phase', 'Unknow', 'Mesophyll', 'Xylem', 'Phloem', 
        'Vegetative nuclei', 'Microspore nuclei', 'Sperm nuclei', 'Generative nuclei', 
        'Contaminating nuclei', 'Transitory']
        data_is_raw = False
        filter_gene_by_counts = False
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
    elif args.data_name == 'cotyledons':
        celltypes=['Spongy mesophyll', 'Palisade mesophyll', 'Leaf guard cell', 
        'Stress response', 'Leaf pavement cell', 'Phloem', 'Xylem', 'Bundle sheath', 
        'S phase', 'Companion cell', 'Phloem parenchyma', 'Unknow']
        data_is_raw = False
        filter_gene_by_counts = False
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
    elif args.data_name == 'seed':
        celltypes=["Seed.coat", "Cotyledons", "Embryo", "Unknown", "Chalazal.seed.coat._CZSC_",
        "Inner.integument._ii_", "Cotyledons._abaxial.side_", "Provasculature,.QC",
        "Outer.integument._oi_", "Endothelium._ii1_", "Endosperm._MCE_", "Vascular.tissue",
        "Endosperm", "Endosperm._PEN_", "Cotyledon.tip", "Chalazal.endosperm._CZE_",
        "Cotyledon.tip.and.vascular.initials", "Cotyledons.epidermis", "Provascular.tissue",
        "Seed.coat.epidermis", "Central.cell", "Chalazal.seed.coat._and.vasculature_",
        "Chalazal.region._chalazal.seed.coat,vascular_", "Procambium,.Stele,.RAM",
        "Unknown_Embryo._SAM_.Seed.Coat", "Cotyledons._cells.around.SAM_", "Endosperm_MCE_",
        "Endosperm_PEN_", "Sperm.cell.and.vegetative.cell", "Chalazal.region",
        "Synergid.cell", "Suspensor", "Egg.cell", "Zygote,.Basal.cell"]
        data_is_raw = False
        filter_gene_by_counts = False
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))

    else:
        print('请在选择范围内给出data_name')

    return celltypes, id2type, num_types




'''
不使用lmdb加载注释的celldata
'''
def cell_annotation_dataset(data_path,args,**kwargs):
    if args.data_name == "ms":
        data_dir = Path(data_path)
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype(
            "category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

        # make the batch category column
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()

        adata_test = adata[adata.obs["str_batch"] == "1"]
        adata = adata[adata.obs["str_batch"] == "0"]
    elif args.data_name == "mye":
        data_dir = Path(data_path)
        adata = sc.read(data_dir / "reference_adata.h5ad")
        adata_test = sc.read(data_dir / "query_adata.h5ad")
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"] = adata.obs["batch"].cat.codes.values
        adata_test.obs["batch_id"] = adata_test.obs["batch"].cat.codes.values
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()

        # merge two data for assign idx to cell type
        adata_total = adata.concatenate(adata_test,
                                        batch_key="data_split")  # batch_key is used to differentiate these two dataset
        celltype_id_labels = adata_total.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata_total.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata_total.obs["celltype"].astype("category").cat.categories))
        adata_total.obs["celltype_id"] = celltype_id_labels
        adata_total.var["gene_name"] = adata_total.var.index.tolist()

        adata_test = adata_total[adata_total.obs["data_split"] == '1']
        adata = adata_total[adata_total.obs["data_split"] == '0']
    elif args.data_name=="pancreas":
        data_dir = Path(data_path)
        adata = sc.read(data_dir / "demo_train.h5ad")
        adata_test = sc.read(data_dir / "demo_test.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        adata.obs["batch_id"] =0
        adata_test.obs["batch_id"] =1
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()

        # merge two data for assign idx to cell type
        adata_total = adata.concatenate(adata_test,
                                        batch_key="data_split")  # batch_key is used to differentiate these two dataset
        celltype_id_labels = adata_total.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata_total.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata_total.obs["celltype"].astype("category").cat.categories))
        adata_total.obs["celltype_id"] = celltype_id_labels
        adata_total.var["gene_name"] = adata_total.var.index.tolist()

        adata_test = adata_total[adata_total.obs["data_split"] == '1']
        adata = adata_total[adata_total.obs["data_split"] == '0']
    elif args.data_name == "zheng68k":
        data_dir = Path(data_path)
        adata = sc.read(data_dir / "Zheng68K.h5ad")
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()

        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        if 'X_umap' not in adata.obsm:
            adata.obsm["X_umap"] = np.array(adata.obs.loc[:, ["TSNE.1", "TSNE.2"]])

        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()

        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1

    elif args.data_name=="plantphone":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "PlantPhoneDB_all.h5ad")
        adata.obs["celltype"] = adata.obs["labels"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()
        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
    elif args.data_name=="annimal":
        data_dir = Path(data_path)
        adata =  anndata.read(data_dir / "blood.h5ad")
        adata.obs["celltype"] = adata.obs["author_cell_type"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()
        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1  
    # elif args.data_name == "zheng68k":
    #     data_dir = Path(data_path)
    #     adata = anndata.read(data_dir /"sc_2_COPILOT.h5ad")  #Zheng68K     sc_2_COPILOT
    #     adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    #     data_is_raw = True
    #     filter_gene_by_counts = False
    #     adata.var["gene_name"] = adata.var.index.tolist()
    #     celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    #     celltypes = adata.obs["celltype"].unique()
    #     num_types = len(np.unique(celltype_id_labels))
    #     id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    #     adata.obs["celltype_id"] = celltype_id_labels
    #     train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
    #     print(train_idx[0],valid_idx[0])
    #     adata_test = adata[valid_idx].copy()
    #     adata = adata[train_idx].copy()
    #     adata_test_raw=adata_test.copy()
    #     adata.obs["batch_id"] = 0
    #     adata_test.obs["batch_id"] = 1
        # print(adata,adata.obs,adata.var,'/n/n/n',adata_test,adata_test.obs,adata_test.var)

    elif args.data_name == "cngb":
        data_dir = Path(data_path)
        adata = anndata.read(data_dir /"merged_cngb.h5ad")  
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()

        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()

        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
    elif args.data_name == "seed":
        data_dir = Path(data_path)
        adata = anndata.read(data_dir /"com_cellanno_seed.h5ad")  #dataset_part_3
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()
        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
    
    elif args.data_name == "pscth":
        data_dir = Path(data_path)
        adata = anndata.read(data_dir /"com_GSE158761_all_cells_labeled_with_celltype.h5ad")  
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()

        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()

        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
    elif args.data_name == "leaf":
        data_dir = Path(data_path)
        adata = anndata.read(data_dir /"raw_5_leaf.h5ad")  
        # adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()

# label方式一

        # celltype_id_labels = adata.obs["Celltype"].astype("category").cat.codes.values
        # celltypes = adata.obs["Celltype"].unique()
        # num_types = len(np.unique(celltype_id_labels))
        # id2type = dict(enumerate(adata.obs["Celltype"].astype("category").cat.categories))
        # adata.obs["celltype_id"] = celltype_id_labels

# label方式二

        # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
        # 'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
        # 'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
        # 'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
        # 'Leaf epidermis', 'Stress response', 'Meristematic cell']
        celltypes = [
        "Leaf epidermis", "Palisade mesophyll cell", "Spongy mesophyll cell", 
        "Vascular tissue", "Leaf guard cell", "Mesophyll", "Phloem parenchyma", 
        "Bundle sheath", "Companion cell", "Hydathodes", "Unknow", "Xylem", 
        "Leaf pavement cell", "S phase", "Shoot system epidermis", "Shoot apical meristem", 
        "G2/M phase", "Sieve element", "Phloem", "Stress response", "Meristematic cell"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
       
        adata.obs["celltype"]=adata.obs['Celltype']
        adata.obs["celltype"] = pd.Categorical(adata.obs["celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
 
 
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()
        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
    elif args.data_name == "root":
        data_dir = Path(data_path)
        adata = anndata.read(data_dir /"train_all.h5ad")  
        # adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()

#label方式一
        # celltype_id_labels = adata.obs["Celltype"].astype("category").cat.codes.values
        # celltypes = adata.obs["Celltype"].unique()
        # num_types = len(np.unique(celltype_id_labels))
        # id2type = dict(enumerate(adata.obs["Celltype"].astype("category").cat.categories))
        # adata.obs["celltype_id"] = celltype_id_labels

# label方式二
        # celltypes = ['G2/M phase', 'Root cortex', 'Root cap', 'S phase', 'Root hair', 
        # 'Lateral root cap', 'Non-hair', 'Columella root cap', 'Phloem', 'Root endodermis', 
        # 'Xylem', 'G1/G0 phase', 'Xylem pole pericycle', 'Root procambium', 'Phloem pole pericycle', 
        # 'Companion cell', 'Metaxylem', 'Sieve element', 'Protoxylem', 'Root stele', 'Pericycle', 
        # 'Meristematic cell', 'Stem cell niche', 'Root epidermis', 'Phloem/Pericycle', 
        # 'Phloem parenchyma', 'Lateral root primordia', 'Unknow']
        celltypes = [
        "Root hair", "Pericycle", "Lateral root cap", "Non-hair", "Root procambium",
        "Root endodermis", "Root cortex", "Xylem", "Columella", "Phloem",
        "Quiescent Center", "Unknow", "Phloem pole pericycle", "Xylem pole",
        "Root epidermis", "Root stele", "Xylem pole pericycle",
        "Lateral root endodermis", "Lateral root primordia", "G2/M phase",
        "Root cap", "S phase", "Columella root cap", "G1/G0 phase",
        "Companion cell", "Metaxylem", "Sieve element", "Protoxylem",
        "Meristematic cell", "Stem cell niche", "Phloem/Pericycle",
        "Phloem parenchyma"]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata.obs["celltype"]=adata.obs['Celltype']
        adata.obs["celltype"] = pd.Categorical(adata.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
 
 
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()
        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
    
    elif args.data_name == "root_152766":
        data_dir = Path(data_path)
        adata = anndata.read(data_dir /"train_all.h5ad")  
        # adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        data_is_raw = True
        filter_gene_by_counts = False
        adata.var["gene_name"] = adata.var.index.tolist()

#label方式一
        # celltype_id_labels = adata.obs["Celltype"].astype("category").cat.codes.values
        # celltypes = adata.obs["Celltype"].unique()
        # num_types = len(np.unique(celltype_id_labels))
        # id2type = dict(enumerate(adata.obs["Celltype"].astype("category").cat.categories))
        # adata.obs["celltype_id"] = celltype_id_labels

# label方式二
        # celltypes = ['G2/M phase', 'Root cortex', 'Root cap', 'S phase', 'Root hair', 
        # 'Lateral root cap', 'Non-hair', 'Columella root cap', 'Phloem', 'Root endodermis', 
        # 'Xylem', 'G1/G0 phase', 'Xylem pole pericycle', 'Root procambium', 'Phloem pole pericycle', 
        # 'Companion cell', 'Metaxylem', 'Sieve element', 'Protoxylem', 'Root stele', 'Pericycle', 
        # 'Meristematic cell', 'Stem cell niche', 'Root epidermis', 'Phloem/Pericycle', 
        # 'Phloem parenchyma', 'Lateral root primordia', 'Unknow']
        celltypes = [
       "Phloem pole pericycle", "Root cortex", "Unknow", "Root hair", "Columella", 
        "Non-hair", "Xylem pole", "Root endodermis", "Quiescent Center", "Phloem", "Xylem"]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata.obs["celltype"]=adata.obs['Celltype']
        adata.obs["celltype"] = pd.Categorical(adata.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
 
 
        train_idx,valid_idx =train_test_split(range(adata.n_obs),test_size=0.2, random_state=42)
        print(train_idx[0],valid_idx[0])
        adata_test = adata[valid_idx].copy()
        adata = adata[train_idx].copy()
        adata_test_raw=adata_test.copy()
        adata.obs["batch_id"] = 0
        adata_test.obs["batch_id"] = 1
            
    else:
        raise f'invalid dataset {args.data_name} for task {args.task}'


    logger=kwargs['logger']
    vocab=kwargs['vocab']
    is_master=kwargs['is_master']
    mask_value=kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master,logger=logger)  # only retain the gene that appears in vocab
    adata_test, _ = filter_gene(vocab=vocab, adata=adata_test, is_master=is_master, logger=logger)

    # set up the preprocessor, use the args to config the workflow
    # if args.data_name !='annimal' :
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
    preprocessor(adata_test, batch_key=None)
    # print(adata.obs['celltype'],'\n')
    # for i in range(adata.n_obs):
    #     print("adata.layers['X_binned'][:][:11]", adata.layers['X_binned'][i][:11])

    train_data_pt,valid_data_pt,test_data_pt,num_batch_types=prepare_cell_data(adata=adata,adata_test=adata_test,args=args,
                                                                        vocab=vocab,is_master=is_master,mask_value=mask_value,
                                                                        pad_value=pad_value,logger=logger,sort_seq_batch=False,pad_token=pad_token)

    return train_data_pt,valid_data_pt,test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_test_raw


def Integration_dataset(data_path,args,**kwargs):
    if args.data_name == "PBMC10K":
        adata = sc.read_h5ad(os.path.join(data_path, 'pbmc_10k.h5ad'))  # 11990 × 19099
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        data_is_raw = False
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()

    elif args.data_name == "pancreas":
        adata = sc.read_h5ad(os.path.join(data_path, 'pancreas.h5ad'))
        ori_batch_col = "batch_id"
        adata.obs["celltype"] = adata.obs["ontology_name"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = False
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var = adata.var.set_index('Symbol')
        adata.var["gene_name"] = adata.var.index.tolist()
        #sc.pp.pca(adata)
    elif args.data_name == "covid":
        adata = sc.read_h5ad(os.path.join(data_path, 'covid_subsampled.h5ad'))
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = True
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        #sc.pp.pca(adata)
    elif args.data_name == "perirhinal":
        adata = sc.read_h5ad(os.path.join(data_path, 'PerirhinalCortex.h5ad'))
        ori_batch_col = "sample_id"
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = True
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var = adata.var.set_index('Gene')
        adata.var["gene_name"] = adata.var.index.tolist()
        #sc.pp.pca(adata)
    elif args.data_name == "humanDC":
        adata = sc.read_h5ad(os.path.join(data_path, 'humanDC.h5ad'))
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = False
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        #adata.var = adata.var.set_index('Gene')
        adata.var["gene_name"] = adata.var.index.tolist()
        #sc.pp.pca(adata)
    elif args.data_name == "hPBMC":
        adata = sc.read_h5ad(os.path.join(data_path, 'hPBMC.h5ad'))
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["CellType"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = False
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        # adata.var = adata.var.set_index('Gene')
        adata.var["gene_name"] = adata.var.index.tolist()
        # sc.pp.pca(adata)
    elif args.data_name == "hPancreas":
        adata = sc.read_h5ad(os.path.join(data_path, 'hPancreas.h5ad'))
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = False
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        # adata.var = adata.var.set_index('Gene')
        adata.var["gene_name"] = adata.var.index.tolist()
        # sc.pp.pca(adata)
    elif args.data_name == "purified_pbmc":
        adata = sc.read_h5ad(os.path.join(data_path, 'purified_pbmc.h5ad'))
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["cell_types"].astype("category")
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = celltype_id_labels
        num_cell_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

        data_is_raw = False
        # %%
        # make the batch category column, used in test data
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        # adata.var = adata.var.set_index('Gene')
        adata.var["gene_name"] = adata.var.index.tolist()
        
    else:
        raise f'unvalid {args.task} dataset {args.data_name}'

    logger = kwargs['logger']
    vocab = kwargs['vocab']
    is_master = kwargs['is_master']
    mask_value = kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata, _ = filter_gene(vocab=vocab, adata=adata, is_master=is_master,
                           logger=logger)  # only retain the gene that appears in vocab

    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=3,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=args.n_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="str_batch" if args.data_name != "heart_cell" else None)
    if args.per_seq_batch_sample:
        # sort the adata by batch_id in advance
        adata_test = adata[adata.obs["batch_id"].argsort()].copy()
    else:
        adata_test=adata.copy()
    train_data_pt, valid_data_pt, test_data_pt, num_batch_types=prepare_cell_data(adata=adata,adata_test=adata_test,args=args,vocab=vocab,
                                                                      is_master=is_master,mask_value=mask_value,
                                                                      pad_value=pad_value,logger=logger,
                                                                      sort_seq_batch=False,pad_token=pad_token)
    return train_data_pt,valid_data_pt,test_data_pt,num_batch_types,adata_test,num_cell_types,id2type

def Pretraining_dataset(data_path,args,**kwargs):
    if args.data_name =='panglao':
        if not args.lmdb:
            data_path=os.path.join(data_path,'binned')
            os.makedirs(data_path,exist_ok=True)
            return H5adDataset(n_bins = args.n_bins, result_binned_key = "X_binned",
                                source_dir=args.source_path,prep_dir = data_path,**kwargs)
        else:
            if args.mask_geneid:
                train_path=osp.join(args.source_path,'train.db')
                valid_path=osp.join(args.source_path,'val.db')
                train_data=gene_mask_LMDBDataset(db_path=train_path,bin_num=args.n_bins,args=args,**kwargs)
                valid_data=gene_mask_LMDBDataset(db_path=valid_path,bin_num=args.n_bins,args=args,**kwargs)
            else:
                train_path=osp.join(args.source_path,'train.db')
                valid_path=osp.join(args.source_path,'val.db')
                train_data=LMDBDataset(db_path=train_path,bin_num=args.n_bins,args=args,**kwargs)
                valid_data=LMDBDataset(db_path=valid_path,bin_num=args.n_bins,args=args,**kwargs)
            return train_data,valid_data
    elif args.data_name=='cellxgene':
        train_path = osp.join(args.source_path, 'all.db.2024.03.06')
        train_data = LMDBDataset(db_path=train_path, bin_num=args.n_bins, args=args, **kwargs)

        valid_path = r'/home/share/huadjyin/home/s_jiangwenjian/proj/scLLM/scGPT/data/Pretraining/panglao/binned/val.db'  # TODO:replace this if new val_set is available
        valid_args=copy.deepcopy(args)
        valid_args.data_name='panglao'# TODO:replace this if new val_set is available
        valid_data = LMDBDataset(db_path=valid_path, bin_num=args.n_bins, args=valid_args, **kwargs)
        return train_data, valid_data
    else:
        raise f'Invalid {args.task} dataset {args.data_name}'

class LMDBDataset(Dataset):
    def __init__(self, db_path,n_bins=51,args=None,pad_value=-2,**kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.args=args
        self.invalid_datapoint_count=0
        self.pad_value=pad_value
        # self.mask_token=kwargs['mask_token']   没用到
        # self.unk_token=kwargs['unk_token']  没用
        # if args.data_name=='panglao':
        #     # self.gene_idx_array = np.array(np.load(args.gene_array_file, allow_pickle=True))
        #     with open(args.gene_array_file, 'r') as file:
        #         gene_number = [int(line.strip()) for line in file]
        #     self.gene_idx_array =np.array(gene_number)
        # else:
        #     self.gene_idx_array=None

        self.n_bins = n_bins
        self.mask_ratio = kwargs['mask_ratio']
        self.append_cls = kwargs['append_cls']
        self.include_zero_gene = kwargs["include_zero_gene"]
        self.max_seq_len = kwargs["max_seq_len"]
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        #arr[arr > threshold] = threshold
        self.graph_sort=args.graph_sort
        self.sampling_etype=args.sampling_etype
        if self.graph_sort:
            self.layer_mask=args.layer_mask
            if args.sampling_etype in ['share_pathway_with','interact_with','co_expression']:
                graph=dgl.load_graphs(os.path.join(args.graph_path,'knowledgebase_cxg.dgl'))[0][0]
                self.sampling_g=dgl.edge_type_subgraph(graph,etypes=[args.sampling_etype])
                del graph
            else:
                self.sampling_g=None
            self.grn=dgl.load_graphs(os.path.join(args.graph_path,'kb_acyclic_reg_cxg.dgl'))[0][0]
            # if args.data_name == 'panglao':
            #     self.valid_idx=self.gene_idx_array<self.grn.num_nodes()#TODO:delete this if new dataset is valid
            #     self.gene_idx_array=self.gene_idx_array[self.valid_idx]#TODO:delete this if new dataset is valid

        with self.env.begin(write=False) as txn:
            self.length = txn.get(b'__len__')

    def __getitem__(self, index):
        values = self.txn.get(u'{}'.format(index).encode())
        try:
            if self.args.data_name == 'panglao':
                try:
                    values = np.frombuffer(values,dtype=np.float16)  # np.array([gene_num,])
                    # print(values)
                    # values = values[self.valid_idx]#TODO:delete this if new dataset is valid
                    # gene_ids=self.gene_idx_array
                    # print("len(values)",len(values))   #用到所有基因，不是高变基因。len(values) 32401
                    # gene_ids = np.random.choice(range(0, len(self.vocab) - 10), size=(self.max_seq_len * 2,),replace=False).astype(np.int64)
                    # print("len(gene_ids)",len(gene_ids))
                    gene_ids=np.array(range(0,len(values)))    #同时gene——id也是32401长度，即每个value的索引为gene_id
                    # print("len(gene_ids)",len(gene_ids))
                except Exception as e:  # 捕获所有类型的异常
                     print("error:", e)
            else:
                print("in else")
                datapoint=json.loads(values) # ['express_x','organ','celltype','sequence','disease','gene_index']
                values=np.array(datapoint['express_x'])
                print("values:",values,"\n",datapoint["gene_index"])
                gene_ids=np.frombuffer(self.txn.get(key=u'{}'.format(datapoint['gene_index']).encode()),dtype=np.int64)
                if len(values) != len(gene_ids):
                    gene_ids=np.random.choice(range(0, len(self.vocab)-10), size=(self.max_seq_len*2,), replace=False).astype(np.int64)
                    self.invalid_datapoint_count+=1
        except:  #统计无效数据点，并将基因id和值随机生成，
            print("出错，执行except")
            self.invalid_datapoint_count += 1
            gene_ids=np.random.choice(range(0, len(self.vocab)-10), size=(self.max_seq_len*2,), replace=False).astype(np.int64)
            values_nz=np.random.uniform(0, 5, size=(int(self.max_seq_len *0.1),)).astype(np.float64)
            values=np.zeros_like(gene_ids,dtype=np.float64)
            values[:int(self.max_seq_len *0.1)]=values_nz
            np.random.shuffle(values)
        assert len(values) == len(gene_ids)
        binned_values,bin_edge=self._binning(values)
        values,gene_ids,masked_values,sorted_gene_ids,masked_sorted_gene_ids,sorted_layer_idx=self._pad_and_mask(binned_values,gene_ids=gene_ids)
        datapoint = {"gene_ids": gene_ids, "masked_values": masked_values, "target_values": values,
                     "sorted_gene_ids": sorted_gene_ids,"masked_sorted_gene_ids":masked_sorted_gene_ids,"sorted_layer_idx":sorted_layer_idx}
        return datapoint

    def __len__(self):
        return int(self.txn.get(b'__len__').decode("utf-8"))
    def _binning(self,values):
        non_zero_ids = values.nonzero()
        non_zero_row = values[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
        # bins = np.sort(np.unique(bins))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        non_zero_digits = self._digitize(non_zero_row, bins)
        assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= self.n_bins - 1
        binned_row = np.zeros_like(values, dtype=np.int64).copy()
        binned_row[non_zero_ids] = non_zero_digits
        bin_edge=np.concatenate([[0], bins])
        return binned_row,bin_edge

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        #这个函数的作用是将数据分箱，即将数据映射到给定的区间（bins）。具体步骤如下：
        assert x.ndim == 1 and bins.ndim == 1
#找到每个数据点在bins中的位置，即数据点所属的区间。这一步会得到每个数据点在bins中左边界的位置和右边界的位置。
        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)
#使用均匀分布的随机数生成器（np.random.rand）生成一组随机数，用于在区间内均匀地填充数据点
        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits#将数据点在区间内均匀地分布
        digits = np.ceil(digits).astype(np.int64)  #分箱数字向上取整，并转换为整数类型，作为数据点在分箱后的结果。
        return digits
    def _pad_and_mask(self,values,gene_ids):
        if self.include_zero_gene:
            values=values
            gene_ids=gene_ids
        else:
            idx=np.nonzero(values)[-1]
            values=np.array(values[idx])
            gene_ids=np.array(gene_ids[idx])


        if len(gene_ids)>self.max_seq_len:
            idx=np.random.choice(len(gene_ids),self.max_seq_len,replace=False)
            gene_ids=gene_ids[idx]
            values=values[idx]
            # masked_values=masked_values[idx]
        # print('value_pretrain',values,len(values))    
        masked_values = random_mask_value1(values, self.mask_ratio)

        if self.graph_sort:
            # if self.sampling_etype=='ori':
            #     sorted_gene_ids, sorted_layer_idx,_ = self.topological_sorting(gene_ids,values,sample=self.sampling_g is not None)# the length here would <=max_len
            # else:
            #     sorted_gene_ids, sorted_layer_idx,_ = self.topological_sorting(gene_ids,values,sample=True)
            sorted_gene_ids, sorted_layer_idx, _ = self.topological_sorting(gene_ids, values=None,
                                                                            sample=self.sampling_g is not None)  # the length here would <=max_len
            gene_ids, layer_idx, values = self.topological_sorting(gene_ids, values,sample=False)
            mask_id = self.vocab[self.mask_token]
            if sorted_gene_ids.__len__()<self.max_seq_len:
                pad_id = self.vocab['<pad>']
                pad_layer_idx=0
                sorted_gene_ids=np.concatenate([sorted_gene_ids,np.full(self.max_seq_len - len(sorted_gene_ids), pad_id, dtype=sorted_gene_ids.dtype)])
                sorted_layer_idx = np.concatenate([sorted_layer_idx,np.full(self.max_seq_len - len(sorted_layer_idx), pad_layer_idx,dtype=sorted_layer_idx.dtype)])
            if self.layer_mask:
                selected_masked_layer = random.sample(range(1, max(sorted_layer_idx)+1), min(1,int(max(sorted_layer_idx)*self.mask_ratio)))
                assert selected_masked_layer.__len__()<max(sorted_layer_idx)
                masking_position=np.isin(sorted_layer_idx,selected_masked_layer)
                masked_sorted_gene_ids = sorted_gene_ids.copy()
                masked_sorted_gene_ids[masking_position] = mask_id
            else:
                masked_sorted_gene_ids=random_mask_value(values=sorted_gene_ids,mask_ratio=self.mask_ratio,
                                                         mask_value=self.vocab[self.mask_token],pad_value=self.vocab['<pad>'])
                masked_sorted_gene_ids=torch.from_numpy(masked_sorted_gene_ids)


        ## padding
        if len(gene_ids) < self.max_seq_len:
            pad_id = self.vocab['<pad>']
            gene_ids = np.concatenate(
                [gene_ids, np.full(self.max_seq_len - len(gene_ids), pad_id, dtype=gene_ids.dtype)])
            values = np.concatenate([values, np.full(self.max_seq_len - len(values), self.pad_value, dtype=values.dtype)])
            masked_values = np.concatenate(
                [masked_values, np.full(self.max_seq_len - len(masked_values), self.pad_value, dtype=masked_values.dtype)])


        if self.append_cls:
            values = np.insert(values, 0, 0)
            gene_ids = np.insert(gene_ids, 0, self.vocab['<cls>'])
            masked_values = np.insert(masked_values, 0, 0)
            if self.graph_sort:
                masked_sorted_gene_ids=np.insert(masked_sorted_gene_ids, 0, self.vocab['<cls>'])
                sorted_gene_ids = np.insert(sorted_gene_ids, 0, self.vocab['<cls>'])
                sorted_layer_idx= np.insert(sorted_layer_idx, 0, 0)

        if self.graph_sort:
            masked_sorted_gene_ids=torch.tensor(masked_sorted_gene_ids).int()
            sorted_gene_ids = torch.tensor(sorted_gene_ids).int()
            sorted_layer_idx = torch.tensor(sorted_layer_idx).int()
        else:
            masked_sorted_gene_ids=0
            sorted_gene_ids=0
            sorted_layer_idx=0

        return torch.tensor(values).float(),torch.tensor(gene_ids).int(),torch.tensor(masked_values).float(),sorted_gene_ids,masked_sorted_gene_ids,sorted_layer_idx
    def topological_sorting(self,gene_ids,values,sample=False):
        if sample and (len(gene_ids)<self.max_seq_len):
            assert self.sampling_g is not None
            sub_g = dgl.sampling.sample_neighbors(self.sampling_g, nodes={'gene': torch.tensor(gene_ids)}, fanout=5,
                                                  edge_dir='out')
            unique_node = torch.cat([torch.tensor(gene_ids), sub_g.edges(order='srcdst')[0],
                                     sub_g.edges(order='srcdst')[1]]).unique().tolist()
            # remove the isolate&not_ori node
            sub_grn = dgl.node_subgraph(self.grn, unique_node)
            is_isolate = np.array(torch.logical_and(sub_grn.in_degrees() == 0, sub_grn.out_degrees() == 0))
            is_ori = np.isin(np.array(sub_grn.ndata[dgl.NID]), gene_ids)
            valid_node = sub_grn.ndata['_ID'][torch.from_numpy(~np.logical_and(is_isolate, ~is_ori))]
            if len(valid_node)>self.max_seq_len:
                valid_graph = dgl.node_subgraph(self.grn, gene_ids)
            else:
                valid_graph = dgl.node_subgraph(self.grn, valid_node)

        else:
            valid_graph = dgl.node_subgraph(self.grn, gene_ids)

        topo_sorting = dgl.topological_nodes_generator(valid_graph)
        sort_layer_idx = []
        for idx, layer in enumerate(topo_sorting):
            sort_layer_idx += [idx+1] * len(layer)
        sorted_index = torch.cat(topo_sorting)
        sorting_gene_ids = valid_graph.ndata['_ID'][sorted_index]
        if values is not None:
            sorting_values=np.array(values[sorted_index])
        else:
            sorting_values =None

        return np.array(sorting_gene_ids),np.array(sort_layer_idx),sorting_values
class gene_mask_LMDBDataset(Dataset):
    def __init__(self, db_path,n_bins=51,args=None,pad_value=-2,**kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.args=args
        self.invalid_datapoint_count=0
        self.pad_value=pad_value
        # self.mask_token=kwargs['mask_token']   没用到
        # self.unk_token=kwargs['unk_token']  没用
        # if args.data_name=='panglao':
        #     # self.gene_idx_array = np.array(np.load(args.gene_array_file, allow_pickle=True))
        #     with open(args.gene_array_file, 'r') as file:
        #         gene_number = [int(line.strip()) for line in file]
        #     self.gene_idx_array =np.array(gene_number)
        # else:
        #     self.gene_idx_array=None

        self.n_bins = n_bins
        self.mask_ratio = kwargs['mask_ratio']
        self.append_cls = kwargs['append_cls']
        self.include_zero_gene = kwargs["include_zero_gene"]
        self.max_seq_len = kwargs["max_seq_len"]
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        #arr[arr > threshold] = threshold
        self.graph_sort=args.graph_sort
        self.sampling_etype=args.sampling_etype
        if self.graph_sort:
            self.layer_mask=args.layer_mask
            if args.sampling_etype in ['share_pathway_with','interact_with','co_expression']:
                graph=dgl.load_graphs(os.path.join(args.graph_path,'knowledgebase_cxg.dgl'))[0][0]
                self.sampling_g=dgl.edge_type_subgraph(graph,etypes=[args.sampling_etype])
                del graph
            else:
                self.sampling_g=None
            self.grn=dgl.load_graphs(os.path.join(args.graph_path,'kb_acyclic_reg_cxg.dgl'))[0][0]
            # if args.data_name == 'panglao':
            #     self.valid_idx=self.gene_idx_array<self.grn.num_nodes()#TODO:delete this if new dataset is valid
            #     self.gene_idx_array=self.gene_idx_array[self.valid_idx]#TODO:delete this if new dataset is valid

        with self.env.begin(write=False) as txn:
            self.length = txn.get(b'__len__')

    def __getitem__(self, index):
        values = self.txn.get(u'{}'.format(index).encode())
        try:
            if self.args.data_name == 'panglao':
                try:
                    values = np.frombuffer(values,dtype=np.float16)  # np.array([gene_num,])
                    # print(values)
                    # values = values[self.valid_idx]#TODO:delete this if new dataset is valid
                    # gene_ids=self.gene_idx_array
                    # print("len(values)",len(values))   #用到所有基因，不是高变基因。len(values) 32401
                    # gene_ids = np.random.choice(range(0, len(self.vocab) - 10), size=(self.max_seq_len * 2,),replace=False).astype(np.int64)
                    # print("len(gene_ids)",len(gene_ids))
                    gene_ids=np.array(range(0,len(values)))    #同时gene——id也是32401长度，即每个value的索引为gene_id
                    # print("len(gene_ids)",len(gene_ids))
                except Exception as e:  # 捕获所有类型的异常
                     print("error:", e)
            else:
                print("in else")
                datapoint=json.loads(values) # ['express_x','organ','celltype','sequence','disease','gene_index']
                values=np.array(datapoint['express_x'])
                print("values:",values,"\n",datapoint["gene_index"])
                gene_ids=np.frombuffer(self.txn.get(key=u'{}'.format(datapoint['gene_index']).encode()),dtype=np.int64)
                if len(values) != len(gene_ids):
                    gene_ids=np.random.choice(range(0, len(self.vocab)-10), size=(self.max_seq_len*2,), replace=False).astype(np.int64)
                    self.invalid_datapoint_count+=1
        except:  #统计无效数据点，并将基因id和值随机生成，
            print("出错，执行except")
            self.invalid_datapoint_count += 1
            gene_ids=np.random.choice(range(0, len(self.vocab)-10), size=(self.max_seq_len*2,), replace=False).astype(np.int64)
            values_nz=np.random.uniform(0, 5, size=(int(self.max_seq_len *0.1),)).astype(np.float64)
            values=np.zeros_like(gene_ids,dtype=np.float64)
            values[:int(self.max_seq_len *0.1)]=values_nz
            np.random.shuffle(values)
        assert len(values) == len(gene_ids)
        binned_values,bin_edge=self._binning(values)
        values,gene_ids,masked_values,masked_genes,sorted_gene_ids,masked_sorted_gene_ids,sorted_layer_idx=self._pad_and_mask(binned_values,gene_ids=gene_ids)
        datapoint = {"target_genes": gene_ids,"masked_genes": masked_genes, "masked_values": masked_values, "target_values": values,
                     "sorted_gene_ids": sorted_gene_ids,"masked_sorted_gene_ids":masked_sorted_gene_ids,"sorted_layer_idx":sorted_layer_idx}
        return datapoint

    def __len__(self):
        return int(self.txn.get(b'__len__').decode("utf-8"))
    def _binning(self,values):
        non_zero_ids = values.nonzero()
        non_zero_row = values[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
        # bins = np.sort(np.unique(bins))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        non_zero_digits = self._digitize(non_zero_row, bins)
        assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= self.n_bins - 1
        binned_row = np.zeros_like(values, dtype=np.int64).copy()
        binned_row[non_zero_ids] = non_zero_digits
        bin_edge=np.concatenate([[0], bins])
        return binned_row,bin_edge            #此处的binned_row是包含0的所有数据

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        #这个函数的作用是将数据分箱，即将数据映射到给定的区间（bins）。具体步骤如下：
        assert x.ndim == 1 and bins.ndim == 1
#找到每个数据点在bins中的位置，即数据点所属的区间。这一步会得到每个数据点在bins中左边界的位置和右边界的位置。
        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)
#使用均匀分布的随机数生成器（np.random.rand）生成一组随机数，用于在区间内均匀地填充数据点
        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits#将数据点在区间内均匀地分布
        digits = np.ceil(digits).astype(np.int64)  #分箱数字向上取整，并转换为整数类型，作为数据点在分箱后的结果。
        return digits
    def _pad_and_mask(self,values,gene_ids):
        if self.include_zero_gene:
            values=values
            gene_ids=gene_ids
        else:
            idx=np.nonzero(values)[-1]
            values=np.array(values[idx])
            gene_ids=np.array(gene_ids[idx])


        if len(gene_ids)>self.max_seq_len:
            idx=np.random.choice(len(gene_ids),self.max_seq_len,replace=False)
            gene_ids=gene_ids[idx]
            values=values[idx]
            # masked_values=masked_values[idx]
        # print('value_pretrain',values,len(values))    
        masked_values = random_mask_value1(values, self.mask_ratio)
        masked_genes = random_mask_value1(gene_ids, self.mask_ratio,self.vocab['<mask>'],self.vocab['<pad>'])
        # print(masked_values[0:15],'\n',masked_genes[0:15])
        if self.graph_sort:
            # if self.sampling_etype=='ori':
            #     sorted_gene_ids, sorted_layer_idx,_ = self.topological_sorting(gene_ids,values,sample=self.sampling_g is not None)# the length here would <=max_len
            # else:
            #     sorted_gene_ids, sorted_layer_idx,_ = self.topological_sorting(gene_ids,values,sample=True)
            sorted_gene_ids, sorted_layer_idx, _ = self.topological_sorting(gene_ids, values=None,
                                                                            sample=self.sampling_g is not None)  # the length here would <=max_len
            gene_ids, layer_idx, values = self.topological_sorting(gene_ids, values,sample=False)
            mask_id = self.vocab[self.mask_token]
            if sorted_gene_ids.__len__()<self.max_seq_len:
                pad_id = self.vocab['<pad>']
                pad_layer_idx=0
                sorted_gene_ids=np.concatenate([sorted_gene_ids,np.full(self.max_seq_len - len(sorted_gene_ids), pad_id, dtype=sorted_gene_ids.dtype)])
                sorted_layer_idx = np.concatenate([sorted_layer_idx,np.full(self.max_seq_len - len(sorted_layer_idx), pad_layer_idx,dtype=sorted_layer_idx.dtype)])
            if self.layer_mask:
                selected_masked_layer = random.sample(range(1, max(sorted_layer_idx)+1), min(1,int(max(sorted_layer_idx)*self.mask_ratio)))
                assert selected_masked_layer.__len__()<max(sorted_layer_idx)
                masking_position=np.isin(sorted_layer_idx,selected_masked_layer)
                masked_sorted_gene_ids = sorted_gene_ids.copy()
                masked_sorted_gene_ids[masking_position] = mask_id
            else:
                masked_sorted_gene_ids=random_mask_value(values=sorted_gene_ids,mask_ratio=self.mask_ratio,
                                                         mask_value=self.vocab[self.mask_token],pad_value=self.vocab['<pad>'])
                masked_sorted_gene_ids=torch.from_numpy(masked_sorted_gene_ids)


        ## padding
        if len(gene_ids) < self.max_seq_len:
            pad_id = self.vocab['<pad>']
            gene_ids = np.concatenate(
                [gene_ids, np.full(self.max_seq_len - len(gene_ids), pad_id, dtype=gene_ids.dtype)])
            values = np.concatenate([values, np.full(self.max_seq_len - len(values), self.pad_value, dtype=values.dtype)])
            masked_values = np.concatenate(
                [masked_values, np.full(self.max_seq_len - len(masked_values), self.pad_value, dtype=masked_values.dtype)])
            masked_genes = np.concatenate(
                [masked_genes, np.full(self.max_seq_len - len(masked_genes), pad_id, dtype=masked_genes.dtype)])


        if self.append_cls:
            values = np.insert(values, 0, 0)
            gene_ids = np.insert(gene_ids, 0, self.vocab['<cls>'])
            masked_values = np.insert(masked_values, 0, 0)
            masked_genes = np.insert(masked_genes, 0, self.vocab['<cls>'])
            if self.graph_sort:
                masked_sorted_gene_ids=np.insert(masked_sorted_gene_ids, 0, self.vocab['<cls>'])
                sorted_gene_ids = np.insert(sorted_gene_ids, 0, self.vocab['<cls>'])
                sorted_layer_idx= np.insert(sorted_layer_idx, 0, 0)

        if self.graph_sort:
            masked_sorted_gene_ids=torch.tensor(masked_sorted_gene_ids).int()
            sorted_gene_ids = torch.tensor(sorted_gene_ids).int()
            sorted_layer_idx = torch.tensor(sorted_layer_idx).int()
        else:
            masked_sorted_gene_ids=0
            sorted_gene_ids=0
            sorted_layer_idx=0

        return torch.tensor(values).float(),torch.tensor(gene_ids).int(),torch.tensor(masked_values).float(),torch.tensor(masked_genes).int(),sorted_gene_ids,masked_sorted_gene_ids,sorted_layer_idx

        # return torch.tensor(values).float(),gene_ids,torch.tensor(masked_values).float(),masked_genes,sorted_gene_ids,masked_sorted_gene_ids,sorted_layer_idx
    def topological_sorting(self,gene_ids,values,sample=False):
        if sample and (len(gene_ids)<self.max_seq_len):
            assert self.sampling_g is not None
            sub_g = dgl.sampling.sample_neighbors(self.sampling_g, nodes={'gene': torch.tensor(gene_ids)}, fanout=5,
                                                  edge_dir='out')
            unique_node = torch.cat([torch.tensor(gene_ids), sub_g.edges(order='srcdst')[0],
                                     sub_g.edges(order='srcdst')[1]]).unique().tolist()
            # remove the isolate&not_ori node
            sub_grn = dgl.node_subgraph(self.grn, unique_node)
            is_isolate = np.array(torch.logical_and(sub_grn.in_degrees() == 0, sub_grn.out_degrees() == 0))
            is_ori = np.isin(np.array(sub_grn.ndata[dgl.NID]), gene_ids)
            valid_node = sub_grn.ndata['_ID'][torch.from_numpy(~np.logical_and(is_isolate, ~is_ori))]
            if len(valid_node)>self.max_seq_len:
                valid_graph = dgl.node_subgraph(self.grn, gene_ids)
            else:
                valid_graph = dgl.node_subgraph(self.grn, valid_node)

        else:
            valid_graph = dgl.node_subgraph(self.grn, gene_ids)

        topo_sorting = dgl.topological_nodes_generator(valid_graph)
        sort_layer_idx = []
        for idx, layer in enumerate(topo_sorting):
            sort_layer_idx += [idx+1] * len(layer)
        sorted_index = torch.cat(topo_sorting)
        sorting_gene_ids = valid_graph.ndata['_ID'][sorted_index]
        if values is not None:
            sorting_values=np.array(values[sorted_index])
        else:
            sorting_values =None

        return np.array(sorting_gene_ids),np.array(sort_layer_idx),sorting_values
class H5adDataset(Dataset):
    def __init__(self,n_bins=51,source_dir='',result_binned_key="X_binned",prep_dir='',**kwargs):
        self.n_bins=n_bins
        h5ad_file_list = [file for file in os.listdir(source_dir) if file.endswith('.h5ad')]
        self.h5ad_file_list = h5ad_file_list
        self.prep_dir=prep_dir
        self.vocab=kwargs['vocab']
        self.mask_ratio=kwargs['mask_ratio']
        self.append_cls=kwargs['append_cls']
        self.include_zero_gene=kwargs["include_zero_gene"]
        self.max_seq_len=kwargs["max_seq_len"]
        print("Binning and filtering data ...")
        if not isinstance(n_bins, int):
            raise ValueError(
                "Binning arg must be an integer, but got {}.".format(n_bins)
            )
        self.length_list=[]
        self.gene_num=[]
        self.n_files=0
        self.max_non_zero_count=0
        self.min_non_zero_count=float('inf')
        for file in self.h5ad_file_list:
            print(file)
            self.n_files+=1
            target_file=osp.join(prep_dir,file)
            if os.path.exists(target_file):
                if kwargs['need_length']:
                    adata = anndata.read_h5ad(target_file)
                    self.length_list.append(adata.n_obs)
                    self.gene_num.append(adata.n_vars)
                    self.max_non_zero_count=max((adata.X > 0).sum(axis=1).max(),self.max_non_zero_count)
                    self.min_non_zero_count = min((adata.X > 0).sum(axis=1).min(), self.min_non_zero_count)
                else:
                    self.length_list.append(0)
                    self.gene_num.append(0)
                continue
            ## filter genes that don't exist in vocab
            adata = anndata.read_h5ad(osp.join(source_dir, file))
            print('filering the genes')
            adata,_=filter_gene(self.vocab,adata,True,None)
            ## binning
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=None)  # Return values for observations in adata.
            ##如果返回的数据是稀疏矩阵，则将其转换为密集矩阵（numpy数组）并存储在layer_data中
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                # if len(non_zero_row)==0:
                #     binned_row = np.zeros_like(row, dtype=np.int64).copy()
                    
                #     binned_rows.append(binned_row)
                #     #记录每个分箱的边界值，并将其添加到bin_edges列表中
                #     bin_edges.append(np.concatenate([[0], bins]))
                    
                #     continue
                #使用np.quantile函数计算非零元素的分位数，将其划分为n_bins - 1个区间。
                # try:
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # except:
                #     print("error file is :",file,non_zero_row)
                    
                #     print("non_zero_ids:",non_zero_ids,"\n","non_zero_row:",non_zero_row)
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                #非零元素的值映射到对应的区间，得到每个元素的分箱数字
                non_zero_digits = self._digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64).copy()
                binned_row[non_zero_ids] = non_zero_digits
                #根据分箱数字构建新的行数据，并将其添加到binned_rows列表中
                binned_rows.append(binned_row)
                #记录每个分箱的边界值，并将其添加到bin_edges列表中
                bin_edges.append(np.concatenate([[0], bins]))
            # try:
            adata.layers[result_binned_key] = np.stack(binned_rows)  #np.stack:将binned_rows中的分箱数据堆叠起来。
            # except:
            #     print("binned_rows",binned_rows)
            
            adata.obsm["bin_edges"] = np.stack(bin_edges)  
            self.length_list.append(adata.n_obs)
            self.gene_num.append(adata.n_vars)
            self.max_non_zero_count = max((adata.X > 0).sum(axis=1).max(), self.max_non_zero_count)
            self.min_non_zero_count = min((adata.X > 0).sum(axis=1).min(), self.min_non_zero_count)
            adata.write_h5ad(target_file)
            self.adata = adata
        assert len(self.length_list)==len(self.h5ad_file_list)
        self.cumulative_sizes=np.cumsum(self.length_list)  #计算列表 self.length_list 的累积和
        print("Binning completed!")


    def __len__(self):
        # Return the total number of samples across all files
        return np.sum(self.length_list)

    def __getitem__(self, idx):
        # Efficiently fetch a single item across the dataset
        if idx < 0 or idx >= self.__len__():
            raise IndexError
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        adjusted_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
        # adata=anndata.read_h5ad(osp.join(self.prep_dir,self.h5ad_file_list[file_idx]))
        adata=anndata.read(osp.join(self.prep_dir,self.h5ad_file_list[file_idx]))
        target_values, gene_ids, masked_values = self._tokenize_and_pad(adata[adjusted_idx], self.mask_ratio,
                                                                        self.max_seq_len)
        target_values=float32(target_values)
        masked_values=float32(masked_values)
        datapoint = {"gene_ids": gene_ids, "masked_values": masked_values, "target_values": target_values,
                     "cell_type": '<unk>'}
        return datapoint

    def _tokenize_and_pad(self,adata,mask_ratio,max_len):
        genes=adata.var_names.tolist()
        values=adata.layers['X_binned']
        if values.shape[1] != len(genes):
            raise ValueError(
                f"Number of features in data ({values.shape[1]}) does not match "
                f"number of gene_ids ({len(genes)})."
            )
        if self.include_zero_gene:
            values=values
            gene_ids=np.array(self.vocab(genes), dtype=int)
        else:
            idx = np.nonzero(adata.X)[-1]
            values =values[:,idx]
            gene_ids = np.array(self.vocab(genes), dtype=int)
            gene_ids = gene_ids[idx]
        if self.append_cls:
            values=np.insert(values,0,0)
            gene_ids=np.insert(gene_ids, 0, self.vocab['<cls>'])
        masked_value=torch.from_numpy(random_mask_value(values,mask_ratio)).float().view(1,-1)
        values, gene_ids=torch.tensor(values).view(1,-1),torch.tensor(gene_ids).view(1,-1)
        if len(gene_ids[-1]) > max_len:
            if not self.append_cls:
                idx = np.random.choice(len(gene_ids[-1]), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids[-1]) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids=gene_ids[:,idx]
            values = values[:,idx]
            masked_value = masked_value[:,idx]
        elif len(gene_ids[-1])< max_len:
            pad_id=self.vocab['<pad>']
            pad_value=-2
            gene_ids = torch.cat([gene_ids,torch.full((1,max_len - gene_ids.size(-1)), pad_id, dtype=gene_ids.dtype)],dim=-1)
            values = torch.cat([values, torch.full((1,max_len - values.size(-1)), pad_value, dtype=values.dtype)],dim=-1)
            masked_value = torch.cat([masked_value, torch.full((1,max_len - masked_value.size(-1)), pad_value, dtype=masked_value.dtype)], dim=-1)
        return values.squeeze(),gene_ids.squeeze(),masked_value.squeeze()

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.  将数据分箱，即将数据映射到给定的区间（bins）

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        #np.digitize函数找到每个数据点在bins中的位置，即数据点所属的区间。
        # 会得到每个数据点在bins中左边界的位置和右边界的位置。
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)
# 用均匀分布的随机数生成器（np.random.rand）生成一组随机数，用于在区间内均匀地填充数据点。
        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits #将数据点在区间内均匀地分布
        digits = np.ceil(digits).astype(np.int64)  #分箱数字向上取整，并转换为整数类型，作为数据点在分箱后的结果
        return digits

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

def random_mask_value2(
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
    return row,mask_idx



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
    

    # print('row',row.shape,'\n',row)
    # print('np.nonzero(row - pad_value)',np.nonzero(row - pad_value))
    # file_path2='/home/share/huadjyin/home/mengqiu/scGPT-main/train/nonzero_row_padid.txt'
    # np.savetxt(file_path2, np.nonzero(row - pad_value), fmt='%i')

    # non_padding_idx = np.nonzero(row - pad_value)[0]
   
    # file_path='/home/share/huadjyin/home/mengqiu/scGPT-main/train/non_padding_idx.txt'
    # np.savetxt(file_path, non_padding_idx, fmt='%i')
    # print('non_padding_idx',non_padding_idx,'\n',len(non_padding_idx))
    
    # n_mask = int(len(non_padding_idx) * mask_ratio)
   
    # print('n_mask',n_mask)
    
    # mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
    
    # print('mask_idx',mask_idx,'\n',len(mask_idx))
   
    # row[mask_idx] = mask_value
    
    # print('row[mask_idx]',row[mask_idx],'\n',row[mask_idx].shape)
    # file_path3='/home/share/huadjyin/home/mengqiu/scGPT-main/train/finall_row.txt'
    # np.savetxt(file_path3, row, fmt='%i')
    # print('row',row)
    # return row

'''
原版
    if isinstance(values, torch.Tensor):
            # it is crutial to clone the tensor, otherwise it changes the original tensor
            values = values.clone().detach().numpy()
        else:
            values = values.copy()
        row = values
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
        return row
'''



        # '''github'''

    # if isinstance(values, torch.Tensor):
    #     # it is crutial to clone the tensor, otherwise it changes the original tensor
    #     values = values.clone().detach().numpy()
    # else:
    #     values = values.copy()

    # for i in range(len(values)):
    #     row = values[i]
    #     non_padding_idx = np.nonzero(row - pad_value)[0]
    #     print('np.nonzero(row - pad_value)',np.nonzero(row - pad_value))
    #     n_mask = int(len(non_padding_idx) * mask_ratio)
    #     print('n_mask',n_mask)
    #     mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
    #     print('mask_idx',mask_idx,len(mask_idx))
    #     row[mask_idx] = mask_value
    #     print('row',row)
    # return torch.from_numpy(values).float()

# data_loader

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

def prepare_cell_data(adata,adata_test,args,vocab,
                 is_master,mask_value,pad_value,logger,sort_seq_batch=False,pad_token='<pad>'
                 ):
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
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    # print('adata.layers[input_layer_key].A',adata.layers[input_layer_key].A)
    genes = adata.var["gene_name"].tolist()
    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)
    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )
    gene_ids = np.array(vocab(genes), dtype=int)
    if args.graph_sort:
        graph = dgl.load_graphs(os.path.join(args.graph_path, 'kb_acyclic_reg_cxg.dgl'))[0][0]
    else:
        graph = None

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=args.append_cls,  # append <cls> token at the beginning
        include_zero_gene=args.include_zero_gene
        # graph=graph
    )
    # print('tokenized_train',tokenized_train)
    # value_at_index_6 = tokenized_train['values'][0][6].item()  # 获取索引6处的值
    # value_at_last_index = tokenized_train['values'][0][-1].item()  # 获取最后一个元素的值

    # 打印并格式化为小数点后多位
    # print(f"Value at index 6: {value_at_index_6:.12f}")
    # print(f"Last value: {value_at_last_index:.12f}")
    # print('tokenized_train[values][0][6]',tokenized_train['values'][0][6],tokenized_train['values'][0][-1])
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=args.append_cls,
        include_zero_gene=args.include_zero_gene
        # graph=graph
    )
    # print('tokenized_valid',tokenized_valid)

    if is_master:
        logger.info(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )
    print('type(tokenized_train["values"])',type(tokenized_train["values"]))
    masked_values_train = torch.from_numpy(random_mask_value(
        tokenized_train["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )).float()
    masked_values_valid = torch.from_numpy(random_mask_value(
        tokenized_valid["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )).float()

    if adata_test is not None:
        all_counts_test = (
            adata_test.layers[input_layer_key].A
            if issparse(adata_test.layers[input_layer_key])
            else adata_test.layers[input_layer_key]
        )
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
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
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
        # "sorted_layer_idx": tokenized_train["sorted_layer_idx"]
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
        # "sorted_layer_idx": tokenized_valid["sorted_layer_idx"]
    }

    return train_data_pt, valid_data_pt,test_data_pt,num_batch_types



def cell_annotation_test_dataset_nobin(testdata_path,args,**kwargs):
    logger = kwargs['logger']
    vocab = kwargs['vocab']
    is_master = kwargs['is_master']
    mask_value = kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    if args.data_name == 'root':
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        # celltypes = ['G2/M phase', 'Root cortex', 'Root cap', 'S phase', 'Root hair', 
        #     'Lateral root cap', 'Non-hair', 'Columella root cap', 'Phloem', 'Root endodermis', 
        #     'Xylem', 'G1/G0 phase', 'Xylem pole pericycle', 'Root procambium', 'Phloem pole pericycle', 
        #     'Companion cell', 'Metaxylem', 'Sieve element', 'Protoxylem', 'Root stele', 'Pericycle', 
        #     'Meristematic cell', 'Stem cell niche', 'Root epidermis', 'Phloem/Pericycle', 
        #     'Phloem parenchyma', 'Lateral root primordia', 'Unknow']
        celltypes = [
        "Root hair", "Pericycle", "Lateral root cap", "Non-hair", "Root procambium",
        "Root endodermis", "Root cortex", "Xylem", "Columella", "Phloem",
        "Quiescent Center", "Unknow", "Phloem pole pericycle", "Xylem pole",
        "Root epidermis", "Root stele", "Xylem pole pericycle",
        "Lateral root endodermis", "Lateral root primordia", "G2/M phase",
        "Root cap", "S phase", "Columella root cap", "G1/G0 phase",
        "Companion cell", "Metaxylem", "Sieve element", "Protoxylem",
        "Meristematic cell", "Stem cell niche", "Phloem/Pericycle",
        "Phloem parenchyma"
        ]

        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1



        # sc.pp.normalize_total(adata_test, target_sum=1e4)
        # # print(adata.X)
        # sc.pp.log1p(adata_test, base=2)


        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        if issparse(adata_test.X):
            # 如果是稀疏格式，转换为密集格式的 NumPy 数组
            all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
        else:
            # 如果已经是密集格式，直接使用 adata.X
            all_counts_test = adata_test.X
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

    elif args.data_name == "root_152766":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
        #     'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
        #     'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
        #     'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
        #     'Leaf epidermis', 'Stress response', 'Meristematic cell']
        celltypes = [
        "Phloem pole pericycle", "Root cortex", "Unknow", "Root hair", "Columella", 
        "Non-hair", "Xylem pole", "Root endodermis", "Quiescent Center", "Phloem", "Xylem"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        # celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1


        # sc.pp.normalize_total(adata_test, target_sum=1e4)
        # # print(adata.X)
        # sc.pp.log1p(adata_test, base=2)


        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        if issparse(adata_test.X):
            # 如果是稀疏格式，转换为密集格式的 NumPy 数组
            all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
        else:
            # 如果已经是密集格式，直接使用 adata.X
            all_counts_test = adata_test.X
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

    elif args.data_name == "leaf":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
        #     'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
        #     'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
        #     'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
        #     'Leaf epidermis', 'Stress response', 'Meristematic cell']
        celltypes = [
        "Leaf epidermis", "Palisade mesophyll cell", "Spongy mesophyll cell", 
        "Vascular tissue", "Leaf guard cell", "Mesophyll", "Phloem parenchyma", 
        "Bundle sheath", "Companion cell", "Hydathodes", "Unknow", "Xylem", 
        "Leaf pavement cell", "S phase", "Shoot system epidermis", "Shoot apical meristem", 
        "G2/M phase", "Sieve element", "Phloem", "Stress response", "Meristematic cell"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        # celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1


        # sc.pp.normalize_total(adata_test, target_sum=1e4)
        # # print(adata.X)
        # sc.pp.log1p(adata_test, base=2)


        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        if issparse(adata_test.X):
            # 如果是稀疏格式，转换为密集格式的 NumPy 数组
            all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
        else:
            # 如果已经是密集格式，直接使用 adata.X
            all_counts_test = adata_test.X
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
    elif args.data_name == "flower":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        celltypes = ['Shoot system epidermis', 'Cortex', 'G2/M phase', 'Vascular cambium', 
            'Flower meristem', 'S phase', 'Unknow', 'Mesophyll', 'Xylem', 'Phloem', 
            'Vegetative nuclei', 'Microspore nuclei', 'Sperm nuclei', 'Generative nuclei', 
            'Contaminating nuclei', 'Transitory']
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1


        # sc.pp.normalize_total(adata_test, target_sum=1e4)
        # # print(adata.X)
        # sc.pp.log1p(adata_test, base=2)


        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        if issparse(adata_test.X):
            # 如果是稀疏格式，转换为密集格式的 NumPy 数组
            all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
        else:
            # 如果已经是密集格式，直接使用 adata.X
            all_counts_test = adata_test.X
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
        
    elif args.data_name == "cotyledons":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        celltypes = ['Spongy mesophyll', 'Palisade mesophyll', 'Leaf guard cell', 
            'Stress response', 'Leaf pavement cell', 'Phloem', 'Xylem', 'Bundle sheath', 
            'S phase', 'Companion cell', 'Phloem parenchyma', 'Unknow']
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1

        # sc.pp.normalize_total(adata_test, target_sum=1e4)
        # # print(adata.X)
        # sc.pp.log1p(adata_test, base=2)


        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        if issparse(adata_test.X):
            # 如果是稀疏格式，转换为密集格式的 NumPy 数组
            all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
        else:
            # 如果已经是密集格式，直接使用 adata.X
            all_counts_test = adata_test.X
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

    elif args.data_name == "seed":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        celltypes=[
        "Seed.coat", "Cotyledons", "Embryo", "Unknown", "Chalazal.seed.coat._CZSC_",
        "Inner.integument._ii_", "Cotyledons._abaxial.side_", "Provasculature,.QC",
        "Outer.integument._oi_", "Endothelium._ii1_", "Endosperm._MCE_", "Vascular.tissue",
        "Endosperm", "Endosperm._PEN_", "Cotyledon.tip", "Chalazal.endosperm._CZE_",
        "Cotyledon.tip.and.vascular.initials", "Cotyledons.epidermis", "Provascular.tissue",
        "Seed.coat.epidermis", "Central.cell", "Chalazal.seed.coat._and.vasculature_",
        "Chalazal.region._chalazal.seed.coat,vascular_", "Procambium,.Stele,.RAM",
        "Unknown_Embryo._SAM_.Seed.Coat", "Cotyledons._cells.around.SAM_", "Endosperm_MCE_",
        "Endosperm_PEN_", "Sperm.cell.and.vegetative.cell", "Chalazal.region",
        "Synergid.cell", "Suspensor", "Egg.cell", "Zygote,.Basal.cell"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1


        # sc.pp.normalize_total(adata_test, target_sum=1e4)
        # # print(adata.X)
        # sc.pp.log1p(adata_test, base=2)


        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels_test = np.array(celltypes_labels_test)
        batch_ids_test = adata_test.obs["batch_id"].tolist()
        batch_ids_test = np.array(batch_ids_test)
        if issparse(adata_test.X):
            # 如果是稀疏格式，转换为密集格式的 NumPy 数组
            all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
        else:
            # 如果已经是密集格式，直接使用 adata.X
            all_counts_test = adata_test.X
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
        print('data_name不正确')


    return test_data_pt,adata_test_raw
    # return adata_test_raw

# class Preprocessor_only_bin:
#     """
#     Prepare data into training, valid and test split. Normalize raw expression
#     values, binning or using other transform into the preset model input format.
#     """

#     def __init__(
#         self,
#         use_key: Optional[str] = None,
#         binning: Optional[int] = None,
#         result_binned_key: str = "X_binned",
#     ):
  
#         self.use_key = use_key

#         self.binning = binning
#         self.result_binned_key = result_binned_key

    

#     def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
   
#         key_to_process = self.use_key
#         # preliminary checks, will use later
#         if key_to_process == "X":
#             key_to_process = None  # the following scanpy apis use arg None to use X
#         # is_logged = self.check_logged(adata, obs_key=key_to_process)
#         def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
#             """
#             Digitize the data into bins. This method spreads data uniformly when bins
#             have same values.

#             Args:

#             x (:class:`np.ndarray`):
#                 The data to digitize.
#             bins (:class:`np.ndarray`):
#                 The bins to use for digitization, in increasing order.
#             side (:class:`str`, optional):
#                 The side to use for digitization. If "one", the left side is used. If
#                 "both", the left and right side are used. Default to "one".

#             Returns:

#             :class:`np.ndarray`:
#                 The digitized data.
#             """
#             assert x.ndim == 1 and bins.ndim == 1
#             left_digits = np.digitize(x, bins)
#             # 将大于7.0的值归到最后一个箱子
#             # last_bin_index = len(bins)
#             # left_digits[x >= 7.0] = last_bin_index
#             if side == "one":
#                 return left_digits
#             right_difits = np.digitize(x, bins, right=True)
#             # 同样处理右侧分箱，将大于7.0的值归到最后一个箱子
#             # right_digits[x >= 7.0] = last_bin_index
#             rands = np.random.rand(len(x))  # uniform random numbers
#             digits = rands * (right_difits - left_digits) + left_digits
#             digits = np.ceil(digits).astype(np.int32)    #np.int64
#             return digits


#         def binning(
#             row: Union[np.ndarray, torch.Tensor], n_bins: int
#         ) -> Union[np.ndarray, torch.Tensor]:
#             """Binning the row into n_bins."""
#             dtype = row.dtype
#             return_np = False if isinstance(row, torch.Tensor) else True
#             row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
#             # TODO: use torch.quantile and torch.bucketize
#             if row.max() == 0:
#                 logger.warning(
#                     "The input data contains row of zeros. Please make sure this is expected."
#                 )
#                 return (
#                     np.zeros_like(row, dtype=dtype)
#                     if return_np
#                     else torch.zeros_like(row, dtype=dtype)
#                 )
#             if row.min() <= 0:
#                 non_zero_ids = row.nonzero()
#                 non_zero_row = row[non_zero_ids]
#                 bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
#                 non_zero_digits = _digitize(non_zero_row, bins)
#                 binned_row = np.zeros_like(row, dtype=np.int32)  #np.int64
#                 binned_row[non_zero_ids] = non_zero_digits
#             else:
#                 bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
#                 binned_row = _digitize(row, bins)
#             return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)



#         # step 6: binning
#         if self.binning:
#             logger.info("Binning data ...")
#             if not isinstance(self.binning, int):
#                 raise ValueError(
#                     "Binning arg must be an integer, but got {}.".format(self.binning)
#                 )
#             n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
#             binned_rows = []
#             bin_edges = []
#             layer_data = _get_obs_rep(adata, layer=key_to_process)  #X_log1p
#             layer_data = layer_data.A if issparse(layer_data) else layer_data
#             if layer_data.min() < 0:
#                 raise ValueError(
#                     f"Assuming non-negative data, but got min value {layer_data.min()}."
#                 )
           
#             # fixed_bins=np.array([0.01,0.1,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.8,1.9,2.0,
#             # 2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,
#             # 4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.7,6,6.5,7])
           
#             for row in layer_data:
#                 if row.max() == 0:
#                     logger.warning(
#                         "The input data contains all zero rows. Please make sure "
#                         "this is expected. You can use the `filter_cell_by_counts` "
#                         "arg to filter out all zero rows."
#                     )
#                     binned_rows.append(np.zeros_like(row, dtype=np.int32))  #np.int64
#                     bin_edges.append(np.array([0] * n_bins))
#                     continue
#                 non_zero_ids = row.nonzero() #找到非零元素的索引和值，计算非零值的分箱边界。
#                 non_zero_row = row[non_zero_ids]
                
#                 bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1)) #使用 np.quantile 计算分箱边界。
#                 # bins=fixed_bins
               
#                 # bins = np.sort(np.unique(bins))
#                 # NOTE: comment this line for now, since this will make the each category
#                 # has different relative meaning across datasets
#                 non_zero_digits = _digitize(non_zero_row, bins)
#                 assert non_zero_digits.min() >= 1
#                 assert non_zero_digits.max() <= len(bins)  #n_bins - 1
#                 binned_row = np.zeros_like(row, dtype=np.int32)  #np.int64
#                 binned_row[non_zero_ids] = non_zero_digits
#                 binned_rows.append(binned_row)
#                 bin_edges.append(np.concatenate([[0], bins]))
#             adata.layers[self.result_binned_key] = np.stack(binned_rows)   # result_binned_key X_binned
#             # print('\n\nadata.layers[X_binned].A\n',adata.layers['X_binned'].A)
#             adata.obsm["bin_edges"] = np.stack(bin_edges)




def cell_annotation_test_dataset(testdata_path,args,only_test,**kwargs):
    
    if args.data_name == 'root':
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        # celltypes = ['G2/M phase', 'Root cortex', 'Root cap', 'S phase', 'Root hair', 
        #     'Lateral root cap', 'Non-hair', 'Columella root cap', 'Phloem', 'Root endodermis', 
        #     'Xylem', 'G1/G0 phase', 'Xylem pole pericycle', 'Root procambium', 'Phloem pole pericycle', 
        #     'Companion cell', 'Metaxylem', 'Sieve element', 'Protoxylem', 'Root stele', 'Pericycle', 
        #     'Meristematic cell', 'Stem cell niche', 'Root epidermis', 'Phloem/Pericycle', 
        #     'Phloem parenchyma', 'Lateral root primordia', 'Unknow']
        celltypes = [
        "Root hair", "Pericycle", "Lateral root cap", "Non-hair", "Root procambium",
        "Root endodermis", "Root cortex", "Xylem", "Columella", "Phloem",
        "Quiescent Center", "Unknow", "Phloem pole pericycle", "Xylem pole",
        "Root epidermis", "Root stele", "Xylem pole pericycle",
        "Lateral root endodermis", "Lateral root primordia", "G2/M phase",
        "Root cap", "S phase", "Columella root cap", "G1/G0 phase",
        "Companion cell", "Metaxylem", "Sieve element", "Protoxylem",
        "Meristematic cell", "Stem cell niche", "Phloem/Pericycle",
        "Phloem parenchyma"
        ]

        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1
    
    elif args.data_name == "root_152766":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
        #     'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
        #     'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
        #     'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
        #     'Leaf epidermis', 'Stress response', 'Meristematic cell']
        celltypes = [
        "Phloem pole pericycle", "Root cortex", "Unknow", "Root hair", "Columella", 
        "Non-hair", "Xylem pole", "Root endodermis", "Quiescent Center", "Phloem", "Xylem"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        # celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1
    
    elif args.data_name == "leaf":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        # celltypes = ['Mesophyll', 'Xylem', 'Leaf pavement cell', 'Leaf guard cell', 
        #     'Phloem parenchyma', 'S phase', 'Companion cell', 'Shoot system epidermis', 
        #     'Vascular tissue', 'Shoot apical meristem', 'G2/M phase', 'Guard cell', 
        #     'Unknow', 'Sieve element', 'Hydathodes', 'Phloem', 'Bundle sheath', 
        #     'Leaf epidermis', 'Stress response', 'Meristematic cell']
        celltypes = [
        "Leaf epidermis", "Palisade mesophyll cell", "Spongy mesophyll cell", 
        "Vascular tissue", "Leaf guard cell", "Mesophyll", "Phloem parenchyma", 
        "Bundle sheath", "Companion cell", "Hydathodes", "Unknow", "Xylem", 
        "Leaf pavement cell", "S phase", "Shoot system epidermis", "Shoot apical meristem", 
        "G2/M phase", "Sieve element", "Phloem", "Stress response", "Meristematic cell"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        # celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1

    elif args.data_name == "flower":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        celltypes = ['Shoot system epidermis', 'Cortex', 'G2/M phase', 'Vascular cambium', 
            'Flower meristem', 'S phase', 'Unknow', 'Mesophyll', 'Xylem', 'Phloem', 
            'Vegetative nuclei', 'Microspore nuclei', 'Sperm nuclei', 'Generative nuclei', 
            'Contaminating nuclei', 'Transitory']
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1
    
    elif args.data_name == "cotyledons":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        celltypes = ['Spongy mesophyll', 'Palisade mesophyll', 'Leaf guard cell', 
            'Stress response', 'Leaf pavement cell', 'Phloem', 'Xylem', 'Bundle sheath', 
            'S phase', 'Companion cell', 'Phloem parenchyma', 'Unknow']
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1
    
    elif args.data_name == "seed":
        # data_dir=Path(test_path)
        adata_test=anndata.read(testdata_path)
        # sc.pp.filter_cells(adata_test, min_genes=200)
        # adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        celltypes=[
        "Seed.coat", "Cotyledons", "Embryo", "Unknown", "Chalazal.seed.coat._CZSC_",
        "Inner.integument._ii_", "Cotyledons._abaxial.side_", "Provasculature,.QC",
        "Outer.integument._oi_", "Endothelium._ii1_", "Endosperm._MCE_", "Vascular.tissue",
        "Endosperm", "Endosperm._PEN_", "Cotyledon.tip", "Chalazal.endosperm._CZE_",
        "Cotyledon.tip.and.vascular.initials", "Cotyledons.epidermis", "Provascular.tissue",
        "Seed.coat.epidermis", "Central.cell", "Chalazal.seed.coat._and.vasculature_",
        "Chalazal.region._chalazal.seed.coat,vascular_", "Procambium,.Stele,.RAM",
        "Unknown_Embryo._SAM_.Seed.Coat", "Cotyledons._cells.around.SAM_", "Endosperm_MCE_",
        "Endosperm_PEN_", "Sperm.cell.and.vegetative.cell", "Chalazal.region",
        "Synergid.cell", "Suspensor", "Egg.cell", "Zygote,.Basal.cell"
        ]
        num_types = len(np.unique(celltypes))
        id2type = dict(enumerate(celltypes))
        adata_test.obs["celltype"] = pd.Categorical(adata_test.obs["Celltype"], categories=celltypes, ordered=False)
        # 将 "Celltype" 列转换为类别类型，并获取类别编码
        celltype_id_labels = adata_test.obs["celltype"].cat.codes.values
        adata_test.obs["celltype_id"] = celltype_id_labels
        adata_test_raw=adata_test.copy()
        adata_test.obs["batch_id"] = 1

    else:
        print('data_name不正确')


    logger=kwargs['logger']
    vocab=kwargs['vocab']
    is_master=kwargs['is_master']
    mask_value=kwargs['mask_value']
    pad_value = kwargs['pad_value']
    pad_token = kwargs['pad_token']

    ## only retain the gene that appears in vocab
    adata_test, _ = filter_gene(vocab=vocab, adata=adata_test, is_master=is_master, logger=logger)
    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor_only_bin(
        use_key="X",  # the key in adata.layers to use as raw data
        # filter_gene_by_counts=filter_gene_by_counts,  #  false step 1
        # filter_cell_by_counts=False,  # step 2
        # normalize_total= None  #1e4,  # 3. whether to normalize the raw data and to what sum
        # result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        # log1p=False,  # 4. whether to log1p the normalized data
        # result_log1p_key="X_log1p",
        # subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        # hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata_test, batch_key=None)

    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[args.input_style]
   #选择的X_binned
    all_counts_test = (
        adata_test.layers[input_layer_key].A
        if issparse(adata_test.layers[input_layer_key])
        else adata_test.layers[input_layer_key]
    )   

    genes = adata_test.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    celltypes_labels_test = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels_test = np.array(celltypes_labels_test)
    batch_ids_test = adata_test.obs["batch_id"].tolist()
    batch_ids_test = np.array(batch_ids_test)
    # if issparse(adata_test.X):
    #     # 如果是稀疏格式，转换为密集格式的 NumPy 数组
    #     all_counts_test = adata_test.X.toarray()  # 或者使用 adata.X.A
    # else:
    #     # 如果已经是密集格式，直接使用 adata.X
    #     all_counts_test = adata_test.X
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
    if is_master:
        logger.info(
            f"test set number of samples: {tokenized_test['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_test['genes'].shape[1]}"
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
    if only_test:
        num_batch_types=1
        return test_data_pt,num_batch_types,celltypes,id2type,num_types,adata_test_raw

    else:
        return test_data_pt,adata_test_raw