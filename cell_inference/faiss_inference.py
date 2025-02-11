import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import sklearn
import warnings

sys.path.insert(0, "../")
import scgpt as scg
from build_atlas_index_faiss import load_index, vote
import faiss

model_dir = Path("path/to/model")
test_adata = sc.read_h5ad("path/to/h5ad")
index_dir = "path/to/faiss_files"
output_path = "path/to/output"
gene_col = "features"
cell_type_key = "Celltype"
k = 50
# print(test_adata.obs,test_adata.var)
#数据处理
sc.pp.filter_cells(test_adata, min_genes=200)  # 保留表达至少200个基因的细胞
sc.pp.filter_genes(test_adata, min_cells=3)  # 保留在至少3个细胞中表达的基因
sc.pp.normalize_total(test_adata, target_sum=1e4)# 归一化到每个细胞的总计数为1e4
sc.pp.log1p(test_adata,base=2)# 对数转换

test_embed_adata = scg.tasks.embed_data(
    test_adata,
    model_dir,
    gene_col=gene_col,
    obs_to_save=cell_type_key,  # optional arg, only for saving metainfo
    batch_size=64,
    return_new_adata=True,
)
print('test_embed_adata',test_embed_adata.shape,'\n',test_embed_adata)
use_gpu = faiss.get_num_gpus() > 0
index, meta_labels = load_index(
    index_dir=index_dir,
    use_config_file=False,
    use_gpu=use_gpu,
)
print(f"Loaded index with {index.ntotal} cells")

test_emebd = test_embed_adata.X
gt = test_adata.obs[cell_type_key].to_numpy()
# test with the first 100 cells
distances, idx = index.search(test_emebd, k)

predict_labels = meta_labels[idx]
# from scipy.stats import mode
from tqdm import tqdm

voting = []
for preds in tqdm(predict_labels):
    voting.append(vote(preds, return_prob=False)[0])
voting = np.array(voting)

# 将 voting 标注信息添加到 test_adata 的 obs 中
test_adata.obs['predicted_cell_type'] = voting
output_path_file = os.path.join(output_path, 'predict_cell_type.h5ad')
print("预测细胞类型已经保存到predict_cell_type.h5ad文件中")
# 保存到 h5ad 文件
test_adata.write_h5ad(output_path_file)

print(voting)
# print(f"Saved the annotated data to {output_path}")
# print('gt[:10]',gt[:10])   # Original labels in the query dataset, used for evaluation
# print('voting[:10]',voting[:10])  # Propagated CellXGene labels


























