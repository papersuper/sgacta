import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import mode
import scanpy as sc
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import sklearn
import seaborn as sns
import pandas as pd
import sys
from torch.utils.data import DataLoader,random_split
sys.path.insert(0, "../")
import hdf5plugin
import scgpt as scg
from sklearn.model_selection import train_test_split
# extra dependency for similarity search


output_dir = 'path/to/output'
model_dir = Path("/path/to/model")
adata = sc.read_h5ad('/path/to/refernce.h5ad')
test_adata = sc.read_h5ad('path/to/predict.h5ad')
cell_type_key = "Celltype"
gene_col = "features"
os.makedirs(output_dir, exist_ok=True)

adata.var['features']=adata.var_names
test_adata.var['features']=test_adata.var_names


sc.pp.filter_cells(adata, min_genes=200)  # 保留表达至少200个基因的细胞
sc.pp.filter_genes(adata, min_cells=3)  # 保留在至少3个细胞中表达的基因
sc.pp.normalize_total(adata, target_sum=1e4)# 归一化到每个细胞的总计数为1e4
sc.pp.log1p(adata,base=2)# 对数转换

sc.pp.filter_cells(test_adata, min_genes=200)  # 保留表达至少200个基因的细胞
sc.pp.filter_genes(test_adata, min_cells=3)  # 保留在至少3个细胞中表达的基因
sc.pp.normalize_total(test_adata, target_sum=1e4)# 归一化到每个细胞的总计数为1e4
sc.pp.log1p(test_adata,base=2)# 对数转换

ref_embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)
# print('ref_embed_adata',ref_embed_adata.shape,'\n',ref_embed_adata)
test_embed_adata = scg.tasks.embed_data(
    test_adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)



'''
We run the reference mapping using cell-level majority voting. 
You may adjust the k parameter to control the number of nearest neighbors
 to consider for voting.
'''
def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims

def get_similar_vectors(vector, ref, top_k=10):
        # sims = cos_sim(vector, ref)
        sims = l2_sim(vector, ref)
        
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        return top_k_idx, sims[top_k_idx]

ref_cell_embeddings = ref_embed_adata.obsm["X_scGPT"]
test_emebd = test_embed_adata.obsm["X_scGPT"]
idx_list = [i for i in range(test_emebd.shape[0])]
preds = []
k = 10  # number of neighbors


from sklearn import metrics
# 获取用户输入
user_choice = input("请选择方法 (输入 'faiss' 或 'similar'): ").strip().lower()

# 根据用户选择执行不同的逻辑
if user_choice == "faiss":
    # 如果选择 faiss
    # Declaring index, using most of the default parameters from
    index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
    index.add(ref_cell_embeddings)
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    distances, labels = index.search(test_emebd, k)
    for k in idx_list:
        idx = labels[k]
        pred = ref_embed_adata.obs[cell_type_key][idx].value_counts()
        preds.append(pred.index[0])

elif user_choice == "similar":
    # 如果选择 similar
    idx_list = [i for i in range(test_emebd.shape[0])]
    preds = []
    for k in idx_list:
        idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings, k)
        pred = ref_embed_adata.obs[cell_type_key][idx].value_counts()
        preds.append(pred.index[0])

else:
    raise ValueError("无效的选择！请输入 'faiss' 或 'similar'。")

test_adata.obs['predicted_cell_type'] = preds
output_path_file = os.path.join(output_dir, 'predict_cell_type.h5ad')
test_adata.write_h5ad(output_path_file)
print("预测细胞类型已经保存到predict_cell_type.h5ad文件中")
print('predict:',preds)

# 计算准确率
# gt = test_adata.obs[cell_type_key].to_numpy()
# res_dict = {
#     "accuracy": accuracy_score(gt, preds),
#     "precision": precision_score(gt, preds, average="macro"),
#     "recall": recall_score(gt, preds, average="macro"),
#     "macro_f1": f1_score(gt, preds, average="macro"),
# }
# print('Evaluate the performance:',res_dict)


