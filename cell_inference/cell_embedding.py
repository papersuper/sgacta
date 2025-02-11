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
adata = sc.read_h5ad('/path/to/embedding.h5ad')

cell_type_key = "Celltype"
gene_col = "features"
os.makedirs(output_dir, exist_ok=True)

adata.var['features']=adata.var_names

sc.pp.filter_cells(adata, min_genes=200)  # 保留表达至少200个基因的细胞
sc.pp.filter_genes(adata, min_cells=3)  # 保留在至少3个细胞中表达的基因
sc.pp.normalize_total(adata, target_sum=1e4)# 归一化到每个细胞的总计数为1e4
sc.pp.log1p(adata,base=2)# 对数转换

embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)

'''
We run the reference mapping using cell-level majority voting. 
You may adjust the k parameter to control the number of nearest neighbors
 to consider for voting.
'''


cell_embeddings = embed_adata.obsm["X_scGPT"]

# 获取用户输入
user_choice = input("请选择保存cell_embedding的文件类型 (输入 '1'代表.csv; '2'代表.h5ad); '3'代表.tsv ").strip()

# 根据用户选择执行不同的逻辑
if user_choice == "1":
    # 将嵌入信息转换为 DataFrame
    cell_embeddings_df = pd.DataFrame(cell_embeddings , index=embed_adata.obs_names)
    # 保存为 CSV 文件
    cell_embeddings_df.to_csv(f"{output_dir}/cell_embeddings.csv")
    print("文件已经保存到cell_embeddings.csv中")
elif user_choice == "2":
    # 将嵌入信息添加到 adata 对象中
    adata.obsm["embed"] = cell_embeddings
    # 保存为 .h5ad 文件
    embed_adata.write(f"{output_dir}/embed_adata.h5ad")
    print("文件已经保存到embed_adata.h5ad中")
elif user_choice == "3":
    # 将嵌入信息转换为 DataFrame
    cell_embeddings_df = pd.DataFrame(cell_embeddings , index=embed_adata.obs_names)
    # 保存为 TSV 文件
    cell_embeddings_df.to_csv(f"{output_dir}/cell_embeddings.tsv", sep="\t")
    print("文件已经保存到cell_embeddings.tsv中")
else:
    raise ValueError("无效的选择！请输入 '1' 或 '2' 或 '3'。")



# 计算准确率
# gt = test_adata.obs[cell_type_key].to_numpy()
# res_dict = {
#     "accuracy": accuracy_score(gt, preds),
#     "precision": precision_score(gt, preds, average="macro"),
#     "recall": recall_score(gt, preds, average="macro"),
#     "macro_f1": f1_score(gt, preds, average="macro"),
# }
# print('Evaluate the performance:',res_dict)


