from typing import Dict, Optional, Union

import numpy as np
import torch
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData

from scgpt import logger


# class Preprocessor:
#     """
#     Prepare data into training, valid and test split. Normalize raw expression
#     values, binning or using other transform into the preset model input format.
#     """

#     def __init__(
#         self,
#         use_key: Optional[str] = None,
#         filter_gene_by_counts: Union[int, bool] = False,
#         filter_cell_by_counts: Union[int, bool] = False,
#         normalize_total: Union[float, bool] = 1e4,
#         result_normed_key: Optional[str] = "X_normed",
#         log1p: bool = False,
#         result_log1p_key: str = "X_log1p",
#         subset_hvg: Union[int, bool] = False,
#         hvg_use_key: Optional[str] = None,
#         hvg_flavor: str = "seurat_v3",
#         binning: Optional[int] = None,
#         result_binned_key: str = "X_binned",
#     ):
#         r"""
#         Set up the preprocessor, use the args to config the workflow steps.

#         Args:

#         use_key (:class:`str`, optional):
#             The key of :class:`~anndata.AnnData` to use for preprocessing.
#         filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
#             Whther to filter genes by counts, if :class:`int`, filter genes with counts
#         filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
#             Whther to filter cells by counts, if :class:`int`, filter cells with counts
#         normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
#             Whether to normalize the total counts of each cell to a specific value.
#         result_normed_key (:class:`str`, default: ``"X_normed"``):
#             The key of :class:`~anndata.AnnData` to store the normalized data. If
#             :class:`None`, will use normed data to replce the :attr:`use_key`.
#         log1p (:class:`bool`, default: ``True``):
#             Whether to apply log1p transform to the normalized data.
#         result_log1p_key (:class:`str`, default: ``"X_log1p"``):
#             The key of :class:`~anndata.AnnData` to store the log1p transformed data.
#         subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
#             Whether to subset highly variable genes.
#         hvg_use_key (:class:`str`, optional):
#             The key of :class:`~anndata.AnnData` to use for calculating highly variable
#             genes. If :class:`None`, will use :attr:`adata.X`.
#         hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
#             The flavor of highly variable genes selection. See
#             :func:`scanpy.pp.highly_variable_genes` for more details.
#         binning (:class:`int`, optional):
#             Whether to bin the data into discrete values of number of bins provided.
#         result_binned_key (:class:`str`, default: ``"X_binned"``):
#             The key of :class:`~anndata.AnnData` to store the binned data.
#         """
#         self.use_key = use_key
#         self.filter_gene_by_counts = filter_gene_by_counts
#         self.filter_cell_by_counts = filter_cell_by_counts
#         self.normalize_total = normalize_total
#         self.result_normed_key = result_normed_key
#         self.log1p = log1p
#         self.result_log1p_key = result_log1p_key
#         self.subset_hvg = subset_hvg
#         self.hvg_use_key = hvg_use_key
#         self.hvg_flavor = hvg_flavor
#         self.binning = binning
#         self.result_binned_key = result_binned_key

#     def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
#         """
#         format controls the different input value wrapping, including categorical
#         binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

#         Args:

#         adata (:class:`AnnData`):
#             The :class:`AnnData` object to preprocess.
#         batch_key (:class:`str`, optional):
#             The key of :class:`AnnData.obs` to use for batch information. This arg
#             is used in the highly variable gene selection step.
#         """
#         key_to_process = self.use_key
#         # preliminary checks, will use later
#         if key_to_process == "X":
#             key_to_process = None  # the following scanpy apis use arg None to use X
#         is_logged = self.check_logged(adata, obs_key=key_to_process)

#         # step 1: filter genes
#         if self.filter_gene_by_counts:
#             logger.info("Filtering genes by counts ...")
#             sc.pp.filter_genes(
#                 adata,
#                 min_counts=self.filter_gene_by_counts
#                 if isinstance(self.filter_gene_by_counts, int)
#                 else None,
#             )
#         # print('filter_gene_by_counts',adata.X)
#         # step 2: filter cells
#         if (
#             isinstance(self.filter_cell_by_counts, int)
#             and self.filter_cell_by_counts > 0
#         ):
#             logger.info("Filtering cells by counts ...")
#             sc.pp.filter_cells(
#                 adata,
#                 min_counts=self.filter_cell_by_counts
#                 if isinstance(self.filter_cell_by_counts, int)
#                 else None,
#             )
#         # print('filter_cells',adata.X)
#         # step 3: normalize total
#         if self.normalize_total:
#             logger.info("Normalizing total counts ...")
#             normed_ = sc.pp.normalize_total(
#                 adata,
#                 target_sum=None,
#                 # target_sum=self.normalize_total,
#                 # if isinstance(self.normalize_total, float)   #归一化后每个细胞的目标和。如果self.normalize_total是一个浮点数，它将使用这个值作为目标和。如果不是浮点数，target_sum将被设置为None，函数将为所有细胞计算一个共同的总和。
#                 # else None,
#                 layer=key_to_process,   #none,将归一化主矩阵（adata.X）
#                 inplace=False,     #如果为True，它将原地修改AnnData对象。如果为False，它将返回一个包含归一化数据的新矩阵。
#             )["X"]

#             # print("\nnormed_",normed_,'\n\n')
#             key_to_process = self.result_normed_key or key_to_process   #X_normed
#             _set_obs_rep(adata, normed_, layer=key_to_process)      #X_normed

#         # step 4: log1p
#         if self.log1p:
#             logger.info("Log1p transforming ...")
#             if is_logged:
#                 logger.warning(
#                     "The input data seems to be already log1p transformed. "
#                     "Set `log1p=False` to avoid double log1p transform."
#                 )
#             if self.result_log1p_key:   #X_log1p
#                 _set_obs_rep(
#                     adata,
#                     _get_obs_rep(adata, layer=key_to_process),
#                     layer=self.result_log1p_key,
#                 )
#                 key_to_process = self.result_log1p_key   #X_log1p
#             sc.pp.log1p(adata, layer=key_to_process,base=2)
#         # print('\nlog1p',adata.X)        
#         # step 5: subset hvg
#         if self.subset_hvg:
#             logger.info("Subsetting highly variable genes ...")
#             if batch_key is None:
#                 logger.warning(
#                     "No batch_key is provided, will use all cells for HVG selection."
#                 )
#             sc.pp.highly_variable_genes(
#                 adata,
#                 layer=self.hvg_use_key,
#                 n_top_genes=self.subset_hvg
#                 if isinstance(self.subset_hvg, int)
#                 else None,
#                 batch_key=batch_key,
#                 flavor=self.hvg_flavor,
#                 subset=True,
#             )

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

#     def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
#         """
#         Check if the data is already log1p transformed.

#         Args:

#         adata (:class:`AnnData`):
#             The :class:`AnnData` object to preprocess.
#         obs_key (:class:`str`, optional):
#             The key of :class:`AnnData.obs` to use for batch information. This arg
#             is used in the highly variable gene selection step.
#         """
#         data = _get_obs_rep(adata, layer=obs_key)
#         max_, min_ = data.max(), data.min()
#         if max_ > 30:
#             return False
#         if min_ < 0:
#             return False

#         non_zero_min = data[data > 0].min()
#         if non_zero_min >= 1:
#             return False

#         return True


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
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

    # 将大于7.0的值归到最后一个箱子
    # last_bin_index = len(bins)
    # left_digits[x >= 7.0] = last_bin_index


    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    # 同样处理右侧分箱，将大于7.0的值归到最后一个箱子
    # right_digits[x >= 7.0] = last_bin_index

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int32)    #np.int64
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.max() == 0:
        logger.warning(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int32)  #np.int64
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)





class Preprocessor_only_bin:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
  
        self.use_key = use_key

        self.binning = binning
        self.result_binned_key = result_binned_key

    

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
   
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        # is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 6: binning
        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)  #X_log1p
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
           
            fixed_bins=np.array([0.01,0.1,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.8,1.9,2.0,
            2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,
            4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,6,6.5,7])
           
            for row in layer_data:
                if row.max() == 0:
                    logger.warning(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int32))  #np.int64
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero() #找到非零元素的索引和值，计算非零值的分箱边界。
                non_zero_row = row[non_zero_ids]
                
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1)) #使用 np.quantile 计算分箱边界。
                # bins=fixed_bins
               
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= len(bins)  #n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int32)  #np.int64
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)   # result_binned_key X_binned
            # print('\n\nadata.layers[X_binned].A\n',adata.layers['X_binned'].A)
            adata.obsm["bin_edges"] = np.stack(bin_edges)







class Preprocessor_annimal_normalized:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        # normalize_total: Union[float, bool] = 1e4,
        # result_normed_key: Optional[str] = "X_normed",
        # log1p: bool = False,
        # result_log1p_key: str = "X_log1p",
        # subset_hvg: Union[int, bool] = False,
        # hvg_use_key: Optional[str] = None,
        # hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        # self.normalize_total = normalize_total
        # self.result_normed_key = result_normed_key
        # self.log1p = log1p
        # self.result_log1p_key = result_log1p_key
        # self.subset_hvg = subset_hvg
        # self.hvg_use_key = hvg_use_key
        # self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        # is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )
        # print('filter_gene_by_counts',adata.X)
        # step 2: filter cells
        if (
            isinstance(self.filter_cell_by_counts, int)
            and self.filter_cell_by_counts > 0
        ):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )
        # print('filter_cells',adata.X)
   

        # step 6: binning
        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)  #X_log1p
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                print(layer_data)
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
           
            # fixed_bins=np.array([0.01,0.1,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.8,1.9,2.0,
            # 2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,
            # 4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.7,6,6.5,7])
           
            for row in layer_data:
                if row.max() == 0:
                    logger.warning(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int32))  #np.int64
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero() #找到非零元素的索引和值，计算非零值的分箱边界。
                non_zero_row = row[non_zero_ids]
                
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1)) #使用 np.quantile 计算分箱边界。
                # bins=fixed_bins
               
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= len(bins)  #n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int32)  #np.int64
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)   # result_binned_key X_binned
            # print('\n\nadata.layers[X_binned].A\n',adata.layers['X_binned'].A)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True




























