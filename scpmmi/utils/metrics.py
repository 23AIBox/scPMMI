r"""
Performance evaluation metrics
"""
import sys
sys.path.append("..")
from typing import Tuple
from threadpoolctl import threadpool_limits
import scipy
import cupy as cp
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

from .metric_utils import (
    avg_silhouette_width_cell_type, mean_average_precision, neighbor_consistency,
    avg_silhouette_width_omics, seurat_alignment_score, graph_connectivity,
    maxARI, NMI, F1_silhouette, entropy_mixing_score
)


def compute_matching_score(embeddings_1, embeddings_2):
    """
    calculate matching Score
    
    parameters:
    embeddings_1 (np.ndarray): modality 1's embeddings, shape: (N, d), N is the number of samples, d is dimension
    embeddings_2 (np.ndarray): modality 2's embeddings, shape: (N, d), N is the number of samples, d is dimension
    
    return:
    matching_score (float): Matching Score
    """
    N = embeddings_1.shape[0]
    
    # calculate the nearest neighbor of modality 1 in the modality 2
    knn_idx_1 = np.argmin(np.linalg.norm(embeddings_1[:, None] - embeddings_2, axis=-1), axis=-1)
    # calculate the nearest neighbor of modality 2 in the modality 1
    knn_idx_2 = np.argmin(np.linalg.norm(embeddings_2[:, None] - embeddings_1, axis=-1), axis=-1)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            knn_ij = {knn_idx_1[i]}
            knn_jj = {j}
            knn_ji = {knn_idx_2[j]}
            knn_ii = {i}
            M[i, j] = len((knn_ij & knn_jj) | (knn_ji & knn_ii)) / len((knn_ij | knn_jj) | (knn_ji | knn_ii))

    # normalize matching matrix
    M_norm = M / (M.sum(axis=1, keepdims=True) + M.sum(axis=0, keepdims=True) - M)

    # calculate matching Score
    matching_score = (1 / N) * np.sum(np.diagonal(M_norm))
    return matching_score

def compute_foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = ((d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)).mean()
    foscttm_y = ((d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)).mean()
    return (foscttm_x + foscttm_y) / 2

def calculate_metrics(
        adata: sc.AnnData, 
        cell_type,
        cluster_labels,
        num_threads=16
):
    label_encoder = LabelEncoder()
    encoded_original_labels = label_encoder.fit_transform(cell_type)
    
    # biology conservation
    with threadpool_limits(limits=num_threads):
        ari_score = metrics.adjusted_rand_score(encoded_original_labels, cluster_labels)
        nmi_score = metrics.normalized_mutual_info_score(encoded_original_labels, cluster_labels)
        MAP = mean_average_precision(adata.X, cell_type)
        ASW_celltype = avg_silhouette_width_cell_type(adata.X, cell_type)
        
        # omics mixing
        BEM_score = entropy_mixing_score(adata.obsm['X_umap'], adata.obs['modality'])
        SAS = seurat_alignment_score(adata.obsm['X_umap'], cell_type)
        GC = graph_connectivity(adata.X, cell_type)
        ASW_omics = avg_silhouette_width_omics(adata.X, adata.obs['modality'].values, cell_type)

    return {
        "ARI": float(ari_score),
        "NMI": float(nmi_score),
        "Mean average precision": float(MAP),
        "ASW_celltype": float(ASW_celltype),
        
        "omics entropy mixing score": float(BEM_score),
        "Seurat alignment score (omics)": float(SAS),
        "Graph connectivity": float(GC),
        "ASW_omics": float(ASW_omics),
    }

def omics_mixing(
        adata: sc.AnnData, 
        cell_type,
        modality,
        num_threads=16
):
    
    with threadpool_limits(limits=num_threads):
        BEM_score = entropy_mixing_score(adata.obsm['X_umap'], adata.obs['modality'])
        SAS = seurat_alignment_score(adata.obsm['X_umap'], modality)
        GC = graph_connectivity(adata.obsm['X_umap'], cell_type)
        ASW_omics = avg_silhouette_width_omics(adata.obsm['X_umap'], modality, cell_type)

    return {
        "omics entropy mixing score": float(BEM_score),
        "Seurat alignment score": float(SAS),
        "Graph connectivity": float(GC),
        "ASW_omics": float(ASW_omics)
    }

def biology_conservation(
        adata: sc.AnnData, 
        cell_type,
        num_threads=16
):  
    # biology conservation
    with threadpool_limits(limits=num_threads):
        max_ari, max_res = maxARI(adata, cell_type="cell_type")
        sc.tl.leiden(adata, resolution=max_res, flavor="igraph", n_iterations=2)
        nmi_score = NMI(adata.obs['cell_type'].values, adata.obs['leiden'].values)
        # ari_score = metrics.adjusted_rand_score(encoded_original_labels, cluster_labels)
        # nmi_score = metrics.normalized_mutual_info_score(encoded_original_labels, cluster_labels)
        MAP = mean_average_precision(adata.X, cell_type)
        ASW_celltype = avg_silhouette_width_cell_type(adata.obsm['X_umap'], cell_type)

    return {
        "ARI": float(max_ari),
        "NMI": float(nmi_score),
        "Mean average precision": float(MAP),
        "ASW_celltype": float(ASW_celltype),
    }, max_res

def batch_correct(
        adata, 
        cell_type,
        batch,
        num_threads=16
):    
    with threadpool_limits(limits=num_threads):
        BEM_score = entropy_mixing_score(adata.obsm['X_umap'], batch)
        SAS = seurat_alignment_score(adata.obsm['X_umap'], batch)
        GC = graph_connectivity(adata.X, cell_type)
        ASW_batch = avg_silhouette_width_omics(adata.X, batch, cell_type)

    return {
        "Batch entropy mixing score": float(BEM_score),
        "Seurat alignment score": float(SAS),
        "Graph connectivity": float(GC),
        "bASW": float(ASW_batch)
    }    

def mean_F1_silhouette(embeds, cell_type, omics, device_id=0, chunk_size=1000):
    F1_sil_cuda = F1_silhouette(
        embeds, cell_type, omics, device_id=device_id, chunk_size=chunk_size
        )
    F1_sil_cuda_cpu = cp.asnumpy(F1_sil_cuda)
    return F1_sil_cuda_cpu.mean()

