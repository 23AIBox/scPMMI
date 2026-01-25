import pandas as pd
import numpy as np
import scipy.sparse as sp
import pyranges as pr
import scanpy as sc
from typing_extensions import Optional

from .. import logger

def align_to_common_genes(
    adata: sc.AnnData,
    gene_list,
    id_col: Optional[str] = None
) -> sc.AnnData:
    """
    Efficiently expand adata_rna features to match the full gene list in gene_vocab.
    Missing genes are filled with zeros. Fully vectorized for speed.

    Args:
        adata_rna (sc.AnnData): AnnData object containing RNA features.
        gene_list: the common gene list.
        id_col (str): Column in adata_rna.var to use as the gene identifier.

    Returns:
        sc.AnnData: New AnnData with features aligned to gene_vocab order.
    """
    # # Ensure var index is gene ID
    if id_col is not None:
        genes = list(adata.var[id_col])
    else:
        genes = list(adata.var.index) # type: ignore     


    # Map gene to index in adata_rna
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    # Build index mapping (-1 for missing genes)
    idx_map = np.array([gene_to_idx.get(g, -1) for g in gene_list])

    # Prepare empty sparse matrix
    n_cells = adata.n_obs
    n_genes = len(gene_list)
    new_X = sp.lil_matrix((n_cells, n_genes), dtype=adata.X.dtype) # type: ignore

    # Assign only for existing genes (vectorized)
    valid_mask = idx_map != -1
    existing_genes = np.array(gene_list)[valid_mask]
    logger.info(f"Number of matched genes: {len(existing_genes)}")

    new_X[:, valid_mask] = adata.X[:, idx_map[valid_mask]] # type: ignore

    # Convert to CSR for efficient downstream use
    new_X = new_X.tocsr()

    # Create new AnnData
    new_adata = sc.AnnData(X=new_X) # type: ignore
    new_adata.obs = adata.obs.copy()
    new_adata.var_names = gene_list
    new_adata.var['gene_name'] = gene_list

    return new_adata

def align_to_common_proteins(
    adata: sc.AnnData,
    protein_list,
    id_col: str = "feature_id"
) -> sc.AnnData:
    """
    Efficiently expand adata features to match a common list of proteins.
    Missing proteins are filled with zeros. Fully vectorized for performance.

    Args:
        adata (sc.AnnData): AnnData object containing protein features.
        protein_list (List[str]): The common list of protein identifiers.
        id_col (str): Column in `adata.var` to use as the protein identifier.

    Returns:
        sc.AnnData: New AnnData with protein features aligned to `protein_list` order.
    """
    # Create mapping from protein name to index in adata.var.index
    protein_to_idx = {protein: i for i, protein in enumerate(adata.var.index)}

    # Map desired proteins to their indices (use -1 for missing)
    idx_map = np.array([protein_to_idx.get(p, -1) for p in protein_list])

    # Prepare empty sparse matrix for the new data
    n_cells = adata.n_obs
    n_proteins = len(protein_list)
    new_X = sp.lil_matrix((n_cells, n_proteins), dtype=adata.X.dtype)  # type: ignore

    # Assign only matched proteins
    valid_mask = idx_map != -1
    matched_proteins = np.array(protein_list)[valid_mask]
    logger.info(f"Number of matched proteins: {len(matched_proteins)}")

    new_X[:, valid_mask] = adata.X[:, idx_map[valid_mask]]  # type: ignore

    # Convert to CSR format for efficiency
    new_X = new_X.tocsr()

    # Build new AnnData object
    new_adata = sc.AnnData(X=new_X)  # type: ignore
    new_adata.obs = adata.obs.copy()
    new_adata.var_names = protein_list
    new_adata.var.index.name = id_col

    return new_adata

def safe_int(x):
    try:
        return int(x)
    except ValueError:
        return int(float(x))

def parse_peak_ids(peak_ids: list[str]) -> pd.DataFrame:
    """
    Parse genomic peak IDs into a DataFrame with columns:
    Chromosome, Start, End.

    Supported formats:
        - "chr:start-end"
        - "chr-start-end"

    Args:
        peak_ids (list[str]): List of peak identifiers.

    Returns:
        pd.DataFrame: Parsed genomic intervals.
    """
    parsed = []
    for peak in peak_ids:
        if ":" in peak:  # Format: chr:start-end
            chrom, pos = peak.split(":")
            parts = pos.split("-")
            start, end = safe_int(parts[0]), safe_int(parts[1])
        elif "-" in peak:  # Format: chr-start-end
            parts = peak.split("-")
            if len(parts) != 3:
                raise ValueError(f"Invalid peak format: {peak}")
            chrom, start, end = parts[0], safe_int(parts[1]), safe_int(parts[2])
        elif " " in peak:
            parts = peak.split(" ")
            if len(parts) != 3:
                raise ValueError(f"Invalid peak format: {peak}")
            chrom, start, end = parts[0], safe_int(parts[1]), safe_int(parts[2])
        else:
            raise ValueError(f"Unrecognized peak format: {peak}")
        parsed.append((chrom, start, end))
    return pd.DataFrame(parsed, columns=["Chromosome", "Start", "End"])


def align_to_common_peaks(adata: sc.AnnData, peak_list: list[str]) -> sc.AnnData:
    """
    Align AnnData object to a given common peak set by genomic overlap.

    Args:
        adata (sc.AnnData): Input AnnData object with peak features in var_names.
        peak_list (list[str]): List of common peak identifiers.

    Returns:
        sc.AnnData: New AnnData aligned to the common peak set.
    """
    # Parse common and input peaks
    common_df = parse_peak_ids(peak_list).copy()
    # common_df["peak_id"] = common_df[["Chromosome", "Start", "End"]].astype(str).agg("-".join, axis=1)
    common_df["peak_id"] = (
        common_df["Chromosome"].astype(str) + ":" +
        common_df["Start"].astype(str) + "-" +
        common_df["End"].astype(str)
    )
    common_df["common_idx"] = np.arange(len(common_df))

    input_df = parse_peak_ids(adata.var_names.tolist()).copy()
    input_df["input_idx"] = np.arange(len(input_df))

    # Create PyRanges
    common_gr = pr.PyRanges(common_df[["Chromosome", "Start", "End"]])
    input_gr = pr.PyRanges(input_df[["Chromosome", "Start", "End"]])

    # Find overlaps between common peaks and input peaks
    overlap = common_gr.join(input_gr)
    overlap_df = overlap.df

    # PyRanges join returns two sets of coordinates; extract the left side (common peaks)
    overlap_df = overlap_df.rename(
        columns={
            "Chromosome": "Chromosome",
            "Start": "Start_common",
            "End": "End_common",
            "Start_b": "Start_input",
            "End_b": "End_input"
        }
    )

    # Merge to get indices
    overlap_df = overlap_df.merge(
        common_df,
        left_on=["Chromosome", "Start_common", "End_common"],
        right_on=["Chromosome", "Start", "End"],
        how="left"
    )
    overlap_df = overlap_df.merge(
        input_df,
        left_on=["Chromosome", "Start_input", "End_input"],
        right_on=["Chromosome", "Start", "End"],
        how="left",
        suffixes=("", "_inputdf")
    )

    overlap_df["overlap_length"] = (
        np.minimum(overlap_df["End_common"], overlap_df["End_input"]) -
        np.maximum(overlap_df["Start_common"], overlap_df["Start_input"])
    )
    overlap_df = overlap_df.loc[overlap_df["overlap_length"] > 200]
    # 选择覆盖长度最大的 input peak
    overlap_df = overlap_df.loc[overlap_df.groupby("common_idx")["overlap_length"].idxmax()]
    
    logger.info(f"Number of matched peaks: {len(overlap_df)}")

    # Build column index mapping: common_idx -> input_idx
    col_indices = np.full(len(common_df), -1, dtype=int)
    col_indices[overlap_df["common_idx"].values] = overlap_df["input_idx"].values

    # Create new sparse matrix
    n_cells = adata.n_obs
    n_peaks = len(common_df)
    new_X = sp.csr_matrix((n_cells, n_peaks), dtype=adata.X.dtype) # type: ignore

    valid_mask = col_indices != -1
    valid_cols = col_indices[valid_mask]
    selected_X = adata.X[:, valid_cols] # type: ignore

    new_X = new_X.tolil()
    new_X[:, valid_mask] = selected_X
    new_X = new_X.tocsr()

    # Build new AnnData
    new_adata = sc.AnnData(X=new_X) # type: ignore
    new_adata.obs = adata.obs.copy()
    new_adata.var_names = common_df["peak_id"].astype(str) # type: ignore

    return new_adata


def align_to_common_peaks_hubmap(adata: sc.AnnData, peak_list: list[str], min_overlap: int = 200, topk: int = 3) -> sc.AnnData:
    """
    Align AnnData object to a given common peak set by genomic overlap.
    For each common peak, select up to topk overlapping input peaks (with overlap_length > min_overlap),
    build multiple matrices, and return the weighted average.

    Args:
        adata (sc.AnnData): Input AnnData object with peak features in var_names.
        peak_list (list[str]): List of common peak identifiers.
        min_overlap (int): Minimum overlap length to keep.
        topk (int): Maximum number of overlapping peaks to keep per common peak.

    Returns:
        sc.AnnData: New AnnData aligned to the common peak set.
    """

    # Parse common and input peaks
    common_df = parse_peak_ids(peak_list).copy()
    common_df["peak_id"] = (
        common_df["Chromosome"].astype(str) + ":" +
        common_df["Start"].astype(str) + "-" +
        common_df["End"].astype(str)
    )
    common_df["common_idx"] = np.arange(len(common_df))

    input_df = parse_peak_ids(adata.var_names.tolist()).copy()
    input_df["input_idx"] = np.arange(len(input_df))

    # Create PyRanges
    common_gr = pr.PyRanges(common_df[["Chromosome", "Start", "End"]])
    input_gr = pr.PyRanges(input_df[["Chromosome", "Start", "End"]])

    # Find overlaps
    overlap = common_gr.join(input_gr)
    overlap_df = overlap.df.rename(
        columns={
            "Start": "Start_common",
            "End": "End_common",
            "Start_b": "Start_input",
            "End_b": "End_input"
        }
    )

    # Merge to get indices
    overlap_df = overlap_df.merge(
        common_df,
        left_on=["Chromosome", "Start_common", "End_common"],
        right_on=["Chromosome", "Start", "End"],
        how="left"
    ).merge(
        input_df,
        left_on=["Chromosome", "Start_input", "End_input"],
        right_on=["Chromosome", "Start", "End"],
        how="left",
        suffixes=("", "_inputdf")
    )

    # Compute overlap length
    overlap_df["overlap_length"] = (
        np.minimum(overlap_df["End_common"], overlap_df["End_input"]) -
        np.maximum(overlap_df["Start_common"], overlap_df["Start_input"])
    )

    # Keep only overlaps with length > min_overlap
    overlap_df = overlap_df.loc[overlap_df["overlap_length"] > min_overlap]

    logger.info(f"Number of matched peaks: {len(overlap_df)}")
    # Select top-k overlaps per common peak
    overlap_topk = (
        overlap_df.sort_values("overlap_length", ascending=False)
        .groupby("common_idx")
        .head(topk)
    )

    # Assign rank within each group
    overlap_topk["rank"] = (
        overlap_topk.groupby("common_idx")["overlap_length"]
        .rank("first", ascending=False).astype(int)
    )

    # Split into multiple overlap_dfs
    overlap_dfs = [
        overlap_topk[overlap_topk["rank"] == r].copy()
        for r in range(1, topk + 1)
    ]

    n_cells = adata.n_obs
    n_peaks = len(common_df)

    matrices = []
    for i, odf in enumerate(overlap_dfs, start=1):
        if len(odf) == 0:
            continue
        print(f"Building matrix for overlap_df{i}, peaks={len(odf)}")

        # Build column index mapping: common_idx -> input_idx
        col_indices = np.full(len(common_df), -1, dtype=int)
        col_indices[odf["common_idx"].values] = odf["input_idx"].values

        # Create new sparse matrix
        new_X = sp.csr_matrix((n_cells, n_peaks), dtype=adata.X.dtype)

        valid_mask = col_indices != -1
        valid_cols = col_indices[valid_mask]
        selected_X = adata.X[:, valid_cols]

        new_X = new_X.tolil()
        new_X[:, valid_mask] = selected_X
        new_X = new_X.tocsr()

        matrices.append(new_X)

    # -------------------------------
    # 自适应平均
    # -------------------------------
    if len(matrices) == 1:
        final_X = matrices[0]
    else:
        # sum 所有矩阵
        final_X = sum(matrices)

        # 每个 common_idx 出现了多少次
        counts = overlap_topk.groupby("common_idx").size().reindex(range(n_peaks), fill_value=0).values

        # 避免除零
        counts[counts == 0] = 1

        # 按列除以对应的 counts
        diag = sp.diags(1.0 / counts, offsets=0, shape=(n_peaks, n_peaks))
        final_X = final_X @ diag

    # # 加权平均 (这里简单平均)
    # if len(matrices) == 1:
    #     final_X = matrices[0]
    # else:
    #     final_X = sum(matrices) / len(matrices)

    # Build new AnnData
    new_adata = sc.AnnData(X=final_X)
    new_adata.obs = adata.obs.copy()
    new_adata.var_names = common_df["peak_id"].astype(str)

    return new_adata
