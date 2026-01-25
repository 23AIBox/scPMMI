import numpy as np


def hierarchical_bayesian_downsampling(
    X, 
    min_total_count=1000, 
    p_gamma=0.4, 
    alpha=2, 
    beta=2
):
    """
    Vectorized Hierarchical Bayesian down-sampling for batch of cells.
    
    Args:
        X (np.ndarray): Raw gene expression matrix, shape [n_cells, n_genes]
        min_total_count (int): Minimum total count threshold
        p_gamma (float): Probability of gamma=1 in Bernoulli
        alpha, beta (float): Parameters of Beta distribution
    
    Returns:
        X_downsampled (np.ndarray): Down-sampled expression matrix, shape [n_cells, n_genes]
    """
    n_cells, _ = X.shape
    X_downsampled = X.copy()
    
    # 1. Total counts per cell
    total_counts = X.sum(axis=1)
    
    # 2. Bernoulli γ for all cells
    gamma = np.random.binomial(1, p_gamma, size=n_cells)
    
    # 3. Cells eligible for down-sampling: total_count >= min_total_count and gamma=1
    eligible_mask = (total_counts >= min_total_count) & (gamma == 1)
    eligible_idx = np.where(eligible_mask)[0]

    if eligible_idx.size > 0:
        # 4. Sample b for these cells (shape [n_eligible])
        b = np.random.beta(alpha, beta, size=eligible_idx.size)
        # 5. Binomial down-sampling (broadcast b[:, None] to match [n_genes])
        # np.random.binomial supports broadcasting
        X_downsampled[eligible_idx] = np.random.binomial(X[eligible_idx], b[:, None])
    
    return X_downsampled


def bayesian_downsampling_rna(
    X, 
    min_total_count=1000, 
    p_gamma=0.5, 
    alpha=2, 
    beta=2,
    min_count_after_downsampling=200,
):
    """
    Vectorized Hierarchical Bayesian down-sampling for batch of cells.
    Cells whose total count < min_count_after_downsampling after sampling will be reverted.

    Args:
        X (np.ndarray): Raw gene expression matrix, shape [n_cells, n_genes]
        min_total_count (int): Minimum total count threshold for eligibility
        p_gamma (float): Probability of gamma=1 in Bernoulli
        alpha, beta (float): Parameters of Beta distribution
        min_count_after_downsampling (int): Threshold below which the cell is reverted

    Returns:
        X_downsampled (np.ndarray): Down-sampled expression matrix
    """
    n_cells, _ = X.shape
    X_downsampled = X.copy()
    
    # 1. Total counts per cell
    total_counts = X.sum(axis=1)
    
    # 2. Bernoulli γ for all cells
    gamma = np.random.binomial(1, p_gamma, size=n_cells)
    
    # 3. Eligible cells
    eligible_mask = (total_counts >= min_total_count) & (gamma == 1)
    eligible_idx = np.where(eligible_mask)[0]

    if eligible_idx.size > 0:
        # 4. Sample b for these cells (shape [n_eligible])
        b = np.random.beta(alpha, beta, size=eligible_idx.size)
        
        # 5. Perform downsampling
        X_tmp = np.random.binomial(X[eligible_idx], b[:, None])
        
        # 6. Compute total counts after downsampling
        total_after = X_tmp.sum(axis=1)
        
        # 7. Identify cells whose total < min_count_after_downsampling
        low_count_mask = total_after < min_count_after_downsampling
        
        # 8. Apply updates only for cells that pass threshold
        X_downsampled[eligible_idx[~low_count_mask]] = X_tmp[~low_count_mask]
        # Cells with too-low total are left as original (no update)

    return X_downsampled


def bayesian_downsampling_atac(
    X, 
    min_total_count=1000, 
    p_gamma=0.5, 
    alpha=2, 
    beta=2,
    p_nonzero_downsample=0.2
):
    """
    Bayesian down-sampling with partial non-zero selection.

    Args:
        X (np.ndarray): Raw gene expression matrix, shape [n_cells, n_genes]
        min_total_count (int): Minimum total count threshold for eligibility
        p_gamma (float): Probability of gamma=1 in Bernoulli
        alpha, beta (float): Parameters of Beta distribution
        p_nonzero_downsample (float): Fraction of non-zero entries to down-sample

    Returns:
        X_downsampled (np.ndarray): Down-sampled expression matrix
    """
    n_cells, _ = X.shape
    X_downsampled = X.copy()
    
    # 1. Total counts per cell
    total_counts = X.sum(axis=1)
    
    # 2. Bernoulli γ for all cells
    gamma = np.random.binomial(1, p_gamma, size=n_cells)
    
    # 3. Eligible cells
    eligible_mask = (total_counts >= min_total_count) & (gamma == 1)
    eligible_idx = np.where(eligible_mask)[0]

    for i in eligible_idx:
        # 4. Draw b for this cell
        b = np.random.beta(alpha, beta)
        
        # 5. Find nonzero indices
        nonzero_idx = np.where(X[i] > 0)[0]
        if nonzero_idx.size == 0:
            continue

        # 6. Randomly select subset of nonzero genes to downsample
        n_select = max(1, int(p_nonzero_downsample * len(nonzero_idx)))
        downsample_idx = np.random.choice(nonzero_idx, n_select, replace=False)
        
        # 7. Apply binomial downsampling only to selected nonzero positions
        X_downsampled[i, downsample_idx] = np.random.binomial(X[i, downsample_idx], b)
        # others remain unchanged
    
    return X_downsampled


def bayesian_downsampling_adt(
    X, 
    min_total_count=1000, 
    p_gamma=0.5, 
    alpha=2, 
    beta=2,
    min_count_after_downsampling=10,
):
    """
    Vectorized Hierarchical Bayesian down-sampling for batch of cells.
    Cells whose total count < min_count_after_downsampling after sampling will be reverted.

    Args:
        X (np.ndarray): Raw gene expression matrix, shape [n_cells, n_genes]
        min_total_count (int): Minimum total count threshold for eligibility
        p_gamma (float): Probability of gamma=1 in Bernoulli
        alpha, beta (float): Parameters of Beta distribution
        min_count_after_downsampling (int): Threshold below which the cell is reverted

    Returns:
        X_downsampled (np.ndarray): Down-sampled expression matrix
    """
    n_cells, _ = X.shape
    X_downsampled = X.copy()
    
    # 1. Total counts per cell
    total_counts = X.sum(axis=1)
    
    # 2. Bernoulli γ for all cells
    gamma = np.random.binomial(1, p_gamma, size=n_cells)
    
    # 3. Eligible cells
    eligible_mask = (total_counts >= min_total_count) & (gamma == 1)
    eligible_idx = np.where(eligible_mask)[0]

    if eligible_idx.size > 0:
        # 4. Sample b for these cells (shape [n_eligible])
        b = np.random.beta(alpha, beta, size=eligible_idx.size)
        
        # 5. Perform downsampling
        X_tmp = np.random.binomial(X[eligible_idx], b[:, None])
        
        # 6. Compute total counts after downsampling
        total_after = X_tmp.sum(axis=1)
        
        # 7. Identify cells whose total < min_count_after_downsampling
        low_count_mask = total_after < min_count_after_downsampling
        
        # 8. Apply updates only for cells that pass threshold
        X_downsampled[eligible_idx[~low_count_mask]] = X_tmp[~low_count_mask]
        # Cells with too-low total are left as original (no update)

    return X_downsampled