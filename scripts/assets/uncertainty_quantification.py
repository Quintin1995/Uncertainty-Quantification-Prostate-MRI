import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk


def calculate_uncertainty_map(
    reconstructions: np.ndarray,
    method: str = 'std',
    percentile: tuple[float, float] = (2.5, 97.5),
    debug: bool = False
) -> np.ndarray:
    """
    Calculate the uncertainty map from a stack of reconstructions.

    Args:
        reconstructions (np.ndarray):
            4D array of shape (num_recon, slices, rows, cols).
        method (str):
            One of {'std', 'cv', 'mad', 'iqr', 'pi'}:
              - 'std':      pixel-wise standard deviation.
              - 'cv':       coefficient of variation (std / mean).
              - 'mad':      median absolute deviation.
              - 'iqr':      interquartile range (75th–25th percentile).
              - 'pi':       percentile interval width (e.g., 95% PI).
        percentile (tuple):
            Lower and upper percentiles for 'pi' (default 2.5, 97.5).
        debug (bool):
            Print shape and min/max of result.

    Returns:
        np.ndarray:
            3D uncertainty map (slices, rows, cols).
    """
    assert isinstance(reconstructions, np.ndarray), "need NumPy array"
    assert reconstructions.ndim == 4, f"Expected 4D, got {reconstructions.ndim}D"
    method = method.lower()
    
    if method == 'std':
        uq = reconstructions.std(axis=0)
    
    elif method == 'cv':
        mean = reconstructions.mean(axis=0)
        std  = reconstructions.std(axis=0)
        # avoid divide-by-zero; set CV to nan where mean≈0
        uq = np.where(np.abs(mean) > 1e-8, std / np.abs(mean), np.nan)
    
    elif method == 'mad':
        med = np.median(reconstructions, axis=0)
        uq = np.median(np.abs(reconstructions - med), axis=0)
    
    elif method == 'iqr':
        p75 = np.percentile(reconstructions, 75, axis=0)
        p25 = np.percentile(reconstructions, 25, axis=0)
        uq = p75 - p25
    
    elif method == 'pi':
        low, high = percentile
        p_lo = np.percentile(reconstructions, low, axis=0)
        p_hi = np.percentile(reconstructions, high, axis=0)
        uq = p_hi - p_lo
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'std','cv','mad','iqr','pi'.")
    
    if debug:
        print(f"\t[UQ:{method}] shape {uq.shape}; min–max ≈ {uq.min():.4f}–{uq.max():.4f}")
    return uq


def apply_percentile_threshold(uq_map: np.ndarray, percentile: float = 95.0, debug=False) -> np.ndarray:
    """
    Apply a percentile threshold to the uncertainty map, keeping only the top percentile values.
    
    Args:
        uq_map (np.ndarray): 3D NumPy array representing the uncertainty map with shape (slices, rows, cols).
        percentile (float): Percentile threshold to apply. Default is 95.0.
    
    Returns:
        np.ndarray: Thresholded uncertainty map with the same shape as input.
    """
    assert isinstance(uq_map, np.ndarray), "uq_map must be a NumPy array."
    assert uq_map.ndim == 3, f"Expected 3D array, got {uq_map.ndim}D array."
    
    threshold_value = np.percentile(uq_map, percentile)
    if debug:
        print(f"Applying percentile threshold: {percentile}%")
        print(f"Threshold value: {threshold_value}")
    
    thresholded_uq_map = np.where(uq_map >= threshold_value, uq_map, 0)
    return thresholded_uq_map


def visualize_uncertainty_map(uq_map: np.ndarray, slice_idx: int = 0, save_path: Path = None, do_round = False, decimals = 3):
    """
    Visualize the uncertainty map and optionally save it as a NIfTI file.
    
    Args:
        uq_map (np.ndarray): 3D NumPy array representing the uncertainty map with shape (slices, rows, cols).
        slice_idx (int): Index of the slice to visualize.
        save_path (Path): Path to save the uncertainty map as a NIfTI file. If None, the map is not saved.
    """
    assert isinstance(uq_map, np.ndarray), "uq_map must be a NumPy array."
    assert uq_map.ndim == 3, f"Expected 3D array, got {uq_map.ndim}D array."
    assert 0 <= slice_idx < uq_map.shape[0], f"slice_idx out of range (0 to {uq_map.shape[0]-1})"
    
    print(f"This function should just be in a jupyter notebook instead of a script")

    # if the data is nicely between 0 and 1, we can use the 'gray' colormap and we can round 
    if do_round:
        print(f'Rounding uncertainty map to {decimals} decimals')
        uq_map = np.round(uq_map, decimals)

    # Plotting
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(uq_map[slice_idx], cmap='gray')
    plt.colorbar()
    plt.title(f'Uncertainty Map - Slice {slice_idx}')
    plt.show()
    
    # Save as NIfTI file if save_path is provided
    if save_path:
        uq_map_sitk = sitk.GetImageFromArray(uq_map)
        sitk.WriteImage(uq_map_sitk, str(save_path))
        print(f'Saved uncertainty map as {save_path}')
