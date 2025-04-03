import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk


def calculate_uncertainty_map(reconstructions: np.ndarray, debug=False) -> np.ndarray:
    """
    Calculate the uncertainty map by computing the standard deviation across reconstructions.
    
    Args:
        reconstructions (np.ndarray): 4D NumPy array with shape (num_reconstructions, slices, rows, cols).
    
    Returns:
        np.ndarray: 3D NumPy array representing the uncertainty map with shape (slices, rows, cols).
    """
    assert isinstance(reconstructions, np.ndarray), "reconstructions must be a NumPy array."
    assert reconstructions.ndim == 4, f"Expected 4D array, got {reconstructions.ndim}D array."
    
    uq_map = reconstructions.std(axis=0)
    if debug: 
        print(f"\tShape of uncertainty map: {uq_map.shape}")
        print(f"\tMin max of the uncertainty map (rounded): {np.round(uq_map.min(), 4)} - {np.round(uq_map.max(), 4)}")
    return uq_map


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
