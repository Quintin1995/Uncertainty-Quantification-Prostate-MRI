import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_blur_3d(volume: np.ndarray, sigma_xy: float = 1.0, sigma_z: float = 0.0) -> np.ndarray:
    """
    Applies a Gaussian blur to a 3D volume with separate control for in-plane and through-slice smoothing.

    Parameters:
        volume (np.ndarray): 3D volume with shape (slices, height, width)
        sigma_xy (float): Standard deviation for Gaussian kernel in x/y plane
        sigma_z (float): Standard deviation for z-axis (slice-wise) smoothing

    Returns:
        np.ndarray: Blurred volume
    """
    sigma = (sigma_z, sigma_xy, sigma_xy)
    return gaussian_filter(volume, sigma=sigma)

