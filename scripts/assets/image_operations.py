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


def adaptive_clip(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    do_normalize: bool = False
) -> np.ndarray:
    """
    Clips the image intensities based on percentile thresholds to improve contrast.
    
    Args:
        image (np.ndarray): The input image (2D or 3D). 
                            If 3D, shape could be (slices, rows, columns).
        low_percentile (float): Lower percentile to clip at. Default = 1.0 (i.e., 1st percentile).
        high_percentile (float): Upper percentile to clip at. Default = 99.0 (i.e., 99th percentile).
        do_normalize (bool): If True, normalize the clipped image to [0,1].
    
    Returns:
        np.ndarray: The clipped (and optionally normalized) image, same shape as input.
    """
    assert isinstance(image, np.ndarray), "image must be a numpy array"
    assert 0.0 <= low_percentile < 100.0, "low_percentile must be between 0 and 100"
    assert 0.0 < high_percentile <= 100.0, "high_percentile must be between 0 and 100"
    assert low_percentile < high_percentile, "low_percentile must be less than high_percentile"

    # Flatten the data for percentile calculation
    flat_data = image.flatten()
    p_low = np.percentile(flat_data, low_percentile)
    p_high = np.percentile(flat_data, high_percentile)

    clipped_image = np.clip(image, p_low, p_high)

    if do_normalize:
        # Avoid division by zero
        denominator = (p_high - p_low) if (p_high - p_low) != 0 else 1e-6
        clipped_image = (clipped_image - p_low) / denominator
        # Now in [0, 1], but still same shape

    return clipped_image