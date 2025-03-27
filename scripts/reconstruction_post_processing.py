
import numpy as np
from typing import Tuple



####### Functions for Post-processing the reconstructions ########################
def center_crop_3d(image_3d: np.ndarray, target_shape: Tuple[int, int, int], debug: bool = False) -> np.ndarray:
    """
    Center-crop a 3D volume back to a specified shape.

    Args:
        image_3d (np.ndarray): 3D array (slices, rows, columns).
        target_shape (Tuple[int, int, int]): Desired (slices, rows, columns).

    Returns:
        np.ndarray: Cropped 3D array of shape target_shape.
    """
    assert isinstance(image_3d, np.ndarray), "image_3d must be a numpy array"
    assert len(target_shape) == 3, "target_shape must be a 3-element tuple"
    slices, rows, cols = image_3d.shape
    t_slices, t_rows, t_cols = target_shape

    # For simplicity, we only crop rows/cols and assume the slice dimension is the same
    assert slices == t_slices, "We only center-crop rows & columns here; slice dim must match."
    row_start = (rows - t_rows) // 2
    col_start = (cols - t_cols) // 2

    print(f"Center-cropping image from {image_3d.shape} to {target_shape}") if debug else None

    return image_3d[:, row_start:row_start+t_rows, col_start:col_start+t_cols]


def zero_pad_in_sim_kspace(
    image_3d: np.ndarray, 
    desired_shape: Tuple[int, int] = (1280, 1280),
) -> np.ndarray:
    """
    Zero pad a 3D image array in the simulated k-space.
    
    Parameters:
    - input_image: 3D ndarray, the image to be zero-padded
    - desired_shape: tuple, the desired shape after zero-padding
    - verbose: boolean, whether to print additional information
    
    Returns:
    - 3D ndarray, the zero-padded image
    """
    assert image_3d.dtype == np.float32, "Expecting the input image to be of type float32."
    assert image_3d.shape[0] < image_3d.shape[1] and image_3d.shape[0] < image_3d.shape[2], "Expecting the first dim to be the slice dim."
    assert len(image_3d.shape) == 3, "Expecting 3D array."
    print(f"\tShape of the input image before zero-padding: {image_3d.shape}, with dtype {image_3d.dtype}")

    n_slices, width, height = image_3d.shape
    padded_image = np.zeros((n_slices, *desired_shape), dtype=image_3d.dtype)

    pad_width = desired_shape[0] - width
    pad_height = desired_shape[1] - height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    for i in range(n_slices):
        img_slice            = image_3d[i, ...]
        kspace_slice         = np.fft.fft2(img_slice)
        kspace_slice_shift1  = np.fft.fftshift(kspace_slice)
        kspace_slice_padded  = np.pad(kspace_slice_shift1, ((pad_left, pad_right), (pad_top, pad_bottom)), mode='constant', constant_values=0)
        kspace_slice_shift2  = np.fft.ifftshift(kspace_slice_padded)
        img_slice_padded     = np.fft.ifft2(kspace_slice_shift2)
        padded_image[i, ...] = np.abs(img_slice_padded)

        # print(f"\t\tSlice {i+1}/{n_slices} done.")
        # print(f"\t\t\tShape of the kspace_slice: {kspace_slice.shape}, with dtype {kspace_slice.dtype}")
        # print(f"\t\t\tShape of the kspace_slice_padded: {kspace_slice_padded.shape}, with dtype {kspace_slice_padded.dtype}")
        # print(f"\t\t\tShape of the img_slice_padded: {img_slice_padded.shape}, with dtype {img_slice_padded.dtype}")
        # print(f"\t\t\tShape of the padded_image: {padded_image.shape}, with dtype {padded_image.dtype}")

    return padded_image


def center_crop(
    image: np.ndarray,
    crop_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Perform a center crop on a 2D or 3D numpy array.
    
    Args:
    - arr (np.ndarray): The array to be cropped. Could be either 2D or 3D.
    - crop_shape (Tuple[int, int]): Desired output shape (height, width).
    
    Returns:
    - np.ndarray: Center cropped array.
    """
    assert len(image.shape) in [2, 3], f"Invalid number of dimensions. Expected 2D or 3D array, got shape {image.shape}."
    assert len(crop_shape) == 2, f"Invalid crop_shape dimension. Expected a tuple of length 2, got {crop_shape}."

    # Original shape and crop shape 
    original_shape = image.shape[-2:]
    crop_height, crop_width = crop_shape
    
    # Padding dimensions 
    pad_height = original_shape[0] - crop_height
    pad_width = original_shape[1] - crop_width
    
    if pad_height < 0 or pad_width < 0:
        raise ValueError(f"Crop shape {crop_shape} larger than the original shape {original_shape}.")
        
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Perform the crop
    if len(image.shape) == 3:
        cropped_arr = image[:, pad_top:-pad_bottom, pad_left:-pad_right]
    else:
        cropped_arr = image[pad_top:-pad_bottom, pad_left:-pad_right]
    print(f"\tCenter crop: Original shape: {original_shape}, crop shape: {crop_shape}, pad: {pad_top, pad_bottom, pad_left, pad_right}, cropped shape: {cropped_arr.shape}")
        
    return cropped_arr


def norm_rescale01(recon: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Normalize and rescale the reconstructions to [0, 1].
    
    Args:
        recon (np.ndarray): 3D NumPy array with shape (slices, rows, cols).
    
    Returns:
        np.ndarray: Normalized and rescaled 3D NumPy array.
    """
    assert isinstance(recon, np.ndarray), "recon must be a NumPy array."

    if debug:
        print(f"\tApplying normalization rescale01 to reconstructions with shape: {recon.shape}")

    # Add a small epsilon to avoid division by zero.
    return (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)


def post_process_3d_image(
    image_3d: np.ndarray,
    zero_pad_shape: Tuple,
    image_space_crop: Tuple
) -> np.ndarray:
    """
    Apply the image processing steps to make the input image look like a dicom image.
    Args:
        image_3d (np.ndarray): The input 3D reconstruction.
        zero_pad_shape (Tuple): The desired shape after zero-padding.
        image_space_crop (Tuple): The desired shape after cropping.
    Returns:
        np.ndarray: The post-processed (dicom-like) reconstruction.
    """

    # Step 1: Flip both the width and height of the image, the RIM output relative to the dicom is flipped
    post_image_3d = np.flip(image_3d, axis=(0,1,2))
    print(f"\tFlipped width and height with axis=(0,1,2)")

    # Step 2: Zero-pad simulated kspace to the desired shape and get the image back from the padded kspace
    post_image_3d = zero_pad_in_sim_kspace(image_3d=post_image_3d, desired_shape=zero_pad_shape)

    # Step 3: Take a center crop of the image space
    post_image_3d = center_crop(post_image_3d, crop_shape=image_space_crop)

    return post_image_3d


