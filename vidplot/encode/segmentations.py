"""
Segmentation mask encoding and decoding utilities.

This module provides functions to encode and decode binary segmentation masks using
Run-Length Encoding (RLE) for efficient storage and transmission.
"""

import numpy as np
from typing import Dict, Any, Tuple, Union, List


def encode_segmentation_mask(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a single binary segmentation mask using Run-Length Encoding (RLE).

    This function encodes all necessary information to rebuild the binary segmentation mask,
    including the original shape, data type, and the RLE-encoded data.

    For binary masks (0s and 1s), only run lengths are stored,
    with values inferred from the alternation pattern starting with the first value.

    Args:
        mask: A 2D numpy array representing the binary segmentation mask.
              Values should be 0s and 1s only.

    Returns:
        A dictionary containing:
        - 'shape': Tuple of (height, width)
        - 'dtype': The original data type as a string
        - 'rle': List of run lengths (values inferred from alternation)
        - 'total_pixels': Total number of pixels in the mask
        - 'first_value': The first value in the mask (0 or 1)

    Raises:
        ValueError: If mask is not 2D, not binary, or not an integer/boolean array
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    # Convert to binary (0s and 1s) if needed
    binary_mask = mask.astype(bool).astype(np.uint8)

    # Verify it's truly binary
    unique_values = np.unique(binary_mask)
    if not (len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values)):
        raise ValueError("Mask must contain only 0s and 1s")

    # Flatten the mask for RLE encoding
    flat_mask = binary_mask.flatten()

    # Initialize RLE encoding
    rle = []
    current_value = int(flat_mask[0])
    current_count = 1

    # Encode runs
    for pixel in flat_mask[1:]:
        pixel_value = int(pixel)
        if pixel_value == current_value:
            current_count += 1
        else:
            rle.append(current_count)
            current_value = pixel_value
            current_count = 1

    # Add the last run
    rle.append(current_count)

    return {
        "shape": mask.shape,
        "dtype": str(mask.dtype),
        "rle": rle,
        "total_pixels": mask.size,
        "first_value": int(flat_mask[0]),
    }


def encode_segmentation_masks(
    seg_ids: List[int],
    seg_masks: List[np.ndarray],
    save_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Encode multiple binary segmentation masks using Run-Length Encoding (RLE).

    This function encodes a list of segmentation masks with their corresponding IDs.
    For video sequences, use save_metadata=False to exclude metadata for subsequent frames.

    Args:
        seg_ids: List of integer IDs identifying each segmentation mask
        seg_masks: List of 2D numpy arrays representing binary segmentation masks
        save_metadata: If True, includes shape, dtype, total_pixels, first_values in output

    Returns:
        A dictionary containing:
        - 'seg_ids': List of segmentation IDs
        - 'rles': List of RLE data for each mask
        - 'shape': Tuple of (height, width) (only if save_metadata=True)
        - 'dtype': The original data type as a string (only if save_metadata=True)
        - 'total_pixels': Total number of pixels in the mask (only if save_metadata=True)
        - 'first_values': List of first values for each mask (only if save_metadata=True)

    Raises:
        ValueError: If inputs are invalid or inconsistent
    """
    if len(seg_ids) != len(seg_masks):
        raise ValueError("seg_ids and seg_masks must have the same length")

    # Validate all masks
    for i, mask in enumerate(seg_masks):
        if mask.ndim != 2:
            raise ValueError(f"Mask {i} must be a 2D array")

        # Convert to binary (0s and 1s) if needed
        binary_mask = mask.astype(bool).astype(np.uint8)

        # Verify it's truly binary
        unique_values = np.unique(binary_mask)
        if not (len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values)):
            raise ValueError(f"Mask {i} must contain only 0s and 1s")

    # Encode each mask
    rles = []
    first_values = []

    for mask in seg_masks:
        binary_mask = mask.astype(bool).astype(np.uint8)
        flat_mask = binary_mask.flatten()

        # Initialize RLE encoding
        rle = []
        current_value = int(flat_mask[0])
        current_count = 1

        # Encode runs
        for pixel in flat_mask[1:]:
            pixel_value = int(pixel)
            if pixel_value == current_value:
                current_count += 1
            else:
                rle.append(current_count)
                current_value = pixel_value
                current_count = 1

        # Add the last run
        rle.append(current_count)

        rles.append(rle)
        first_values.append(int(flat_mask[0]))

    # Build result dictionary
    result = {"seg_ids": seg_ids, "rles": rles}

    if save_metadata:
        # Include full metadata
        result.update(
            {
                "shape": seg_masks[0].shape,
                "dtype": str(seg_masks[0].dtype),
                "total_pixels": seg_masks[0].size,
                "first_values": first_values,
            }
        )

    return result


def decode_segmentation_masks(
    encoded_data: Dict[str, Any], reference_data: Dict[str, Any] = None
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Decode multiple binary segmentation masks from RLE-encoded data.

    This function reconstructs the original binary segmentation masks from the encoded data
    produced by encode_segmentation_masks().

    Args:
        encoded_data: Dictionary containing the encoded mask data
        reference_data: Reference data from first frame (if encoded without metadata)

    Returns:
        Tuple of (seg_ids, seg_masks) where seg_ids is a list of integers and
        seg_masks is a list of 2D numpy arrays

    Raises:
        ValueError: If encoded_data is invalid
        KeyError: If required keys are missing
    """
    required_keys = ["seg_ids", "rles"]
    for key in required_keys:
        if key not in encoded_data:
            raise KeyError(f"Missing required key '{key}' in encoded_data")

    seg_ids = encoded_data["seg_ids"]
    rles = encoded_data["rles"]

    if len(seg_ids) != len(rles):
        raise ValueError("seg_ids and rles must have the same length")

    # Determine if this has metadata or needs reference data
    has_metadata = "shape" in encoded_data

    if has_metadata:
        # Use metadata from encoded_data
        shape = encoded_data["shape"]
        dtype_str = encoded_data["dtype"]
        total_pixels = encoded_data["total_pixels"]
        first_values = encoded_data["first_values"]
    else:
        # Need reference data for metadata
        if reference_data is None:
            raise ValueError("reference_data is required when encoded_data has no metadata")

        # Use reference data for metadata
        shape = reference_data["shape"]
        dtype_str = reference_data["dtype"]
        total_pixels = reference_data["total_pixels"]
        first_values = encoded_data.get("first_values", [0] * len(rles))

    # Validate the data
    if len(shape) != 2:
        raise ValueError("Shape must be 2D")

    if total_pixels != shape[0] * shape[1]:
        raise ValueError("Total pixels doesn't match shape")

    # Decode each mask
    seg_masks = []

    for i, (rle, first_value) in enumerate(zip(rles, first_values)):
        if first_value not in [0, 1]:
            raise ValueError(f"first_value {i} must be 0 or 1")

        # Reconstruct the flattened mask using alternation pattern
        flat_mask = []
        current_value = first_value

        for count in rle:
            flat_mask.extend([current_value] * count)
            # Alternate between 0 and 1
            current_value = 1 - current_value  # 0 -> 1, 1 -> 0

        # Validate that we have the right number of pixels
        if len(flat_mask) != total_pixels:
            raise ValueError(f"RLE data {i} doesn't match total_pixels")

        # Reshape and convert to the original dtype
        mask = np.array(flat_mask, dtype=np.dtype(dtype_str))
        mask = mask.reshape(shape)

        seg_masks.append(mask)

    return seg_ids, seg_masks


def decode_segmentation_mask(encoded_data: Dict[str, Any]) -> np.ndarray:
    """
    Decode a single binary segmentation mask from RLE-encoded data.

    This function reconstructs the original binary segmentation mask from the encoded data
    produced by encode_segmentation_mask().

    Args:
        encoded_data: Dictionary containing the encoded mask data with keys:
                     'shape', 'dtype', 'rle', 'total_pixels', 'first_value'

    Returns:
        A 2D numpy array representing the reconstructed binary segmentation mask.

    Raises:
        ValueError: If encoded_data is missing required keys or is invalid
        KeyError: If required keys are missing from encoded_data
    """
    required_keys = ["shape", "dtype", "rle", "total_pixels", "first_value"]
    for key in required_keys:
        if key not in encoded_data:
            raise KeyError(f"Missing required key '{key}' in encoded_data")

    shape = encoded_data["shape"]
    dtype_str = encoded_data["dtype"]
    rle = encoded_data["rle"]
    total_pixels = encoded_data["total_pixels"]
    first_value = encoded_data["first_value"]

    # Validate the data
    if len(shape) != 2:
        raise ValueError("Shape must be 2D")

    if total_pixels != shape[0] * shape[1]:
        raise ValueError("Total pixels doesn't match shape")

    if first_value not in [0, 1]:
        raise ValueError("first_value must be 0 or 1")

    # Reconstruct the flattened mask using alternation pattern
    flat_mask = []
    current_value = first_value

    for count in rle:
        flat_mask.extend([current_value] * count)
        # Alternate between 0 and 1
        current_value = 1 - current_value  # 0 -> 1, 1 -> 0

    # Validate that we have the right number of pixels
    if len(flat_mask) != total_pixels:
        raise ValueError("RLE data doesn't match total_pixels")

    # Reshape and convert to the original dtype
    mask = np.array(flat_mask, dtype=np.dtype(dtype_str))
    mask = mask.reshape(shape)

    return mask


def rle_to_binary_mask(rle: list, shape: Tuple[int, int], first_value: int = 0) -> np.ndarray:
    """
    Convert RLE data to a binary mask.

    This is a utility function that can be used to create binary masks
    from RLE data.

    Args:
        rle: List of run lengths from RLE encoding
        shape: Tuple of (height, width) for the output mask
        first_value: The first value in the alternation pattern (0 or 1)

    Returns:
        A 2D binary numpy array where True indicates the presence of the target class
    """
    if first_value not in [0, 1]:
        raise ValueError("first_value must be 0 or 1")

    flat_mask = []
    current_value = first_value

    for count in rle:
        flat_mask.extend([current_value] * count)
        # Alternate between 0 and 1
        current_value = 1 - current_value  # 0 -> 1, 1 -> 0

    # Convert to boolean array and reshape
    mask = np.array(flat_mask, dtype=bool).reshape(shape)

    return mask


def calculate_compression_ratio(
    original_masks: Union[np.ndarray, List[np.ndarray]], encoded_data: Dict[str, Any]
) -> float:
    """
    Calculate the compression ratio achieved by RLE encoding.

    Can handle both single mask (from encode_segmentation_mask) and multiple
    masks (from encode_segmentation_masks).

    Args:
        original_masks: A single mask or list of original binary segmentation masks.
        encoded_data: The encoded data from encode_segmentation_mask() or
                      encode_segmentation_masks().

    Returns:
        Compression ratio (original_size / encoded_size)
    """
    if isinstance(original_masks, list):
        original_size = sum(mask.nbytes for mask in original_masks)
    else:
        # It's a single ndarray
        original_size = original_masks.nbytes

    encoded_size = 0

    # Case for multiple masks from encode_segmentation_masks
    if "rles" in encoded_data:
        total_rle_size = sum(len(rle) * 4 for rle in encoded_data["rles"])
        seg_ids_size = len(encoded_data["seg_ids"]) * 4

        metadata_size = 0
        if "shape" in encoded_data:
            # Full encoding with metadata
            shape_str = str(encoded_data["shape"])
            dtype_str = encoded_data["dtype"]
            first_values_size = len(encoded_data["first_values"]) * 4

            metadata_size = len(shape_str) + len(dtype_str) + 8 + first_values_size + 30  # overhead
        else:
            # Subsequent frame, no metadata
            metadata_size = 20  # overhead

        encoded_size = total_rle_size + seg_ids_size + metadata_size

    # Case for single mask from encode_segmentation_mask
    elif "rle" in encoded_data:
        # Single mask encoding always includes metadata
        total_rle_size = len(encoded_data["rle"]) * 4

        shape_str = str(encoded_data["shape"])
        dtype_str = str(encoded_data["dtype"])
        first_value_size = 4  # one int for 'first_value'

        metadata_size = (
            len(shape_str) + len(dtype_str) + 8 + first_value_size + 30  # total_pixels  # overhead
        )
        encoded_size = total_rle_size + metadata_size
    else:
        raise KeyError("Encoded data must contain 'rle' or 'rles' key for compression calculation.")

    if encoded_size == 0:
        return float("inf") if original_size > 0 else 1.0

    return original_size / encoded_size
