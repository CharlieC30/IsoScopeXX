#!/usr/bin/env python3
"""Read metadata from Imaris (.ims) files.

This script reads and displays metadata from .ims files, which are HDF5-based
image files commonly used in microscopy. It extracts key information including
image dimensions and voxel sizes.

Usage:
    python read_metadata_ims.py <path>

    where <path> can be:
    - A single .ims file
    - A directory containing .ims files
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import h5py
import numpy as np


def decode_attr(val: Any) -> str:
    """Decode HDF5 attribute value to string.

    Args:
        val: Attribute value (bytes, ndarray of bytes, or other).

    Returns:
        Decoded string value.
    """
    if isinstance(val, bytes):
        return val.decode("utf-8")
    elif isinstance(val, np.ndarray):
        # Array of bytes like [b'1', b'0', b'2', b'4'] -> "1024"
        if val.dtype.kind == "S":
            return b"".join(val).decode("utf-8")
        return str(val.item()) if val.size == 1 else str(val)
    return str(val)


def get_dtype_from_dataset(f: h5py.File) -> Optional[str]:
    """Get data type from the dataset.

    Args:
        f: An open HDF5 file handle.

    Returns:
        Data type string or None if not found.
    """
    try:
        data = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
        return str(data.dtype)
    except KeyError:
        return None


def get_image_metadata(f: h5py.File) -> Dict[str, Any]:
    """Extract image metadata from DataSetInfo.

    Args:
        f: An open HDF5 file handle.

    Returns:
        A dictionary containing image metadata.
    """
    metadata = {}

    if "DataSetInfo" not in f:
        return metadata

    info = f["DataSetInfo"]

    # Image dimensions and voxel sizes
    if "Image" in info:
        image_attrs = info["Image"].attrs

        # Size (pixels) - stored as "X", "Y", "Z" in IMS files
        for dim in ["X", "Y", "Z"]:
            if dim in image_attrs:
                val = decode_attr(image_attrs[dim])
                try:
                    metadata[f"Size{dim}"] = int(val) if val else None
                except ValueError:
                    pass

        # Resolution (physical size in micrometers) from ExtMin/ExtMax
        for i, dim in enumerate(["X", "Y", "Z"]):
            ext_min_key = f"ExtMin{i}"
            ext_max_key = f"ExtMax{i}"
            size_key = f"Size{dim}"

            if ext_min_key in image_attrs and ext_max_key in image_attrs:
                try:
                    ext_min = float(decode_attr(image_attrs[ext_min_key]))
                    ext_max = float(decode_attr(image_attrs[ext_max_key]))
                    size = metadata.get(size_key)
                    if size and size > 0:
                        resolution = abs(ext_max - ext_min) / size
                        metadata[f"Resolution{dim}"] = resolution
                except (ValueError, TypeError):
                    pass

    return metadata


def read_ims_metadata(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read metadata from a single .ims file.

    Args:
        filepath: Path to the .ims file.

    Returns:
        A dictionary containing all extracted metadata, or None if error.
    """
    try:
        with h5py.File(filepath, "r") as f:
            metadata = {"filename": filepath.name}

            # Get data type
            dtype = get_dtype_from_dataset(f)
            if dtype:
                metadata["Dtype"] = dtype

            # Get image metadata
            image_meta = get_image_metadata(f)
            metadata.update(image_meta)

            return metadata

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata for compact display.

    Args:
        metadata: Dictionary containing metadata.

    Returns:
        Formatted string for printing.
    """
    filename = metadata.get("filename", "Unknown")

    # Shape (ZYX order)
    size_x = metadata.get("SizeX", "?")
    size_y = metadata.get("SizeY", "?")
    size_z = metadata.get("SizeZ", "?")
    size_str = f"{size_z} x {size_y} x {size_x}"

    # Voxel size (ZYX order)
    res_x = metadata.get("ResolutionX")
    res_y = metadata.get("ResolutionY")
    res_z = metadata.get("ResolutionZ")
    if res_x is not None and res_y is not None and res_z is not None:
        voxel_str = f"{res_z:.4f} x {res_y:.4f} x {res_x:.4f}"
        # Scale: Z voxel / XY voxel (how much Z needs to be upscaled)
        scale = res_z / res_x
        scale_str = f"{scale:.2f}x"
    else:
        voxel_str = "N/A"
        scale_str = "N/A"

    # Data type
    dtype = metadata.get("Dtype", "?")

    return f"{filename}\n  Shape (ZYX): {size_str} | Voxel (ZYX): {voxel_str} | Scale: {scale_str} | {dtype}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Read metadata from Imaris (.ims) files."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to .ims file or directory containing .ims files"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    # Collect files to process
    if path.is_file():
        if path.suffix.lower() != ".ims":
            print(f"Error: File is not an .ims file: {path}")
            sys.exit(1)
        files = [path]
    else:
        files = sorted(path.glob("*.ims"))
        if not files:
            print(f"No .ims files found in: {path}")
            sys.exit(1)

    print(f"Found {len(files)} .ims file(s)\n")

    # Process each file
    for filepath in files:
        metadata = read_ims_metadata(filepath)
        if metadata:
            print(format_metadata(metadata))
            print()


if __name__ == "__main__":
    main()


# python data_tools/read_metadata_ims.py /media/aero/HDD01/CharlieChang/Data/filopodia_ims/
# python data_tools/read_metadata_ims.py /media/aero/HDD01/CharlieChang/Data/filopodia_ims/10X_G0431\;6xGFP_incubator_10xSp_4d_Ch-GFP_G-ch-bio_SA635_2_Stitch.ims
