#!/usr/bin/env python3
"""Convert IMS (Imaris) files to TIFF format.

This script converts .ims files (HDF5-based Imaris format) to TIFF files,
preserving voxel information in ImageJ-compatible metadata.

Usage:
    python ims_to_tif.py --input <path> --output <path>

    where input/output can be:
    - A single .ims/.tif file
    - A directory (for batch conversion)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
import tifffile


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
        if val.dtype.kind == "S":
            return b"".join(val).decode("utf-8")
        return str(val.item()) if val.size == 1 else str(val)
    return str(val)


def get_voxel_info(f: h5py.File) -> Dict[str, float]:
    """Extract voxel size from DataSetInfo/Image.

    Voxel size is calculated as: (ExtMax - ExtMin) / Size

    Args:
        f: Open HDF5 file handle.

    Returns:
        Dictionary with 'voxel_x', 'voxel_y', 'voxel_z' in micrometers.
    """
    voxel = {}

    if "DataSetInfo" not in f or "Image" not in f["DataSetInfo"]:
        return voxel

    image_attrs = f["DataSetInfo"]["Image"].attrs

    # Dimension mapping: 0=X, 1=Y, 2=Z
    dim_names = ["x", "y", "z"]
    size_keys = ["X", "Y", "Z"]

    for i, (dim, size_key) in enumerate(zip(dim_names, size_keys)):
        ext_min_key = f"ExtMin{i}"
        ext_max_key = f"ExtMax{i}"

        if ext_min_key in image_attrs and ext_max_key in image_attrs and size_key in image_attrs:
            try:
                ext_min = float(decode_attr(image_attrs[ext_min_key]))
                ext_max = float(decode_attr(image_attrs[ext_max_key]))
                size = int(decode_attr(image_attrs[size_key]))
                if size > 0:
                    voxel[f"voxel_{dim}"] = abs(ext_max - ext_min) / size
            except (ValueError, TypeError):
                pass

    return voxel


def get_image_size(f: h5py.File) -> Dict[str, int]:
    """Extract image size from DataSetInfo/Image metadata.

    This returns the logical image size (without padding).

    Args:
        f: Open HDF5 file handle.

    Returns:
        Dictionary with 'size_x', 'size_y', 'size_z'.
    """
    size = {}

    if "DataSetInfo" not in f or "Image" not in f["DataSetInfo"]:
        return size

    image_attrs = f["DataSetInfo"]["Image"].attrs

    for dim in ["X", "Y", "Z"]:
        if dim in image_attrs:
            try:
                size[f"size_{dim.lower()}"] = int(decode_attr(image_attrs[dim]))
            except (ValueError, TypeError):
                pass

    return size


def get_channel_count(f: h5py.File) -> int:
    """Count the number of channels in the IMS file.

    Args:
        f: Open HDF5 file handle.

    Returns:
        Number of channels.
    """
    base_path = "DataSet/ResolutionLevel 0/TimePoint 0"
    if base_path not in f:
        return 0

    count = 0
    for key in f[base_path].keys():
        if key.startswith("Channel"):
            count += 1
    return count


def read_ims_data(f: h5py.File, channel: int = 0) -> np.ndarray:
    """Read image data from the specified channel at full resolution.

    Args:
        f: Open HDF5 file handle.
        channel: Channel index (default: 0).

    Returns:
        numpy array with shape (Z, Y, X).
    """
    data_path = f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"
    return f[data_path][:]


def write_tiff(data: np.ndarray, path: Path, voxel: Dict[str, float]) -> None:
    """Write image data to TIFF with voxel information.

    Args:
        data: Image data with shape (Z, Y, X).
        path: Output file path.
        voxel: Dictionary with 'voxel_x', 'voxel_y', 'voxel_z'.
    """
    # Prepare ImageJ metadata
    # Must explicitly set all dimensions for correct interpretation
    metadata = {
        "axes": "ZYX",
        "slices": data.shape[0],  # Z slices
        "channels": 1,             # Color channels
        "frames": 1,               # Time frames
    }
    if "voxel_z" in voxel:
        metadata["spacing"] = voxel["voxel_z"]
    metadata["unit"] = "um"

    # Prepare resolution (pixels per micrometer)
    resolution = None
    if "voxel_x" in voxel and "voxel_y" in voxel:
        if voxel["voxel_x"] > 0 and voxel["voxel_y"] > 0:
            resolution = (1.0 / voxel["voxel_x"], 1.0 / voxel["voxel_y"])

    # Check if BigTIFF is needed (>4GB)
    data_size_gb = data.nbytes / (1024**3)
    use_bigtiff = data_size_gb > 4.0

    # Write TIFF
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        bigtiff=use_bigtiff,
        metadata=metadata,
        resolution=resolution,
        resolutionunit=1 if resolution else None,
    )


def convert_file(
    input_path: Path,
    output_path: Path,
    index: int = 1,
    total: int = 1
) -> bool:
    """Convert a single IMS file to TIFF.

    Args:
        input_path: Path to input .ims file.
        output_path: Path to output .tif file or directory.
        index: Current file index (for display).
        total: Total number of files (for display).

    Returns:
        True if successful, False otherwise.
    """
    # Determine output file path
    if output_path.is_dir() or not output_path.suffix:
        output_file = output_path / (input_path.stem + ".tif")
    else:
        output_file = output_path

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{index}/{total}] {input_path.name}")

    try:
        with h5py.File(input_path, "r") as f:
            # Check channel count
            n_channels = get_channel_count(f)
            if n_channels > 1:
                print(f"  Warning: {n_channels} channels found, only converting Channel 0")

            # Get voxel info and image size
            voxel = get_voxel_info(f)
            size = get_image_size(f)

            # Read image data (Channel 0 only)
            data = read_ims_data(f, channel=0)

            # Crop padding if metadata size is available
            if all(k in size for k in ["size_x", "size_y", "size_z"]):
                sz, sy, sx = size["size_z"], size["size_y"], size["size_x"]
                if data.shape != (sz, sy, sx):
                    data = data[:sz, :sy, :sx]

            # Format shape string
            shape_str = f"{data.shape[0]} x {data.shape[1]} x {data.shape[2]}"

            # Format voxel string
            if all(k in voxel for k in ["voxel_x", "voxel_y", "voxel_z"]):
                voxel_str = f"{voxel['voxel_z']:.4f} x {voxel['voxel_y']:.4f} x {voxel['voxel_x']:.4f}"
            else:
                voxel_str = "N/A"

            # Print info and write TIFF
            print(f"  Shape (ZYX): {shape_str} | Voxel (ZYX): {voxel_str} -> {output_file.name}")

            write_tiff(data, output_file, voxel)

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert IMS files to TIFF format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input .ims file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output .tif file or directory"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input path
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Collect files to process
    if input_path.is_file():
        if input_path.suffix.lower() != ".ims":
            print(f"Error: Input file is not an .ims file: {input_path}")
            sys.exit(1)
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.ims"))
        if not files:
            print(f"Error: No .ims files found in: {input_path}")
            sys.exit(1)

    # Ensure output directory exists (if output is a directory)
    if len(files) > 1 or (not output_path.suffix and not output_path.exists()):
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} .ims file(s)\n")

    # Process each file
    success_count = 0
    for i, filepath in enumerate(files, start=1):
        if convert_file(filepath, output_path, index=i, total=len(files)):
            success_count += 1
        print()

    print(f"Done. Converted {success_count} file(s).")


if __name__ == "__main__":
    main()

# python script/ims_to_tif.py --input /media/aero/HDD01/CharlieChang/Data/filopodia_ims/10X_G0431\;6xGFP_incubator_10xSp_4d_Ch-GFP_G-ch-bio_SA635_2_Stitch.ims --output /media/aero/HDD01/CharlieChang/Data/10X_G0431\;6xGFP_incubator_10xSp_4d_Ch-GFP_G-ch-bio_SA635_2_Stitch.tif
# python script/ims_to_tif.py --input /media/aero/HDD01/CharlieChang/Data/filopodia_ims/ --output /media/aero/HDD01/CharlieChang/Data/filopodia_tif/