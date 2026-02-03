#!/usr/bin/env python3
"""Read metadata from TIFF files.

This script reads and displays metadata from TIFF files, commonly used in
microscopy. It extracts key information including image dimensions, voxel
sizes (if available), and byte order.

Usage:
    python read_metadata_tif.py <path>

    where <path> can be:
    - A single .tif/.tiff file
    - A directory containing .tif/.tiff files
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import tifffile


def get_tiff_metadata(tif: tifffile.TiffFile) -> Dict[str, Any]:
    """Extract metadata from a TiffFile object.

    Args:
        tif: An open TiffFile object.

    Returns:
        A dictionary containing image metadata.
    """
    metadata = {}

    # 1. Get shape and dtype from series (for correct 3D shape)
    if tif.series:
        series = tif.series[0]
        metadata["shape"] = series.shape
        metadata["dtype"] = str(series.dtype)
    else:
        # Fallback: count pages
        metadata["shape"] = (len(tif.pages),) + tif.pages[0].shape
        metadata["dtype"] = str(tif.pages[0].dtype)

    # 2. Get byte order
    byteorder = tif.byteorder
    if byteorder == '<':
        metadata["byteorder"] = "little-endian"
    elif byteorder == '>':
        metadata["byteorder"] = "big-endian"
    else:
        metadata["byteorder"] = "unknown"

    # 3. Try to get voxel info from ImageJ metadata
    voxel_z = None
    voxel_y = None
    voxel_x = None

    if tif.imagej_metadata:
        # Z spacing from ImageJ metadata
        spacing = tif.imagej_metadata.get('spacing')
        if spacing is not None:
            voxel_z = float(spacing)

        # XY resolution from ImageJ metadata (if available)
        resolution = tif.imagej_metadata.get('resolution')
        unit = tif.imagej_metadata.get('unit', '')

        if resolution is not None and float(resolution) > 0:
            # resolution is pixels per unit
            if unit.lower() in ('micron', 'um', 'µm', 'micrometer'):
                voxel_x = 1.0 / float(resolution)
                voxel_y = 1.0 / float(resolution)

    # 4. If XY not found in ImageJ metadata, try TIFF tags
    if voxel_x is None and tif.pages:
        page = tif.pages[0]
        tags = page.tags

        # Get resolution unit (default: no unit)
        res_unit = 1  # 1=no unit, 2=inch, 3=cm
        if 'ResolutionUnit' in tags:
            res_unit = tags['ResolutionUnit'].value

        # Get X and Y resolution
        x_res_tag = tags.get('XResolution')
        y_res_tag = tags.get('YResolution')

        if x_res_tag is not None and y_res_tag is not None:
            # Resolution is stored as (numerator, denominator) tuple
            x_res_val = x_res_tag.value
            y_res_val = y_res_tag.value

            if isinstance(x_res_val, tuple):
                x_res = x_res_val[0] / x_res_val[1] if x_res_val[1] != 0 else 0
            else:
                x_res = float(x_res_val)

            if isinstance(y_res_val, tuple):
                y_res = y_res_val[0] / y_res_val[1] if y_res_val[1] != 0 else 0
            else:
                y_res = float(y_res_val)

            # Convert to micrometers based on unit
            if res_unit == 2 and x_res > 0 and y_res > 0:
                # pixels per inch -> um per pixel
                # 1 inch = 25400 um
                voxel_x = 25400.0 / x_res
                voxel_y = 25400.0 / y_res
            elif res_unit == 3 and x_res > 0 and y_res > 0:
                # pixels per cm -> um per pixel
                # 1 cm = 10000 um
                voxel_x = 10000.0 / x_res
                voxel_y = 10000.0 / y_res
            elif res_unit == 1 and x_res > 0 and y_res > 0:
                # No unit specified - assume pixels per micron if in reasonable range
                if 0.01 < x_res < 100 and 0.01 < y_res < 100:
                    voxel_x = 1.0 / x_res
                    voxel_y = 1.0 / y_res

    # Store voxel info if available
    if voxel_z is not None:
        metadata["voxel_z"] = voxel_z
    if voxel_y is not None:
        metadata["voxel_y"] = voxel_y
    if voxel_x is not None:
        metadata["voxel_x"] = voxel_x

    return metadata


def read_tiff_metadata(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read metadata from a single TIFF file.

    Args:
        filepath: Path to the TIFF file.

    Returns:
        A dictionary containing all extracted metadata, or None if error.
    """
    try:
        with tifffile.TiffFile(filepath) as tif:
            metadata = {"filename": filepath.name}
            tiff_meta = get_tiff_metadata(tif)
            metadata.update(tiff_meta)
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
    shape = metadata.get("shape", ())
    if len(shape) >= 3:
        shape_str = f"{shape[0]} x {shape[1]} x {shape[2]}"
    elif len(shape) == 2:
        shape_str = f"1 x {shape[0]} x {shape[1]}"
    else:
        shape_str = str(shape)

    # Voxel size (ZYX order to match shape)
    voxel_z = metadata.get("voxel_z")
    voxel_y = metadata.get("voxel_y")
    voxel_x = metadata.get("voxel_x")

    if voxel_z is not None and voxel_y is not None and voxel_x is not None:
        voxel_str = f"{voxel_z:.4f} x {voxel_y:.4f} x {voxel_x:.4f}"
        # Scale: Z voxel / X voxel
        scale = voxel_z / voxel_x if voxel_x > 0 else 0
        scale_str = f"{scale:.2f}x"
    elif voxel_z is not None:
        # Only Z spacing available
        voxel_str = f"{voxel_z:.4f} x N/A x N/A"
        scale_str = "N/A"
    else:
        voxel_str = "N/A"
        scale_str = "N/A"

    # Data type and byte order
    dtype = metadata.get("dtype", "?")
    byteorder = metadata.get("byteorder", "?")

    return f"{filename}\n  Shape (ZYX): {shape_str} | Voxel (ZYX): {voxel_str} | Scale: {scale_str} | {dtype} | {byteorder}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Read metadata from TIFF files."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to TIFF file or directory containing TIFF files"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    # Collect files to process
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix not in (".tif", ".tiff"):
            print(f"Error: File is not a TIFF file: {path}")
            sys.exit(1)
        files = [path]
    else:
        files = sorted(list(path.glob("*.tif")) + list(path.glob("*.tiff")))
        if not files:
            print(f"No TIFF files found in: {path}")
            sys.exit(1)

    print(f"Found {len(files)} TIFF file(s)\n")

    # Process each file
    for filepath in files:
        metadata = read_tiff_metadata(filepath)
        if metadata:
            print(format_metadata(metadata))
            print()


if __name__ == "__main__":
    main()


# python data_tools/read_metadata_tif.py /media/aero/HDD01/CharlieChang/Data/filopodia/
# python data_tools/read_metadata_tif.py /media/aero/HDD01/CharlieChang/Data/filopodia_tif/
# python data_tools/read_metadata_tif.py /media/aero/HDD01/CharlieChang/Data/filopodia/sample.tif
# python data_tools/read_metadata_tif.py /media/aero/HDD01/CharlieChang/Data/filopodia/10X_G0431\;6xGFP_incubator_10xSp_4d_Ch-GFP_G-ch-bio_SA635_2_Stitch.tif
