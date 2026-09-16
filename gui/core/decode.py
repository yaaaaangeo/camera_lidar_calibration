"""Turning bag messages into numpy arrays.

Keeps the rest of the code free of ROS message details: everything downstream
works with plain arrays.
"""

from __future__ import annotations

import numpy as np

# sensor_msgs/PointField datatype enum -> numpy dtype
_PF_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def _raw_bytes(data) -> bytes:
    return data.tobytes() if isinstance(data, np.ndarray) else bytes(data)


def cloud_fields(msg) -> list[tuple[str, str, int]]:
    """(name, dtype name, byte offset) for each field, in offset order."""
    out = []
    for f in msg.fields:
        dt = _PF_DTYPE.get(f.datatype)
        out.append((f.name, np.dtype(dt).name if dt else f"?{f.datatype}", f.offset))
    return sorted(out, key=lambda t: t[2])


def cloud_to_struct(msg) -> np.ndarray:
    """Decode a PointCloud2 into a structured array, one row per point.

    Padding between fields is preserved as unused bytes so point_step is
    honoured exactly -- some drivers leave gaps.
    """
    names, formats, offsets = [], [], []
    for f in msg.fields:
        dt = _PF_DTYPE.get(f.datatype)
        if dt is None:
            continue
        names.append(f.name)
        formats.append(dt)
        offsets.append(f.offset)

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step}
    )
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")

    n = msg.width * msg.height
    return np.frombuffer(_raw_bytes(msg.data), dtype=dtype, count=n)


def cloud_xyz(msg, drop_nonfinite: bool = True) -> np.ndarray:
    """Nx3 float32 of point positions."""
    rec = cloud_to_struct(msg)
    xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
    if drop_nonfinite:
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz


def cloud_extra(msg, field: str) -> np.ndarray | None:
    """One non-positional field (`ring`, `intensity`, ...) if the cloud has it."""
    rec = cloud_to_struct(msg)
    return np.asarray(rec[field]) if field in (rec.dtype.names or ()) else None


def image_to_bgr(msg) -> np.ndarray:
    """Decode Image or CompressedImage into an HxWx3 BGR array."""
    import cv2

    if hasattr(msg, "format"):  # CompressedImage
        buf = np.frombuffer(_raw_bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode compressed image (format={msg.format})")
        return img

    enc = msg.encoding.lower()
    buf = np.frombuffer(_raw_bytes(msg.data), dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    if enc.startswith("bayer"):
        code = {
            "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
            "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
            "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
            "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
        }.get(enc)
        if code is not None:
            return cv2.cvtColor(buf.reshape(msg.height, msg.width), code)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")
