import numpy as np


def pack_uint10(codes: np.ndarray, filename):

    codes = np.asarray(codes, dtype=np.uint16)

    shape = np.array(codes.shape, dtype=np.uint32)

    x = codes.ravel()

    if np.any(x > 1023):
        raise ValueError("Codes must be <=1023")

    # pad to multiple of 4
    pad = (-len(x)) % 4

    if pad:
        x = np.pad(x, (0, pad))

    x = x.reshape(-1, 4)

    packed = np.empty((len(x), 5), dtype=np.uint8)

    packed[:, 0] = x[:, 0] & 0xFF

    packed[:, 1] = (
        ((x[:, 0] >> 8) & 0x03)
        | ((x[:, 1] & 0x3F) << 2)
    )

    packed[:, 2] = (
        ((x[:, 1] >> 6) & 0x0F)
        | ((x[:, 2] & 0x0F) << 4)
    )

    packed[:, 3] = (
        ((x[:, 2] >> 4) & 0x3F)
        | ((x[:, 3] & 0x03) << 6)
    )

    packed[:, 4] = (x[:, 3] >> 2)

    with open(filename, "wb") as f:

        np.array([codes.ndim], np.uint8).tofile(f)

        shape.tofile(f)

        np.array([pad], np.uint8).tofile(f)

        packed.tofile(f)


def unpack_uint10(filename):

    with open(filename, "rb") as f:

        ndim = np.fromfile(f, np.uint8, 1)[0]

        shape = tuple(np.fromfile(f, np.uint32, ndim))

        pad = np.fromfile(f, np.uint8, 1)[0]

        packed = np.fromfile(f, np.uint8)

    packed = packed.reshape(-1, 5)

    x = np.empty((len(packed), 4), dtype=np.uint16)

    x[:, 0] = (
        packed[:, 0].astype(np.uint16)
        | ((packed[:, 1].astype(np.uint16) & 0x03) << 8)
    )

    x[:, 1] = (
        ((packed[:, 1].astype(np.uint16) >> 2) & 0x3F)
        | ((packed[:, 2].astype(np.uint16) & 0x0F) << 6)
    )

    x[:, 2] = (
        ((packed[:, 2].astype(np.uint16) >> 4) & 0x0F)
        | ((packed[:, 3].astype(np.uint16) & 0x3F) << 4)
    )

    x[:, 3] = (
        ((packed[:, 3].astype(np.uint16) >> 6) & 0x03)
        | (packed[:, 4].astype(np.uint16) << 2)
    )

    x = x.ravel()

    if pad:
        x = x[:-pad]

    return x.reshape(shape)