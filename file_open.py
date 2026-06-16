import numpy as np

def thermal_file_open(
    thermal_file_path,
    rows=240,
    cols=320,
    bands=3000,
    dtype=np.float64,
    offset=0,
    byteorder="<"
):
    """
    Read raw thermal imaging data stored in BSQ format and return an array
    with shape (rows, cols, bands).

    Parameters
    ----------
    thermal_file_path : str
        Path to the raw thermal data file.
    rows : int
        Image height.
    cols : int
        Image width.
    bands : int
        Number of frames.
    dtype : data-type
        Data type of the raw thermal file.
    offset : int
        Byte offset before reading the file.
    byteorder : str
        Byte order. "<" means little-endian.

    Returns
    -------
    data : numpy.ndarray
        Thermal data with shape (rows, cols, bands).
    """
    with open(thermal_file_path, "rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=byteorder + "f8")
        data = data.reshape((bands, rows, cols))
        data = np.transpose(data, (1, 2, 0))

    return data