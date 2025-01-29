import ast

import numcodecs
import numpy as np


class AsciiTableCodec(numcodecs.abc.Codec):
    """Decodes ASCII-TABLE extensions in FITS files"""

    codec_id = "FITSAscii"

    def __init__(self, indtypes, outdtypes):
        """

        Parameters
        ----------
        indtypes: list[str]
            dtypes of the fields as in the table
        outdtypes: list[str]
            requested final dtypes
        """
        self.indtypes = indtypes
        self.outdtypes = outdtypes

    def decode(self, buf, out=None):
        indtypes = np.dtype([tuple(d) for d in self.indtypes])
        outdtypes = np.dtype([tuple(d) for d in self.outdtypes])
        arr = np.frombuffer(buf, dtype=indtypes)
        return arr.astype(outdtypes)

    def encode(self, _):
        pass


class VarArrCodec(numcodecs.abc.Codec):
    """Variable length arrays in a FITS BINTABLE extension"""

    codec_id = "FITSVarBintable"
    # https://heasarc.gsfc.nasa.gov/docs/software/fitsio/quick/node10.html
    ftypes = {"B": "uint8", "I": ">i2", "J": ">i4"}

    def __init__(self, dt_in, dt_out, nrow, types):
        self.dt_in = dt_in
        self.dt_out = dt_out
        self.nrow = nrow
        self.types = types

    def encode(self, _):
        raise NotImplementedError

    def decode(self, buf, out=None):
        dt_in = np.dtype(ast.literal_eval(self.dt_in))
        dt_out = np.dtype(ast.literal_eval(self.dt_out))
        arr = np.frombuffer(buf, dtype=dt_in, count=self.nrow)
        arr2 = np.empty((self.nrow,), dtype=dt_out)
        heap = buf[arr.nbytes :]
        for name in dt_out.names:

            if dt_out[name] == "O":
                dt = np.dtype(self.ftypes[self.types[name]])
                counts = arr[name][:, 0]
                offs = arr[name][:, 1]
                for i, (off, count) in enumerate(zip(offs, counts)):
                    arr2[name][i] = np.frombuffer(
                        heap[off : off + count * dt.itemsize], dtype=dt
                    )
            else:
                arr2[name][:] = arr[name][:]
        return arr2
