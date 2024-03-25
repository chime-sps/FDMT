"""FDMT."""
from time import time
from typing import Tuple

import numpy as np
import numpy.typing as npt

import ctests.fdmt_cyfuncs as fdmtcy

# Subband delta time or delay time
def subDT(
    freqs: npt.NDArray[np.float64],
    freqs_stepsize: np.float64,
    min_freq_mhz: float,
    max_freq_mhz: float,
    max_time_samples: int,
) -> npt.NDArray[np.int32]:
    """Get needed DT of subband to yield maxDT over entire band

    Args:
        freqs (npt.NDArray[np.float32]): Frequency channels.
        freqs_stepsize (np.float32): Frequency step size.
        min_freq_mhz (float): Minimum frequency.
        max_freq_mhz (float): Maximum frequency.
        max_time_samples (int): Maximum time samples.

    Returns:
        npt.NDArray[np.int32]: Frequency channels.

    Yields:
        Iterator[npt.NDArray[np.int32]]: Frequency channels.
    """
    loc = np.power(freqs, -2.0) - np.power((freqs + freqs_stepsize), -2.0)
    glo = np.power(min_freq_mhz, -2.0) - np.power(max_freq_mhz, -2.0)
    dt: npt.NDArray[np.int32] = np.ceil((max_time_samples - 1) * loc / glo).astype(
        np.int32
    ) + np.int32(1)
    return dt

def buildQ(
    freqs: npt.NDArray[np.float64],
    freqs_stepsize: np.float64,
    freq_channels: int,
    min_freq_mhz: float,
    max_freq_mhz: float,
    max_time_samples: int,
) -> npt.NDArray[np.int32]:
    """Build Q required for FDMT."""
    # Build matrices required for FDMT
    Q = np.zeros((int(np.log2(freq_channels)) + 1, freq_channels), dtype=np.int32)
    for idx in range(int(np.log2(freq_channels)) + 1):
        needed = subDT(
            freqs[:: 2**idx],
            freqs_stepsize * np.power(2, idx),
            min_freq_mhz,
            max_freq_mhz,
            max_time_samples,
        )
        Q[idx, : len(needed)] = np.cumsum(needed) - needed
    return Q

#@numba.njit(boundscheck=False)  # type: ignore
def fdmt(
    spectra: npt.NDArray[np.float32],
    min_freq_mhz: float = 400.1953125,
    max_freq_mhz: float = 800.1953125,
    freq_channels: int = 4096,
    max_time_samples: int = 2048,
    frontpadding: bool = False,
    backpadding: bool = False,
    threads: int = 4,
) -> npt.NDArray[np.float32]:
    """Perform the Fast Dispersion Measure Transform (FDMT).

    Args:
        spectra (npt.NDArray[np.float32]): Intensity spectra.
        min_freq_mhz (float, optional): Minimum Frequency.
            Defaults to 400.1953125.
        max_freq_mhz (float, optional): Maximum Frequency.
            Defaults to 800.1953125.
        freq_channels (int, optional): Frequency Channels.
            Defaults to 1024.
        max_time_samples (int, optional): Frequency Channels.
            Defaults to 2048.
        frontpadding (bool, optional): Whether to pad the front.
            Defaults to True.
        backpadding (bool, optional): Whether to pad the back.
            Defaults to False.
        threads (int, optional): Number of Numba threads to use.
            Defaults to 1.

    Returns:
        npt.NDArray[np.float32]: Dedispersed time series.
    """
    ### removed like
    freqs: npt.NDArray[np.float32] = np.zeros(
        freq_channels, dtype=np.float32, #like=np.empty_like(np.float32)
    )
    freqs_stepsize: np.float32 = np.float32(np.NAN)
    # Compute Frequencies and Frequency Step Size

    ### removed retstep, endpoint, dtype
    freqs = np.linspace(
        min_freq_mhz,
        max_freq_mhz,
        freq_channels,
    )
    freqs_stepsize = freqs[1] - freqs[0]

    chDTs = subDT(freqs, freqs_stepsize, min_freq_mhz, max_freq_mhz, max_time_samples)

    # Build matrices required for FDMT
    columns = spectra.shape[1]
    rows_A = chDTs.sum(axis=0, dtype=np.int32)  # type: ignore
    ### numba.empty requires tuple for 2D array axes instead of a list
    A: npt.NDArray[np.float32] = np.zeros((rows_A, columns), dtype=np.float32)
    rows_B = subDT(freqs[::2], freqs_stepsize * 2, min_freq_mhz, max_freq_mhz, max_time_samples).sum(axis=0, dtype=np.int32)  # type: ignore
    B: npt.NDArray[np.float32] = np.zeros((rows_B, columns), dtype=np.float32)
    # A and B are the matrices used in the FDMT algorithm
    Q = buildQ(
        freqs=freqs,
        freqs_stepsize=freqs_stepsize,
        freq_channels=freq_channels,
        min_freq_mhz=min_freq_mhz,
        max_freq_mhz=max_freq_mhz,
        max_time_samples=max_time_samples,
    )

    A[Q[0], :] = spectra
    commonDTs: npt.NDArray[np.int32] = np.ones(chDTs.min() - 1, dtype=np.int32) * spectra.shape[1]  # type: ignore
    DTsteps: npt.NDArray[np.int32] = np.where(chDTs[:-1] - chDTs[1:] != 0)[0]
    DTplan: npt.NDArray[np.int32] = np.concatenate( (commonDTs, DTsteps[::-1]) )

    """for i, t in enumerate(DTplan, 1):
        A[Q[0][:t] + i, i:] = A[Q[0][:t] + i - 1, i:] + spectra[:t, :-i]
    for i, t in enumerate(DTplan, 1):
        # A[Q[0][:t]+i,i:] /= int(i+1)
        A[Q[0][:t] + i, i:] /= int(i + 1)"""

    print(A.dtype, B.dtype, Q.dtype, DTplan.dtype, spectra.dtype)
    A = fdmtcy.buildA(A, B, Q, spectra, DTplan)

    for i in range(1, int(np.log2(freq_channels)) + 1):
        src, dest = (A, B) if (i % 2 == 1) else (B, A)
        dest = fdmtcy.fdmt_iter_par(
            fs=freqs,
            nchan=freq_channels,
            df=freqs_stepsize,
            Q=Q,
            src=src,
            dest=dest,
            i=i,
            fmin=min_freq_mhz,
            fmax=max_freq_mhz,
            maxDT=max_time_samples,
            threads=threads,
        )

    return dest[:max_time_samples]#[:, max_time_samples:]


if __name__ == "__main__":
    data = np.random.normal(size=(4096, 40960))
    data = data.astype(np.float32)

    max_time_samples = 10484
    padding = ((0, 0), (0, max_time_samples))
    data = np.pad(data, padding, mode="constant", constant_values=np.float32(0.0))

    start = time()
    fdmt(data, max_time_samples=max_time_samples)
    end = time()
    print(f"Iteration: {end - start}s")
