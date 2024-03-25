cimport numpy as np
import numpy as np

cpdef fdmt_iter_par(np.ndarray[np.float64_t, ndim=1] fs,
                    int nchan,
                    float df,
                    np.ndarray[np.int32_t, ndim=2] Q,
                    np.ndarray[np.float32_t, ndim=2] src,
                    np.ndarray[np.float32_t, ndim=2] dest,
                    int i,
                    float fmin,
                    float fmax,
                    int maxDT,
                    int threads):
    """
    Perform a single iteration of the Fast Dispersion Measure Transform (FDMT)

    Args:
        fs (np.ndarray[np.float64_t]): Array of center frequencies for each channel.
        nchan (int): Number of frequency channels.
        df (float): Frequency resolution.
        Q (np.ndarray[np.int32_t]): List of indices for each frequency channel.
        src (np.ndarray[np.float32_t]): Input data array.
        dest (np.ndarray[np.float32_t]): Output data array.
        i (int): Iteration index.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        maxDT (int): Maximum time delay.
        threads (int): Number of threads to use for parallelization.
    """
    cdef int T
    cdef float dF
    cdef np.ndarray[np.float64_t, ndim=1] f_starts
    cdef np.ndarray[np.float64_t, ndim=1] f_ends
    cdef np.ndarray[np.float64_t, ndim=1] f_mids
    cdef int i_F
    cdef float f0, f1, f2, cor, C, C01, C12, loc, glo
    cdef int R, i_dT, dT_mid01, dT_mid12, dT_rest

    T = src.shape[1]
    dF = df * 2**i
    f_starts = fs[:: int(2**i)]  # Cast the result of 2**i to int
    f_ends = f_starts + dF
    f_mids = fs[int(2 ** (i - 1)) :: int(2**i)]  # Cast the result of 2 ** (i - 1) to int
    for i_F in range(nchan // int(2**i)):  # Cast the result of 2**i to int
        f0 = f_starts[i_F]
        f1 = f_mids[i_F]
        f2 = f_ends[i_F]
        cor = df if i > 1 else 0

        C = (f1**-2 - f0**-2) / (f2**-2 - f0**-2)
        C01 = ((f1 - cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)
        C12 = ((f1 + cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)

        loc = f0**-2 - (f0 + dF) ** -2
        glo = fmin**-2 - fmax**-2
        R = int((maxDT - 1) * loc / glo) + 2

        for i_dT in range(0, R):
            dT_mid01 = int(round(i_dT * C01))  # Cast the result of round to int
            dT_mid12 = int(round(i_dT * C12))  # Cast the result of round to int
            dT_rest = i_dT - dT_mid12
            dest[Q[i, i_F] + i_dT, :] = src[Q[i - 1, 2 * i_F] + dT_mid01, :]
            dest[Q[i, i_F] + i_dT, dT_mid12:] += src[Q[i - 1, 2 * i_F + 1] + dT_rest, : T - dT_mid12]

    return dest


cpdef np.ndarray buildA(np.ndarray A, np.ndarray B, np.ndarray Q, np.ndarray spectra, np.ndarray DTplan):
    cdef int i, t
    for i, t in enumerate(DTplan, 1):
        A[Q[0][:t] + i, i:] = A[Q[0][:t] + i - 1, i:] + spectra[:t, :-i]
    for i, t in enumerate(DTplan, 1):
        A[Q[0][:t] + i, i:] /= int(i + 1)
    return A