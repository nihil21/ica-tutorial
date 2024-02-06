"""Utility functions.


Copyright 2023 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import warnings
from math import ceil

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.cluster.vq import kmeans2
from sklearn.metrics import silhouette_score

from .._base import Signal, signal_to_array


def eigendecomposition(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform eigendecomposition of the covariance matrix of a given Tensor.

    Parameters
    ----------
    m: Tensor
        Square Tensor.

    Returns
    -------
    Tensor:
        2D Tensor of eigenvectors sorted by the corresponding eigenvalue in descending order.
    Tensor:
        1D Tensor of eigenvalues sorted in descending order.
    """
    eig_vals, eig_vecs = torch.linalg.eigh(m)

    # Improve numerical stability
    eps = torch.finfo(eig_vals.dtype).eps
    degenerate_idx = torch.lt(eig_vals, eps).nonzero()
    if torch.any(degenerate_idx):
        warnings.warn(f"Some eigenvalues are smaller than epsilon ({eps:.3e}).")
    eig_vals[degenerate_idx] = eps
    eig_vecs *= torch.sign(eig_vecs[0])  # guarantee consistent sign

    sort_idx = torch.argsort(eig_vals, descending=True)
    eig_vals, eig_vecs = eig_vals[sort_idx], eig_vecs[:, sort_idx]

    return eig_vecs, eig_vals


def power_spectrum(x: Signal, fs: float) -> pd.DataFrame:
    """Compute the power spectrum of a given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    fs : float
        Sampling frequency.

    Returns
    -------
    DataFrame
        Power spectrum for each channel.
    """

    # Convert to array
    x_array = signal_to_array(x, allow_1d=True).T

    n_ch, n_samp = x_array.shape
    spec_len = n_samp // 2 + 1
    # Compute frequencies
    freqs = np.fft.rfftfreq(n_samp, 1 / fs)
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    # Compute power spectrum channel-wise
    pow_spec = np.zeros(shape=(spec_len, n_ch))
    for i in range(n_ch):
        # Compute FFT for current channel
        ch_fft = np.fft.rfft(x_array[i])
        # Compute power spectrum for current channel
        ch_pow_spec = np.abs(ch_fft) ** 2
        pow_spec[:, i] = ch_pow_spec[idx]

    return pd.DataFrame(pow_spec, index=freqs)


def _compute_delay(s1: np.ndarray, s2: np.ndarray) -> int:
    """Find the lag between two signals with the same length."""
    # Compute cross-correlation
    corr = signal.correlate(s2, s1, mode="same")
    delay_steps = int(round(s1.shape[0] / 2))
    delay_arr = np.arange(-delay_steps, delay_steps)

    # Return optimal delay
    return delay_arr[np.argmax(corr)].item()


def check_delayed_pair(
    ref_pulses_bin: np.ndarray,
    sec_pulses_bin: np.ndarray,
    fs: float,
    tol_ms: float,
    min_perc: float,
) -> tuple[bool, int, int, int, int]:
    """Check if two pulse trains are the same up to a delay by counting the common pulses.

    Parameters
    ----------
    ref_pulses_bin : ndarray
        Reference pulse train represented as an array of 1s and 0s with shape (n_pulses,).
    sec_pulses_bin : ndarray
        Secondary pulse train represented as an array of 1s and 0s with shape (n_pulses,).
    fs : float
        Sampling frequency of the pulse trains.
    tol_ms : float
        Tolerance for considering two pulses as synchronized.
    min_perc : float
        Minimum percentage of common pulses for considering the two pulse trains as the same.

    Returns
    -------
    bool
        Whether the two pulse trains are the same or not.
    int
        Number of samples representing the lag between the pulse trains.
    int
        Number of TPs if the pulse trains are the same, zero otherwise.
    int
        Number of FPs if the pulse trains are the same, zero otherwise.
    int
        Number of FNs if the pulse trains are the same, zero otherwise.
    """
    assert (
        ref_pulses_bin.shape == sec_pulses_bin.shape
    ), "The two pulse trains must have the same length."
    assert len(ref_pulses_bin.shape) == 1, "The pulse trains must be 1D."

    # Find delay between reference and secondary pulse trains
    delay = _compute_delay(ref_pulses_bin, sec_pulses_bin)

    # Adjust for delay and get time of pulses
    ref_pulses_t = np.flatnonzero(ref_pulses_bin) / fs
    sec_pulses_t = (np.flatnonzero(sec_pulses_bin) - delay) / fs  # compensate for delay

    # Filter secondary pulses
    n_sec = sec_pulses_t.size
    sec_pulses_t = sec_pulses_t[sec_pulses_t >= 0]

    if ref_pulses_t.size == 0 or sec_pulses_t.size == 0:
        return False, delay, 0, 0, 0

    # Check pulse correspondence and count TP, FP and FN
    tol_s = tol_ms / 1000
    tp, fn = 0, 0
    for ref_pulse_t in ref_pulses_t:
        common_pulses = np.count_nonzero(
            (sec_pulses_t >= ref_pulse_t - tol_s)
            & (sec_pulses_t <= ref_pulse_t + tol_s)
        )
        if common_pulses == 0:  # no pulses found near the reference pulse -> one FN
            fn += 1
        elif common_pulses == 1:  # one pulse found near the reference pulse -> one TP
            tp += 1
    # The difference between the n. of secondary pulses and
    # the n. of TPs yields the n. of FPs
    fp = n_sec - tp

    # The pulse trains are the same if TPs > 30%
    same1 = tp / ref_pulses_t.size > min_perc
    same2 = tp / n_sec > min_perc

    return same1 and same2, delay, tp, fp, fn


def find_replicas(
    pulse_trains: Signal,
    fs: float,
    tol_ms: float,
    min_perc: float,
) -> dict[int, list[int]]:
    """Given a set of pulse trains, find delayed replicas by checking each pair.

    Parameters
    ----------
    pulse_trains : Signal
        Set of pulse trains represented as arrays of 1s and 0s with shape (n_samples, n_trains).
    fs : float
        Sampling frequency of the pulse trains.
    tol_ms : float
        Tolerance for considering two pulses as synchronized.
    min_perc : float
        Minimum percentage of common pulses for considering the two pulse trains as the same.

    Returns
    -------
    dict of (int: list of int)
        Dictionary containing delayed replicas.
    """
    # Convert to array
    pulse_trains_array = signal_to_array(pulse_trains)
    n_trains = pulse_trains_array.shape[1]

    # Convert to dictionary
    pulse_train_dict = {i: pulse_trains_array[:, i] for i in range(n_trains)}

    # Check each pair
    cur_tr = 0
    tr_idx = list(pulse_train_dict.keys())
    duplicate_tr = {}
    while cur_tr < len(tr_idx):
        # Find index of replicas by checking synchronization
        i = 1
        while i < len(tr_idx) - cur_tr:
            # Find delay in binarized sources
            same = check_delayed_pair(
                ref_pulses_bin=pulse_train_dict[tr_idx[cur_tr]],
                sec_pulses_bin=pulse_train_dict[tr_idx[cur_tr + i]],
                fs=fs,
                tol_ms=tol_ms,
                min_perc=min_perc,
            )[0]

            if same:
                duplicate_tr[tr_idx[cur_tr]] = duplicate_tr.get(tr_idx[cur_tr], []) + [
                    tr_idx[cur_tr + i]
                ]
                del tr_idx[cur_tr + i]
            else:
                i += 1
        cur_tr += 1

    return duplicate_tr


def _find_threshold(peaks: np.ndarray, th_init: float) -> float:
    """Find optimal spike/noise threshold iteratively."""
    max_iter = 10
    conv_th = 1e-4
    prev_th = new_th = th_init
    for i in range(max_iter):
        th_h = peaks[peaks >= prev_th].mean()
        th_l = peaks[peaks < prev_th].mean()
        new_th = (th_h + th_l) / 2

        if abs(new_th - prev_th) < conv_th:
            break

        prev_th = new_th

    return new_th


def _otsu_score(x: np.ndarray, th: float) -> float:
    """Compute Otsu's score given an array and a threshold."""
    x_bin = np.zeros_like(x)
    x_bin[x >= th] = 1
    n_tot = x.size
    n_bin = np.count_nonzero(x_bin)
    w1 = n_bin / n_tot
    w0 = 1 - w1
    if w1 == 0 or w0 == 0:
        return np.inf

    x0 = x[x_bin == 0]
    x1 = x[x_bin == 1]
    var0 = np.var(x0).item()
    var1 = np.var(x1).item()
    return w0 * var0 + w1 * var1


def detect_spikes(
    ic: Signal,
    ref_period: int | None = None,
    bin_alg: str = "kmeans",
    threshold: float | None = None,
    compute_sil: bool = False,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float]:
    """Detect spikes in the given IC.

    Parameters
    ----------
    ic : Signal
        Estimated IC with shape (n_samples,).
    ref_period : int or None, default=None
        Refractory period for spike detection.
    bin_alg : {"kmeans", "otsu"}, default="kmeans"
        Binarization algorithm.
    threshold : float or None, default=None
        Threshold for spike/noise classification.
    compute_sil : bool, default=False
        Whether to compute SIL measure or not.
    seed : int or Generator or None, default=None
        Seed for PRNG.

    Returns
    -------
    ndarray
        Location of spikes.
    float
        Threshold for spike/noise classification.
    float
        SIL measure.
    """
    assert bin_alg in (
        "kmeans",
        "otsu",
    ), f'The binarization algorithm can be either "kmeans" or "otsu": the provided one was {bin_alg}.'

    # Convert to array
    ic_array = signal_to_array(
        ic, allow_1d=True
    ).flatten()  # find_peaks expects a 1D array

    peaks, _ = signal.find_peaks(ic_array, height=0, distance=ref_period)
    ic_peaks = ic_array[peaks]

    if threshold is None:
        if bin_alg == "kmeans":
            centroids, labels = kmeans2(
                ic_peaks.reshape(-1, 1), k=2, minit="++", seed=seed
            )
            high_cluster_idx = np.argmax(centroids)  # consider only high peaks
            spikes = peaks[labels == high_cluster_idx]
            threshold_new = centroids.mean()
        else:
            th_range = np.linspace(0, ic_peaks.max())
            otsu_scores = np.asarray([_otsu_score(ic_peaks, th) for th in th_range])
            threshold_new = _find_threshold(
                ic_peaks, th_init=th_range[np.argmin(otsu_scores)].item()
            )
            labels = ic_peaks >= threshold_new
            spikes = peaks[labels]
    else:
        threshold_new = threshold
        labels = ic_peaks >= threshold_new
        spikes = peaks[labels]

    sil = np.nan
    if compute_sil:
        sil = float(silhouette_score(ic_peaks.reshape(-1, 1), labels))

    return spikes, threshold_new, sil


def sparse_to_dense(
    spikes_t: dict[str, np.ndarray],
    sig_len_s: float,
    fs: float,
) -> pd.DataFrame:
    """Convert a DataFrame of spikes from sparse to dense format.

    Parameters
    ----------
    spikes_t : dict of {str : ndarray}
        Dictionary containing the discharge times for each IC.
    sig_len_s : float
        Length of the signal (in seconds).
    fs : float
        Sampling frequency.

    Returns
    -------
    DataFrame
        Binary DataFrame with shape (n_samples, n_ic) containing either ones or zeros (spike/not spike).
    """
    n_ic = len(spikes_t.keys())
    n_samp = ceil(sig_len_s * fs)
    spikes_bin = pd.DataFrame(
        data=np.zeros(shape=(n_samp, n_ic), dtype="uint8"),
        index=np.arange(n_samp) / fs,
        columns=list(spikes_t.keys()),
    )
    for ic, cur_spikes in spikes_t.items():
        spikes_bin.loc[cur_spikes, ic] = 1

    return spikes_bin


def dense_to_sparse(
    spikes_bin: pd.DataFrame,
    fs: float,
) -> dict[str, np.ndarray]:
    """Convert a DataFrame of spikes from sparse to dense format.

    Parameters
    ----------
    spikes_bin : DataFrame
        Binary DataFrame with shape (n_samples, n_ic) containing either ones or zeros (spike/not spike).
    fs : float
        Sampling frequency.

    Returns
    -------
    dict of {str : ndarray}
        Spike times of each ic.
    """
    spikes_t = {str(ic): np.flatnonzero(spikes_bin[ic]) / fs for ic in spikes_bin}

    return spikes_t
