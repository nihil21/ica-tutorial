"""Functions for visualizations.


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

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from .._base import Signal

sns.set_style("whitegrid")


def plot_signal(
    s: Signal,
    fs: float = 1.0,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s : Signal
        A signal with shape (n_samples, n_channels).
    fs : float, default=1.0
        Sampling frequency of the signal (relevant if s is a NumPy array).
    title : str or None, default=None
        Title of the plot.
    x_label : str, default="Time [s]"
        Label for X axis.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Convert signal to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    elif isinstance(s, pd.Series):
        s_df = s.to_frame()
    else:
        s_array = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        if len(s_array.shape) == 1:
            s_array = s_array.reshape(-1, 1)
        s_df = pd.DataFrame(s_array, index=np.arange(s_array.shape[0]) / fs)

    # Create figure with subplots and shared X axis
    n_cols = 1
    n_rows = s_df.shape[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )
    axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes
    # Set title and label of X and Y axes
    if title is not None:
        fig.suptitle(title, fontsize="xx-large")
    fig.supxlabel(x_label)
    fig.supylabel(y_label)

    # Plot signal
    for i, ch_i in enumerate(s_df):
        axes[i].plot(s_df[ch_i])

    # Show or save plot
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_correlation(
    s: Signal,
    write_annotations: bool = False,
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the correlation matrix between the channels of a given signal.

    Parameters
    ----------
    s : Signal
        A signal with shape (n_samples, n_channels).
    write_annotations : bool, default=False
        Whether to write annotations inside the correlation matrix or not.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Convert to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    else:
        s_array = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        s_df = pd.DataFrame(s_array)

    # Compute correlation and plot heatmap
    corr = s_df.corr()
    _, ax = plt.subplots(figsize=fig_size, layout="constrained")
    sns.heatmap(
        corr,
        vmax=1.0,
        vmin=-1.0,
        cmap="icefire",
        annot=write_annotations,
        square=True,
        ax=ax,
    )

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
