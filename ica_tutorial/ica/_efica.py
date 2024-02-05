"""Function and class implementing the EFICA algorithm
(https://doi.org/10.1109/TNN.2006.875991).


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

import logging
import warnings
from functools import partial
from math import sqrt
from typing import Callable

import torch

from .._base import Signal, signal_to_tensor
from ..preprocessing import PCAWhitening, WhiteningModel, ZCAWhitening
from . import _contrast_functions as cf
from ._abc_ica import ICA
from ._utils import ConvergenceWarning, sym_orth


def _gg_score_function(
    u: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score function for Generalized Gaussian distributions."""

    u_tmp = u.abs() ** (alpha - 2)
    g_u = u * u_tmp
    g1_u = (alpha - 1) * u_tmp

    return g_u, g1_u


def _exp1(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 'exp1' non-linearity."""

    eta = 3.3476
    u_tmp = torch.exp(-eta * u.abs())
    g_u = u * u_tmp
    g1_u = (1 - eta * u.abs()) * u_tmp

    return g_u, g1_u


class EFICA(ICA):
    """Class implementing EFICA.

    Parameters
    ----------
    n_ics : int or str, default="all"
        Number of components to estimate:
        - if set to the string "all", it will be set to the number of channels in the signal;
        - otherwise, it will be set to the given number.
    whiten_alg : {"zca", "pca", "none"}, default="zca"
        Whitening algorithm.
    g_name : {"logcosh", "gauss", "kurtosis", "rati", "exp1"}, default="logcosh"
        Name of the contrast function for FastICA.
    w_init : Tensor or None, default=None
        Initial separation matrix with shape (n_components, n_channels).
    conv_th : float, default=1e-4
        Threshold for convergence for symmetric FastICA.
    conv_th_ft : float, default=1e-5
        Threshold for convergence for fine-tuning.
    max_iter : int, default=200
        Maximum n. of iterations for symmetric FastICA.
    max_iter_ft : int, default=50
        Maximum n. of iterations for fine-tuning.
    device : device or str or None, default=None
        Torch device.
    seed : int or None, default=None
        Seed for the internal PRNG.
    **kwargs
        Keyword arguments forwarded to whitening algorithm.

    Attributes
    ----------
    _n_ics : int
        Number of components to estimate.
    _g_func : ContrastFunction
        Contrast function.
    _conv_th : float
        Threshold for convergence for symmetric FastICA.
    _conv_th_ft : float
        Threshold for convergence for fine-tuning.
    _max_iter : int
        Maximum n. of iterations for symmetric FastICA.
    _max_iter_ft : int
        Maximum n. of iterations for fine-tuning.
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        n_ics: int | str = "all",
        whiten_alg: str = "zca",
        g_name: str = "logcosh",
        w_init: torch.Tensor | None = None,
        conv_th: float = 1e-4,
        conv_th_ft: float = 1e-5,
        max_iter: int = 200,
        max_iter_ft: int = 50,
        device: torch.device | str | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        assert (isinstance(n_ics, int) and n_ics > 0) or (
            isinstance(n_ics, str) and n_ics == "all"
        ), 'n_ics must be either a positive integer or "all".'
        assert whiten_alg in (
            "zca",
            "pca",
            "none",
        ), f'Whitening can be either "zca", "pca" or "none": the provided one was "{whiten_alg}".'
        assert g_name in (
            "logcosh",
            "gauss",
            "kurtosis",
            "skewness",
            "rati",
        ), (
            'Contrast function can be either "logcosh", "gauss", "kurtosis", "skewness" or "rati": '
            f'the provided one was "{g_name}".'
        )
        assert conv_th > 0, "Convergence threshold for FastICA must be positive."
        assert conv_th_ft > 0, "Convergence threshold for fine-tuning must be positive."
        assert (
            max_iter > 0
        ), "The maximum n. of iterations for FastICA must be positive."
        assert (
            max_iter_ft > 0
        ), "The maximum n. of iterations for fine-tuning must be positive."

        logging.info(f'Instantiating EFICA using "{g_name}" contrast function.')

        self._device = torch.device(device) if isinstance(device, str) else device

        # Whitening model
        whiten_dict = {
            "zca": ZCAWhitening,
            "pca": PCAWhitening,
            "none": lambda **_: None,
        }
        whiten_kw = kwargs
        whiten_kw["device"] = self._device
        self._whiten_model: WhiteningModel | None = whiten_dict[whiten_alg](**whiten_kw)

        # Weights
        self._sep_mtx: torch.Tensor = None  # type: ignore
        if w_init is not None:
            self._sep_mtx = w_init.to(self._device)
            self._n_ics = w_init.size(0)
        else:
            self._n_ics = 0 if n_ics == "all" else n_ics  # map "all" -> 0

        g_dict = {
            "logcosh": cf.logcosh,
            "gauss": cf.gauss,
            "kurtosis": cf.kurtosis,
            "skewness": cf.skewness,
            "rati": cf.rati,
        }
        self._g_func = g_dict[g_name]
        self._conv_th = conv_th
        self._conv_th_ft = conv_th_ft
        self._max_iter = max_iter
        self._max_iter_ft = max_iter_ft

        if seed is not None:
            torch.manual_seed(seed)

    @property
    def sep_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated separation matrix."""
        return self._sep_mtx

    @property
    def whiten_model(self) -> WhiteningModel | None:
        """WhiteningModel or None: Property for getting the whitening model."""
        return self._whiten_model

    def decompose_training(self, x: Signal) -> torch.Tensor:
        """Train the ICA model to decompose the given signal into independent components (ICs).

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Estimated ICs with shape (n_samples, n_components).

        Warns
        -----
        ConvergenceWarning
            The algorithm didn't converge.
        """
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device)

        # Whitening
        if self._whiten_model is not None:
            x_tensor = self._whiten_model.whiten_training(x_tensor)
        x_tensor = x_tensor.T

        n_ch, n_samp = x_tensor.size()
        if self._n_ics == 0:
            self._n_ics = n_ch
        assert (
            n_ch >= self._n_ics
        ), f"Too few channels ({n_ch}) with respect to target components ({self._n_ics})."

        if self._sep_mtx is None:
            self._sep_mtx = torch.randn(
                self._n_ics, n_ch, dtype=x_tensor.dtype, device=self._device
            )

        w_no_decorr = self._sep_mtx
        w = sym_orth(w_no_decorr)

        # 1. Get initial estimation using symmetric FastICA + saddle point test
        saddle_test_done = False
        rot_mtx = 1 / torch.as_tensor(
            [[sqrt(2), -sqrt(2)], [sqrt(2), sqrt(2)]],
            dtype=x_tensor.dtype,
            device=self._device,
        )
        rotated = torch.zeros(self._n_ics, dtype=torch.bool)
        while True:
            iter_idx = 1
            converged = False
            while iter_idx <= self._max_iter:
                g_res = self._g_func(w @ x_tensor)
                w_new_no_decorr = (
                    g_res.g1_u @ x_tensor.T / n_samp
                    - g_res.g2_u.mean(dim=1, keepdim=True) * w
                )
                w_new = sym_orth(w_new_no_decorr)

                # Compute distance:
                # 1. Compute absolute dot product between old and new separation vectors (i.e., the rows of W)
                distance = torch.abs(torch.einsum("ij,ij->i", w, w_new))
                # 2. Absolute dot product should be close to 1, thus subtract 1 and take absolute value
                distance = torch.abs(distance - 1)
                # 3. Consider maximum distance
                distance = torch.max(distance).item()
                logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")

                w_no_decorr = w_new_no_decorr
                w = w_new

                if distance < self._conv_th:
                    converged = True
                    logging.info(
                        f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                    )
                    break

                iter_idx += 1
            if not converged:
                warnings.warn("FastICA didn't converge.", ConvergenceWarning)

            if saddle_test_done:
                break

            logging.info("Performing saddle test...")
            ics = w @ x_tensor
            ics_g_ret = self._g_func(ics)
            ics_score = (ics_g_ret.g_u.mean(dim=1) - ics_g_ret.g_nu) ** 2
            # Check each pair that has not already been rotated
            positive = False
            for i in range(self._n_ics):
                for j in range(i + 1, self._n_ics):
                    if torch.all(~rotated[[i, j]]):
                        # Rotate pair and compute score
                        rot_ics = rot_mtx @ ics[[i, j]]
                        rot_ics_g_ret = self._g_func(rot_ics)
                        rot_ics_score = (
                            rot_ics_g_ret.g_u.mean(dim=1) - rot_ics_g_ret.g_nu
                        ) ** 2
                        # If the score of rotated ICs is higher, apply rotation
                        if rot_ics_score.max() > ics_score[[i, j]].max():
                            w_no_decorr[[i, j]] = rot_mtx @ w_no_decorr[[i, j]]
                            w[[i, j]] = rot_mtx @ w[[i, j]]
                            rotated[[i, j]] = True
                            positive = True

            if positive:
                logging.info(
                    "Some ICs were found to be positive at saddle point test, refining..."
                )
                saddle_test_done = True
            else:
                logging.info("Saddle point test ok.")
                break

        # 2-3. Adaptive choice of non-linearities + Refinement 1
        emp_kurt = ((w @ x_tensor) ** 4).mean(dim=1)
        mu = torch.zeros(
            self._n_ics,
            dtype=x_tensor.dtype,
            device=self._device,
        )
        rho = torch.zeros(
            self._n_ics,
            dtype=x_tensor.dtype,
            device=self._device,
        )
        beta = torch.zeros(
            self._n_ics,
            dtype=x_tensor.dtype,
            device=self._device,
        )
        for k in range(self._n_ics):
            if emp_kurt[k] > 3:  # super-Gaussian
                logging.info(f"IC {k}: super-Gaussian (kurtosis = {emp_kurt[k]:.3e}).")
                g_k = _exp1
            else:  # sub-Gaussian
                logging.info(f"IC {k}: sub-Gaussian (kurtosis = {emp_kurt[k]:.3e}).")
                if emp_kurt[k] <= 1.8:  # uniform
                    g_k = partial(_gg_score_function, alpha=15)
                else:  # not uniform
                    tmp = emp_kurt[k].item() - 1.8
                    emp_alpha = 1 / (0.2906 * sqrt(tmp) - 0.1851 * tmp)
                    g_k = partial(_gg_score_function, alpha=min(emp_alpha, 15))

            w_k_new, w_k_new_normal, uncorrelated = self._fine_tuning(
                x_tensor, g_k, w[k]
            )
            if uncorrelated:
                logging.info(
                    f"IC {k}: new weight too distant from the previous one, keeping the latter..."
                )
                w[k] = w_no_decorr[k]
                s_k = w_k_new_normal @ x_tensor
                g_res = self._g_func(s_k)
                mu[k] = s_k @ g_res.g1_u / n_samp
                rho[k] = g_res.g2_u.mean()
                beta[k] = (g_res.g1_u**2).mean()
            else:
                w[k] = w_k_new
                s_k = w_k_new_normal @ x_tensor
                g_u, g1_u = g_k(s_k)
                mu[k] = s_k @ g_u / n_samp
                rho[k] = g1_u.mean()
                beta[k] = (g_u**2).mean()

        # 4. Refinement 2
        tau = (mu - rho).abs()
        gamma = beta - mu**2
        self._sep_mtx = torch.zeros_like(w)
        for k in range(self._n_ics):
            c = tau * gamma[k] / (tau[k] * (gamma + tau**2))
            c[k] = 1
            w_k = torch.diag(c) @ w
            w_k = sym_orth(w_k)
            self._sep_mtx[k] = w_k[k]
        ics = self._sep_mtx @ x_tensor

        return ics.T

    def decompose_inference(self, x: Signal) -> torch.Tensor:
        """Decompose the given signal into independent components (ICs) using the frozen ICA model.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Estimated ICs with shape (n_samples, n_components).
        """
        assert self._sep_mtx is not None, "Fit the model first."

        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device)

        # Decompose signal
        if self._whiten_model is not None:
            ics = self._whiten_model.whiten_inference(x_tensor) @ self._sep_mtx.T
        else:
            ics = x_tensor @ self._sep_mtx.T

        return ics

    def _fine_tuning(
        self,
        x: torch.Tensor,
        g_k: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        w_k_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Helper method for the one-unit fine-tuning."""

        w_k = w_k_init
        w_k_normal = w_k_init
        iter_idx = 1
        converged = False
        uncorrelated = False
        while iter_idx <= self._max_iter_ft:
            w_k /= torch.linalg.norm(w_k)
            g_u, g1_u = g_k(w_k @ x)
            w_k_new = (x * g_u).mean(dim=1) - g1_u.mean() * w_k
            w_k_new_normal = w_k_new / torch.linalg.norm(w_k_new)

            distance = 1 - abs((w_k_new_normal @ w_k).item())
            correlation = abs((w_k_new_normal @ w_k_init).item())
            logging.info(
                f"Fine-tuning iteration {iter_idx}: distance is {distance:.3e}, "
                f"correlation is {correlation:.2f}."
            )

            w_k = w_k_new
            w_k_normal = w_k_new_normal

            if distance < self._conv_th_ft:
                converged = True
                logging.info(
                    f"Fine-tuning converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break
            if correlation < 0.95:
                converged = True
                uncorrelated = True
                break

            iter_idx += 1

        if not converged:
            warnings.warn(
                "Fine-tuning didn't converge for current component.",
                ConvergenceWarning,
            )

        return w_k, w_k_normal, uncorrelated
