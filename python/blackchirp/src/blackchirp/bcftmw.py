from __future__ import annotations


import os
import warnings
import numpy as np
import pandas as pd
from .bcfid import BCFid


class BCFTMW:
    """Storage object for CP-FTMW data

    The ``BCFTMW`` class loads FID and processing data for CP-FTMW experiments.
    It provides functions for accessing FIDs and for performing processing
    tasks that involve analyzing multiple FIDs in a single operation (e.g., sideband
    deconvolution in an LO Scan experiment).

    This class is not meant to be initialized directly by the user; it is created
    automatically by BCExperiment.


    Args:
        path: Path to fid directory
        sep: CSV delimiter

    Examples:

        Loading an experiment and reading a FID (0.csv)::

            from blackchirp import *

            exp = BCExperiment('path/to/experiment')

            # The BCFTMW object is exp.ftmw if the experiment contains CP-FTMW data.

            fid = exp.ftmw.get_fid()

        Performing upper sideband deconvolution, harmonic mean, FT range 0.1-1.0 GHz
        per segment::

            x, y = exp.ftmw.process_sideband(which='upper',
                                             avg='harmonic',
                                             min_ft_offset=100.0,
                                             max_ft_offset=1000.0)

    Attributes:

        fidparams (DataFrame): Contents of fidparams.csv file
        processing (DataFrame): Contents of proessing.csv file
        numfids (int): Number of FIDs specified in fidparams.csv

    """

    def __init__(self, path: str, sep: str):
        """Loads fidparams.csv and processing.csv

        Args:
            path: Path to fid directory
            sep: CSV delimiter
        """
        self.path = path
        self._sep = sep
        self.fidparams = pd.read_csv(
            os.path.join(self.path, "fid/fidparams.csv"),
            sep=self._sep,
            header=0,
            index_col=0,
        )
        self.proc = {}
        try:
            pdf = pd.read_csv(
                os.path.join(self.path, "fid/processing.csv"),
                sep=self._sep,
                header=0,
                index_col=0,
            )
            for r in pdf.itertuples():
                self.proc[r[0]] = r[1]
        except FileNotFoundError:
            pass

        self.numfids = len(self.fidparams)

    def get_fid(self, num: int = 0) -> BCFid:
        """Loads a (potentially multi-frame) FID from disk.

        For standard acquisition modes (Target Shots, Target Time, and Forever),
        the complete FID is stored as ``0.csv`` and would correspond to ``num=0``, the
        default. Any other CSV files are backups and are accessed by providing the
        desired backup number for ``num``.

        For other acquisition modes that consist of several segments (LO Scan, DR Scan),
        ``num`` corresponds to the desired segment.

        Args:
            num: Number of FID file to load.

        Returns:
            A BCFid object containing the requested FID

        Raises ValueError:
            If num is out of range.


        """
        if num >= self.numfids:
            raise ValueError(
                f"Invalid FID number ({num}). Must be between 0 and {self.numfids-1}"
            )
        return BCFid(num, self.path, self.fidparams, self._sep, self.proc)

    def process_sideband(
        self,
        which: str = "both",
        avg: str = "harmonic",
        min_ft_offset: float = None,
        max_ft_offset: float = None,
        frame: int = 0,
        verbose: bool = False,
        **proc_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs sideband deconvolution on an LO Scan experiment

        See `Sideband Deconvolution <../user_guide/cp-ftmw.html#sideband-deconvolution>`_
        for an explanation of the purpose of the algorithm. The arguments correspond
        to the options available in the Sideband Processing and FID processing menus
        in Blackchirp.

        Args:
            which: {'both', 'upper', 'lower'}
                Sideband to use in deconvolution.
            avg: {'harmonic', 'geometric'}
                Whether to use a harmonic or geometric mean for overlapped segments.
            min_ft_offset: Minimum offset frequency for sideband, in MHz.
                Value is relative to the Downconversion LO frequency.
                Defaults to 0.0 MHz.
            max_ft_offset: Maximum offset frequency for sideband, in MHz.
                Value is relative to the Downconversion LO frequency.
                Defaults to the FT bandwidth.
            frame: Frame number (indexed from 0).
                If negative, all frames will be averaged
            verbose: If true, print a progress message during the deconvolution.
            \*\*proc_kwargs: FID processing settings passed as \*\*kwargs to BCFid.ft.

        Returns:
            Frequency and intensity arrays for processed spectrum.

        Raises:
            ValueError: If a provided argument is invalid.
        """

        # figure out range of data and create the global x array.
        min_probe = np.min(self.fidparams.probefreq.to_numpy())
        max_probe = np.max(self.fidparams.probefreq.to_numpy())
        fid = self.get_fid(0)
        x, ft = fid.ft(frame=frame, **proc_kwargs)
        bw = np.abs(np.max(x) - np.min(x))
        ftsp = bw / (len(x) - 1)

        si = 0
        ei = 0
        if min_ft_offset is not None:
            if min_ft_offset <= 0.0:
                raise ValueError("min_ft_offset must be positive")

            si = int(np.ceil(min_ft_offset / ftsp))
        else:
            min_ft_offset = 0.0

        if max_ft_offset is not None:
            if max_ft_offset <= 0.0:
                raise ValueError("max_ft_offset must be positive")

            ei = int(np.floor(max_ft_offset / ftsp))
        else:
            max_ft_offset = bw

        if which == "lower":
            xx = np.arange(min_probe - max_ft_offset, max_probe - min_ft_offset, ftsp)
        elif which == "upper":
            xx = np.arange(min_probe + min_ft_offset, max_probe + max_ft_offset, ftsp)
        elif which == "both":
            xx = np.arange(min_probe - max_ft_offset, max_probe + max_ft_offset, ftsp)
        else:
            raise ValueError("Which must be 'lower', 'upper', or 'both'")

        shots_array = np.zeros_like(xx)
        y_out = np.zeros_like(xx)

        if avg == "harmonic":
            avg_f = BC_harm_mean
        elif avg == "geometric":
            avg_f = BC_geo_mean
        else:
            raise ValueError("Avg must be 'geometric' or 'harmonic'")

        for i in range(self.numfids):
            if verbose:
                print(f"Processing {i+1}/{self.numfids}")
            fid = self.get_fid(i)
            if (len(fid) == 0) or (fid.shots == 0):
                continue

            if frame < 0:
                fid.average_frames()
            x, ft = fid.ft(frame=frame, **proc_kwargs)
            x = x[si:ei]
            ftu = ft.flatten()[si:ei]
            ftl = ftu[::-1]

            if fid.fidparams["sideband"] == 1:
                xl = x[::-1]
                xu = (fid.fidparams.probefreq - x) + fid.fidparams.probefreq
            else:
                t = (fid.fidparams.probefreq - x) + fid.fidparams.probefreq
                xl = t[::-1]
                xu = x

            if (which == "lower") or (which == "both"):
                yint = np.interp(xx, xl, ftl, left=0.0, right=0.0)
                yshots = np.where(yint > 0.0, fid.shots, 0.0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_out = avg_f(y_out, yint, shots_array, yshots)
                shots_array += yshots

            if (which == "upper") or (which == "both"):
                yint = np.interp(xx, xu, ftu, left=0.0, right=0.0)
                yshots = np.where(yint > 0.0, fid.shots, 0.0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_out = avg_f(y_out, yint, shots_array, yshots)
                shots_array += yshots

        return xx, y_out


def BC_harm_mean(
    y1: np.ndarray, y2: np.ndarray, s1: np.ndarray, s2: np.ndarray
) -> np.ndarray:
    return np.where(s1 == 0, y2, np.where(s2 == 0, y1, (s1 + s2) / (s1 / y1 + s2 / y2)))


def BC_geo_mean(
    y1: np.ndarray, y2: np.ndarray, s1: np.ndarray, s2: np.ndarray
) -> np.ndarray:
    return np.where(
        s1 == 0,
        y2,
        np.where(s2 == 0, y1, np.exp((s1 * np.log(y1) + s2 * np.log(y2)) / (s1 + s2))),
    )
