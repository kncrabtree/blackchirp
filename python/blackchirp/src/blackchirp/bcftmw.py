from __future__ import annotations


import os
import warnings
import numpy as np
import pandas as pd
from .bcfid import BCFid

_MULTI_SEGMENT_FTMW_TYPES = frozenset({"LO_Scan", "DR_Scan", "Peak_Up"})


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
        ftmw_type: ``FtmwType`` value from ``objectives.csv``. Used to gate
            the differential-FID API to single-segment acquisitions.

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

        Viewing only the shots collected after the second backup, mirroring
        Blackchirp's "differential backup" view::

            diff_fid = exp.ftmw.get_differential_fid(start=2)
            x, y = diff_fid.ft()

    Attributes:

        fidparams (DataFrame): Contents of fidparams.csv file
        proc (dict): Contents of processing.csv file
        numfids (int): Number of FIDs specified in fidparams.csv
        ftmw_type (str): ``FtmwType`` value from ``objectives.csv``
            (e.g. ``"Forever"``, ``"LO_Scan"``).

    """

    def __init__(self, path: str, sep: str, ftmw_type: str = ""):
        """Loads fidparams.csv and processing.csv

        Args:
            path: Path to fid directory
            sep: CSV delimiter
            ftmw_type: FtmwType value from objectives.csv.
        """
        self.path = path
        self._sep = sep
        self.ftmw_type = ftmw_type
        self.fidparams = pd.read_csv(
            os.path.join(self.path, "fid/fidparams.csv"),
            sep=self._sep,
            header=0,
            index_col=0,
            keep_default_na=False,
        )
        self.proc = {}
        try:
            pdf = pd.read_csv(
                os.path.join(self.path, "fid/processing.csv"),
                sep=self._sep,
                header=0,
                index_col=0,
                keep_default_na=False,
            )
            for r in pdf.itertuples():
                self.proc[r[0]] = r[1]
        except FileNotFoundError:
            pass

        self.numfids = len(self.fidparams)

    def is_multi_segment(self) -> bool:
        """Indicate whether the FID files represent independent segments.

        Returns:
            ``True`` for ``LO_Scan``, ``DR_Scan``, and ``Peak_Up``
            acquisitions, where each ``N.csv`` is an independent
            segment rather than a backup. ``False`` for
            ``Target_Shots``, ``Target_Duration``, and ``Forever``,
            where ``0.csv`` is the cumulative final FID and
            ``1.csv``, ``2.csv``, … are intermediate backups.
        """
        return self.ftmw_type in _MULTI_SEGMENT_FTMW_TYPES

    def num_backups(self) -> int:
        """Return the number of backup FIDs available.

        Returns:
            Number of intermediate backup FIDs available
            (``len(fidparams) - 1`` for single-segment acquisitions;
            ``0`` for multi-segment acquisitions, which do not have
            backups).
        """
        if self.is_multi_segment():
            return 0
        return max(self.numfids - 1, 0)

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

        Raises:
            ValueError: If ``num`` is out of range for this experiment.
        """
        if num < 0 or num >= self.numfids:
            raise ValueError(
                f"Invalid FID number ({num}). Must be between 0 and {self.numfids-1}"
            )
        return BCFid(num, self.path, self.fidparams, self._sep, self.proc)

    def get_differential_fid(self, start: int = 0, end: int = -1) -> BCFid:
        """Build a FID from shots collected between two backup points.

        Mirrors Blackchirp's "differential backup" view, with a more general
        two-bound interface. The returned ``BCFid`` represents the shots
        collected between backup ``start`` and backup ``end``: its raw
        integer data is ``raw[end] - raw[start]`` and its
        ``fidparams.shots`` is ``shots[end] - shots[start]``.

        Args:
            start: Backup index to subtract. ``0`` (default) means do not
                subtract anything — the differential begins at the start
                of the experiment. A positive value loads ``start.csv``
                and subtracts it from the upper-bound FID.
            end: Backup index to use as the upper bound. ``-1`` (default)
                means the cumulative final FID (``0.csv``). A positive
                value loads ``end.csv``.

        Returns:
            A BCFid containing the differential data, with
            ``fidparams.shots`` and ``shots`` set to the shot difference.

        Raises:
            ValueError: If the experiment is multi-segment (``LO_Scan``,
                ``DR_Scan``, ``Peak_Up`` — backups do not exist for these
                modes), if ``start`` or ``end`` is out of range, if the
                resolved ``(start, end)`` pair is not strictly ordered,
                or if the start and end FIDs have a different number of
                frames.
        """
        if self.is_multi_segment():
            raise ValueError(
                f"Differential FIDs are not available for FtmwType={self.ftmw_type!r}"
            )

        nb = self.num_backups()
        if start < 0 or start > nb:
            raise ValueError(f"start={start} out of range [0, {nb}]")
        if end != -1 and (end < 1 or end > nb):
            raise ValueError(f"end={end} out of range; must be -1 or in [1, {nb}]")

        end_idx = 0 if end == -1 else end

        # Determine ordering: end_idx == 0 means "the cumulative final"
        # which is always the upper bound, so it is valid for any
        # start in [0, nb]. For end_idx > 0, require start < end_idx.
        if end_idx != 0 and start >= end_idx:
            raise ValueError(
                f"start ({start}) must be strictly less than end ({end_idx})"
            )

        upper = self.get_fid(end_idx)
        if start == 0:
            return upper

        lower = self.get_fid(start)
        if upper.frames != lower.frames:
            raise ValueError(
                f"Frame count mismatch: end={end_idx} has {upper.frames} frames, "
                f"start={start} has {lower.frames} frames"
            )

        upper_shots = int(upper.fidparams.shots)
        lower_shots = int(lower.fidparams.shots)
        diff_shots = upper_shots - lower_shots
        if diff_shots <= 0:
            raise ValueError(
                f"Non-positive differential shot count: end shots ({upper_shots}) "
                f"≤ start shots ({lower_shots})"
            )

        upper._rawdata = upper._rawdata - lower._rawdata
        upper.fidparams.loc["shots"] = diff_shots
        upper.shots = diff_shots
        upper.data = upper._rawdata * upper.fidparams.vmult / diff_shots
        return upper

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
            \\*\\*proc_kwargs: FID processing settings passed as \\*\\*kwargs to BCFid.ft.

        Returns:
            Frequency and intensity arrays for processed spectrum.

        Raises:
            ValueError: If ``which`` is not one of ``'both'``, ``'upper'``,
                ``'lower'``; if ``avg`` is not ``'harmonic'`` or
                ``'geometric'``; if ``min_ft_offset`` or ``max_ft_offset``
                is non-positive; or if any FID's ``sideband`` value is
                unrecognised (propagated from ``BCFid.is_lower_sideband``).
        """

        min_probe = np.min(self.fidparams.probefreq.to_numpy())
        max_probe = np.max(self.fidparams.probefreq.to_numpy())
        fid = self.get_fid(0)
        x, ft = fid.ft(frame=frame, **proc_kwargs)
        bw = np.abs(np.max(x) - np.min(x))
        ftsp = bw / (len(x) - 1)

        if min_ft_offset is None:
            min_ft_offset = 0.0
        elif min_ft_offset <= 0.0:
            raise ValueError("min_ft_offset must be positive")
        si = int(np.ceil(min_ft_offset / ftsp))

        if max_ft_offset is None:
            max_ft_offset = bw
        elif max_ft_offset <= 0.0:
            raise ValueError("max_ft_offset must be positive")
        ei = int(np.floor(max_ft_offset / ftsp))

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

            if fid.is_lower_sideband():
                xl = x[::-1]
                xu = (fid.fidparams.probefreq - x) + fid.fidparams.probefreq
            else:
                t = (fid.fidparams.probefreq - x) + fid.fidparams.probefreq
                xl = t[::-1]
                xu = x

            if which in ("lower", "both"):
                yint = np.interp(xx, xl, ftl, left=0.0, right=0.0)
                yshots = np.where(yint > 0.0, fid.shots, 0.0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_out = avg_f(y_out, yint, shots_array, yshots)
                shots_array += yshots

            if which in ("upper", "both"):
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
