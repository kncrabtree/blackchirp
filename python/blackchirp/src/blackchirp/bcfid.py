from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.fft as sfft
import scipy.signal as spsig

from ._enum_helpers import _resolve_enum

_WINDOW_MAP = {
    "None": "boxcar",
    "Bartlett": "bartlett",
    "Blackman": "blackman",
    "BlackmanHarris": "blackmanharris",
    "Hamming": "hamming",
    "Hanning": "hann",
    "KaiserBessel": ("kaiser", 14.0),
}

_WINDOW_INT_MAP = {
    0: "None",
    1: "Bartlett",
    2: "Blackman",
    3: "BlackmanHarris",
    4: "Hamming",
    5: "Hanning",
    6: "KaiserBessel",
}

_FT_UNITS_MAP = {
    "FtV": 0,
    "FtmV": 3,
    "FtuV": 6,
    "FtnV": 9,
}

_FT_UNITS_INT_MAP = {0: "FtV", 3: "FtmV", 6: "FtuV", 9: "FtnV"}

_SIDEBAND_MAP = {"UpperSideband": 0, "LowerSideband": 1}
_SIDEBAND_INT_MAP = {0: "UpperSideband", 1: "LowerSideband"}

_TIME_UNIT_SCALES = {
    "s": 1.0,
    "ms": 1.0e3,
    "us": 1.0e6,
    "μs": 1.0e6,
    "ns": 1.0e9,
}

_FREQ_UNIT_SCALES_FROM_MHZ = {
    "Hz": 1.0e6,
    "kHz": 1.0e3,
    "MHz": 1.0,
    "GHz": 1.0e-3,
    "THz": 1.0e-6,
}


def _resolve_time_scale(units: str) -> float:
    try:
        return _TIME_UNIT_SCALES[units]
    except KeyError as err:
        raise ValueError(
            f"Unknown time unit {units!r}; choose from " f"{sorted(_TIME_UNIT_SCALES)}"
        ) from err


def _resolve_freq_scale_from_mhz(units: str) -> float:
    try:
        return _FREQ_UNIT_SCALES_FROM_MHZ[units]
    except KeyError as err:
        raise ValueError(
            f"Unknown frequency unit {units!r}; choose from "
            f"{sorted(_FREQ_UNIT_SCALES_FROM_MHZ)}"
        ) from err


class BCFid:
    """Container for FID data

    The ``BCFid`` class reads in raw Blackchirp data from disk and converts it
    from base-36 integers to voltage using the conversion values located in
    fidparams.csv. In addition, it provides convenience functions for coaveraging
    FIDs, subtracting FIDs, and computing Fourier transforms using
    Blackchirp's FID processing settings.

    A single FID may consist of multiple frames. The FID data is represented as a 2D
    numpy array, where the first axis corresponds to the time points and the second axis
    to the frame number. This is true even if the FID contains only a single frame.

    A BCFid object should not be created by the end user; it is designed to work with
    the BCFTMW class.

    Args:
        num: Number of the FID csv file
        path: Base path of experiment
        fidparams: DataFrame loaded from fidparams.csv (indexed by FID number)
        sep: CSV delimiter
        proc: Default processing settings from processing.csv

    Attributes:
        fidparams: pandas Series containing corresponding row from fidparams.csv
        proc: Dictionary of processing settings from processing.csv
        shots: Number of shots
        frames: Number of frames
        data: Array of shape (len(fid),frames) containing FID voltage data

    """

    def __init__(
        self, num: int, path: str, fidparams: pd.DataFrame, sep: str, proc: dict
    ):
        self._num = num
        self.fidparams = fidparams.loc[num].copy()
        self.proc = proc
        self.shots = self.fidparams.shots

        d = pd.read_csv(
            os.path.join(path, f"fid/{num}.csv"),
            sep=sep,
            header=0,
            dtype="str",
            keep_default_na=False,
        )
        ic = np.frompyfunc(int, 2, 1)
        self.frames = len(d.columns)

        self._rawdata = ic(d.to_numpy(dtype="str"), 36).astype(np.int64)
        self.data = self._rawdata * self.fidparams.vmult / self.fidparams.shots

    def x(self, units: str = "s") -> np.ndarray:
        """Compute time array for FID.

        Args:
            units: One of ``"s"``, ``"ms"``, ``"us"`` (or ``"μs"``),
                ``"ns"``. ``"s"`` is the default; other choices
                rescale the array for plotting convenience and have
                no effect on the stored sample spacing.

        Returns:
            1D numpy array of sample times in the requested units.

        Raises:
            ValueError: If ``units`` is not a recognised time-unit
                string.
        """
        scale = _resolve_time_scale(units)
        return np.arange(self.fidparams["size"]) * (self.fidparams.spacing * scale)

    def xy(self, units: str = "s") -> tuple[np.ndarray, np.ndarray]:
        """Get time and voltage arrays for FID.

        The FID data is a 2D numpy array whose second axis corresponds to the frame
        index. The time array is 1D.

        Args:
            units: Time-axis units (see :meth:`x`).

        Returns:
            Time array, FID array.
        """
        return self.x(units), self.data

    def is_lower_sideband(self) -> bool:
        """Indicate whether downconversion uses the lower sideband.

        Returns:
            ``True`` if the ``sideband`` field in ``fidparams`` denotes the
            lower sideband (canonical name ``LowerSideband`` or integer ``1``).

        Raises:
            ValueError: If the ``sideband`` value is neither a recognised
                name nor a recognised integer.
        """
        name = _resolve_enum(
            self.fidparams["sideband"],
            _SIDEBAND_MAP,
            int_map=_SIDEBAND_INT_MAP,
        )
        return name == "LowerSideband"

    def apply_lo(self, freqMHz: np.ndarray) -> np.ndarray:
        """Compute molecular frequency from scope frequency.

        If the downconversion mixer is the lower sideband, then the molecular
        frequency is the Downconversion LO frequency - scope frequency. Otherwise,
        it is Downconverion LO frequency + scope frequency.

        Args:
            freqMHz: Scope frequency array, in MHz

        Returns:
            1D numpy array of molecular frequencies, in MHz

        Raises:
            ValueError: If ``fidparams['sideband']`` is unrecognised.

        """
        if self.is_lower_sideband():
            return self.fidparams.probefreq - freqMHz
        return self.fidparams.probefreq + freqMHz

    def average_frames(self) -> None:
        """Coaverages all frames in the time domain

        For FIDs with multiple frames, this function performs a coaverage, reducing the
        FID y data second axis to length 1. If there is only 1 frame, the function has no effect.

        """
        self._rawdata = np.sum(self._rawdata, axis=1).reshape(
            int(self.fidparams["size"]), 1
        )
        s = self.fidparams.shots
        self.fidparams.loc["shots"] = self.frames * s
        self.frames = 1
        self.data = self._rawdata * self.fidparams.vmult / self.fidparams.shots

    def ft(
        self,
        *,
        start_us: float = None,
        end_us: float = None,
        winf: str = None,
        zpf: int = None,
        rdc: bool = None,
        expf_us: float = None,
        autoscale_MHz: float = None,
        units_power: int = None,
        frame: int = None,
        freq_units: str = "MHz",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Fourier transform of the FID

        By default, this computes the FT for each frame in the FID using the settings
        stored in the proc dictionary. This behavior can be overridden by specifying
        any combination of the keyword arguments.

        Args:
            start_us: Starting time, in μs. Points at earlier times are set to 0.
            end_us: Ending time, in μs. Points at later times are set to 0.
            winf: Window function name. One of ``None``, ``Bartlett``,
                ``Blackman``, ``BlackmanHarris``, ``Hamming``, ``Hanning``,
                ``KaiserBessel``. May also be passed in any form accepted by
                `scipy.signal.get_window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_
                — tuples and arbitrary names are forwarded directly.
            zpf: Zero-padding factor (positive integer). If nonzero, the FID is
                padded with zeroes until its length reaches the next power of 2,
                Then, its length is further extended by 2\\*\\*zpf.
            rdc: If true, the average of the FID is subtracted before the FT is
                computed.
            expf_us: Time constant for an exponential decay filter, in μs.
            autoscale_MHz: Range of FT points to set to 0, relative to the Downconversion
                LO frequency. Useful for suppressing noise near DC.
            units_power: FT is scaled by 10\\*\\*units_power. For μV units, set
                units_power=6. May also be specified in ``processing.csv`` as
                an enum name (``FtV`` / ``FtmV`` / ``FtuV`` / ``FtnV``) or
                as the corresponding integer.
            frame: Apply FT to only the specified frame.
            freq_units: Units for the returned frequency array. One of
                ``"Hz"``, ``"kHz"``, ``"MHz"`` (default), ``"GHz"``,
                ``"THz"``. ``start_us``, ``end_us``, ``expf_us``, and
                ``autoscale_MHz`` are interpreted in their declared
                units regardless of this choice.

        Returns:
            Frequency array (in ``freq_units``), Intensity array.

        Raises:
            ValueError: If ``winf`` is a string that does not match a known
                window-function name, if the ``FidWindowFunction`` value
                stored in ``processing.csv`` is unrecognised, if the
                ``FtUnits`` value is unrecognised, or if ``freq_units``
                is not a recognised frequency-unit string.

        Examples:
            Assuming a BCFid object named ``fid``::

                #default FT calculation
                x,y = fid.ft()

                #override some processing settings
                x,y = fid.ft(start_us=3.0,rdc=False,units_power=3)

                #compute ft for only frame 3 (assuming the number of frames is >=4)
                x,y = fid.ft(frame=3)

                #average all frames, then apply a custom window function
                fid.average_frames()
                p = 1.5
                sigma = len(fid)//5
                x,y = fid.ft(winf=('general_gaussian',p,sigma))

        """

        size = int(self.fidparams["size"])
        if start_us is None:
            try:
                start = max(
                    round(
                        float(self.proc["FidStartUs"]) / 1e6 / self.fidparams.spacing
                    ),
                    0,
                )
            except KeyError:
                start = 0
        else:
            start = max(round(start_us / 1e6 / self.fidparams.spacing), 0)

        if end_us is None:
            try:
                end = round(float(self.proc["FidEndUs"]) / 1e6 / self.fidparams.spacing)
            except KeyError:
                end = size
        else:
            end = round(end_us / 1e6 / self.fidparams.spacing)

        if end < start:
            end = size

        # Copy the slice so subsequent in-place ops (DC subtraction,
        # exp-decay multiplication) do not mutate self.data through
        # the view.
        fid_data = self.data[start:end, :].copy()

        if rdc is None:
            try:
                if self.proc["FidRemoveDC"] == "true":
                    fid_data -= np.mean(fid_data, axis=0)
            except KeyError:
                pass
        elif rdc:
            fid_data -= np.mean(fid_data, axis=0)

        if expf_us is None:
            try:
                expf_us = float(self.proc["FidExpfUs"])
            except KeyError:
                expf_us = 0.0

        if expf_us > 0.0:
            decay = np.exp(
                -np.arange(end - start) * self.fidparams.spacing / expf_us * 1e6
            )
            fid_data = fid_data * decay[:, None]

        winf_arg = self._resolve_window(winf)

        fid_data = fid_data * (
            np.repeat(
                spsig.get_window(winf_arg, end - start), self.data.shape[1]
            ).reshape(end - start, self.data.shape[1])
        )
        if frame is not None:
            fid_data = fid_data[:, frame].reshape(-1, 1)

        if zpf is None:
            try:
                zpf = int(self.proc["FidZeroPadFactor"])
                zpf = min(zpf, 4)
                zpf = max(zpf, 0)
            except KeyError:
                zpf = 0

        if zpf > 0:
            s = 1

            while s <= size:
                s = s << 1

            s = s << 1

            for _ in range(0, zpf - 1):
                s = s << 1
        else:
            s = size

        ft = sfft.rfft(fid_data, n=s, axis=0)
        out_x = self.apply_lo(sfft.rfftfreq(s, self.fidparams.spacing) * 1e-6)
        out_y = np.absolute(ft)
        out_y /= fid_data.shape[0]

        p = self._resolve_units_power(units_power)
        out_y *= 10**p

        if autoscale_MHz is None:
            try:
                autoscale_MHz = float(self.proc["AutoscaleIgnoreMHz"])
            except KeyError:
                autoscale_MHz = 0.0

        if autoscale_MHz > 0.0:
            for i in range(out_y.shape[1]):
                out_y[
                    np.abs(out_x - self.apply_lo(autoscale_MHz)) <= autoscale_MHz, i
                ] = 0

        out_x = out_x * _resolve_freq_scale_from_mhz(freq_units)
        return out_x, out_y

    def _resolve_window(self, winf):
        """Resolve a window-function argument to a scipy ``get_window`` spec.

        Lookup chain: explicit ``winf`` kwarg → ``processing.csv``
        ``FidWindowFunction`` → default ``"None"`` (boxcar). String inputs
        and integers go through the canonical enum name; any other type
        (tuple, etc.) is forwarded to scipy unchanged so callers can use
        scipy's full vocabulary.
        """
        if winf is None:
            wf = self.proc.get("FidWindowFunction", "None")
            name = _resolve_enum(
                wf, _WINDOW_MAP, int_map=_WINDOW_INT_MAP, default="None"
            )
            return _WINDOW_MAP[name]
        if isinstance(winf, str):
            name = _resolve_enum(winf, _WINDOW_MAP, int_map=_WINDOW_INT_MAP)
            return _WINDOW_MAP[name]
        return winf

    def _resolve_units_power(self, units_power):
        """Resolve the FT-units exponent.

        Lookup chain: explicit ``units_power`` kwarg → ``processing.csv``
        ``FtUnits`` (string name or integer) → default 0.
        """
        if units_power is not None:
            return int(units_power)
        if "FtUnits" not in self.proc:
            return 0
        name = _resolve_enum(
            self.proc["FtUnits"],
            _FT_UNITS_MAP,
            int_map=_FT_UNITS_INT_MAP,
            default="FtV",
        )
        return _FT_UNITS_MAP[name]

    def __len__(self):
        return self.data.shape[0]
