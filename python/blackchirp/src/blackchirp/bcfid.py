from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.fft as sfft
import scipy.signal as spsig


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
            os.path.join(path, f"fid/{num}.csv"), sep=sep, header=0, dtype="str"
        )
        ic = np.frompyfunc(int, 2, 1)
        self.frames = len(d.columns)

        self._rawdata = ic(d.to_numpy(dtype="str"), 36).astype(np.int64)
        self.data = self._rawdata * self.fidparams.vmult / self.fidparams.shots

    # @classmethod
    # def create_coaverage(
    #     cls,
    #     explist: list,
    #     pathlist: list,
    #     sep: str = ";",
    #     pc_start: int = None,
    #     pc_end: int = None,
    # ) -> BCFid:
    #     fidlist = []
    #     for e, p in zip(explist, pathlist):
    #         exp = BCExperiment(e, p)
    #         fidlist.append(exp.ftmw.get_fid(0))
    #         fp = exp.ftmw.fidparams
    #
    #     shots = np.sum(np.asarray([x.fidparams.shots for x in fidlist]))
    #     if pc_start is not None and pc_end is not None:
    #         rd = fidlist[0]._rawdata
    #         for i in range(len(fidlist) - 1):
    #             reff = fidlist[0]._rawdata[pc_start:pc_end].flatten() / shots
    #             thisfid = fidlist[i + 1]._rawdata[pc_start:pc_end].flatten() / shots
    #             shift = np.argmax(np.correlate(reff, thisfid, mode="full")) - (
    #                 len(reff) - 1
    #             )
    #             if shift < 0:
    #                 rd[:shift] += fidlist[i + 1]._rawdata[-shift:]
    #             elif shift > 0:
    #                 rd[:-shift] += fidlist[i + 1]._rawdata[shift:]
    #             else:
    #                 rd += fidlist[i + 1]._rawdata
    #     else:
    #         rd = np.sum(np.asarray([x._rawdata for x in fidlist]), axis=0)
    #
    #     fp.loc[0, "shots"] = shots
    #     out = cls()
    #     out.fidparams = fp.loc[0]
    #     out._rawdata = rd
    #     out.data = out._rawdata * fp.loc[0].vmult / shots
    #     return out

    def x(self) -> np.ndarray:
        """Compute time array for FID (units: s)

        Returns:
            Numpy array containing time points
        """
        return np.arange(self.fidparams["size"]) * self.fidparams.spacing

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Get time and voltage arrays for FID

        The FID data is a 2D numpy array whose second axis corresponds to the frame
        index. The time array is 1D.

        Returns:
            Time array, FID array

        """
        return self.x(), self.data

    def apply_lo(self, freqMHz: np.ndarray) -> np.ndarray:
        """Compute molecular frequency from scope frequency.

        If the downconversion mixer is the lower sideband, then the molecular
        frequency is the Downconversion LO frequency - scope frequency. Otherwise,
        it is Downconverion LO frequency + scope frequency.

        Args:
            freqMHz: Scope frequency array, in MHz

        Returns:
            1D numpy array of molecular frequencies, in MHz

        """
        if self.fidparams["sideband"] == 1:
            return self.fidparams.probefreq - freqMHz
        else:
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Fourier transform of the FID

        By default, this computes the FT for each frame in the FID using the settings
        stored in the proc dictionary. This behavior can be overridden by specifying
        any combination of the keyword arguments.

        Args:
            start_us: Starting time, in μs. Points at earlier times are set to 0.
            end_us: Ending time, in μs. Points at later times are set to 0.
            winf: Window function applied to points between start and end.
                This is passed directly to `scipy.signal.get_window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html>`_.
            zpf: Zero-padding factor (positive integer). If nonzero, the FID is
                padded with zeroes until its length reaches the next power of 2,
                Then, its length is further extended by 2\*\*zpf.
            rdc: If true, the average of the FID is subtracted before the FT is
                computed.
            expf_us: Time constant for an exponential decay filter, in μs.
            autoscale_MHz: Range of FT points to set to 0, relative to the Downconversion
                LO frequency. Useful for suppressing noise near DC.
            units_power: FT is scaled by 10\*\*units_power. For μV units, set
                units_power=6.
            frame: Apply FT to only the specified frame.

        Returns:
            Frequency array (MHz), Intensity array

        Raises:
            ValueError: If supplied arguments are invalid

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

        if winf is None:
            try:
                wf = int(self.proc["FidWindowFunction"])
            except KeyError:
                wf = 0

            if wf == 0:
                winf = "boxcar"
            if wf == 1:
                winf = "bartlett"
            elif wf == 2:
                winf = "blackman"
            elif wf == 3:
                winf = "blackmanharris"
            elif wf == 4:
                winf = "hamming"
            elif wf == 5:
                winf = "hann"
            elif wf == 6:
                winf = ("kaiser", 14.0)

        fid_data = self.data[start:end, :] * (
            np.repeat(spsig.get_window(winf, end - start), self.data.shape[1]).reshape(
                end - start, self.data.shape[1]
            )
        )
        if frame is not None:
            fid_data = fid_data[:, frame].reshape(-1, 1)

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
            for j in range(fid_data.shape[1]):
                fid_data[:, j] = fid_data[:, j] * np.exp(
                    -np.arange(len(fid_data[:, j]))
                    * self.fidparams.spacing
                    / expf_us
                    * 1e6
                )

        if zpf is None:
            try:
                zpf = int(self.proc["FidZeroPadFactor"])
                zpf = min(zpf, 4)
                zpf = max(zpf, 0)
            except KeyError:
                zpf = 0

        if zpf > 0:
            s = 1

            # the operation << 1 is a fast implementation of *2
            # starting at 1, double until s is the next bigger power of 2
            # compared to the length of the FID
            while s <= size:
                s = s << 1

            # double 1 more time
            s = s << 1

            # if zpf=1, we are done. Otherwise, keep multiplying zpf-1 times
            for i in range(0, zpf - 1):
                s = s << 1
        else:
            s = size

        ft = sfft.rfft(fid_data, n=s, axis=0)
        out_x = self.apply_lo(sfft.rfftfreq(s, self.fidparams.spacing) * 1e-6)
        out_y = np.absolute(ft)
        out_y /= fid_data.shape[0]

        if units_power is None:
            try:
                p = int(self.proc["FtUnits"])
            except KeyError:
                p = 0
        else:
            p = units_power

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

        return out_x, out_y

    def __len__(self):
        return self.data.shape[0]
