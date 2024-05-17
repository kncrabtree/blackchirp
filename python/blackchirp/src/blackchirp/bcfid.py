from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.fft as sfft
import scipy.signal as spsig


class BCFid:
    """Container for FID data

    The ``BCFid`` class reads in raw Blackchirp data from disk and converts it
    from base-36 integers to voltages. In addition, it provides convenience functions
    for coaveraging FIDs, subtracting FIDs, and computing Fourier transforms using
    Blackchirp's FID processing settings.

    A single FID may consist of multiple frames. The FID data is represented as a 2D
    numpy array, where the first axis corresponds to the time points and the second axis to the frame number. This is true even if the FID contains only a single frame.

    """

    def __init__(self):
        return

    @classmethod
    def create(
        cls, num: int, path: str, fidparams: pd.DataFrame, sep: str, proc: dict
    ) -> BCFid:
        out = cls()
        out._num = num
        out.fidparams = fidparams.loc[num].copy()
        out.proc = proc
        out.shots = out.fidparams.shots

        d = pd.read_csv(
            os.path.join(path, f"fid/{num}.csv"), sep=sep, header=0, dtype="str"
        )
        ic = np.frompyfunc(int, 2, 1)
        out.frames = len(d.columns)

        out._rawdata = ic(d.to_numpy(dtype="str"), 36).astype(np.int64)
        out.data = out._rawdata * out.fidparams.vmult / out.fidparams.shots
        return out

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

    def x(self) -> np.array:
        return np.arange(self.fidparams["size"]) * self.fidparams.spacing

    def xy(self) -> (np.array, np.array):
        return self.x(), self.data

    def apply_lo(self, freqMHz: np.array) -> np.array:
        if self.fidparams["sideband"] == 1:
            return self.fidparams.probefreq - freqMHz
        else:
            return self.fidparams.probefreq + freqMHz

    def average_frames(self) -> None:
        self._rawdata = np.sum(self._rawdata, axis=1).reshape(
            int(self.fidparams["size"]), 1
        )
        s = self.fidparams.shots
        self.fidparams.loc["shots"] = self.frames * s
        self.frames = 1
        self.data = self._rawdata * self.fidparams.vmult / self.fidparams.shots

    def ft(
        self,
        start_us: float = None,
        end_us: float = None,
        winf: str = None,
        zpf: int = None,
        rdc: bool = None,
        expf_us: float = None,
        autoscale_MHz: float = None,
        frame: int = None,
    ) -> (np.array, np.array):

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
        return len(self.data)
