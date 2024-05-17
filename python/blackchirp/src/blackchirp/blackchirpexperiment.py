# -*- coding: utf-8 -*-

from __future__ import annotations


import os
import pandas as pd
from .bcftmw import BCFTMW
from .bclif import BCLIF


class BCExperiment:
    """Container for Blackchirp experimental data

    The ``BCExperiment`` class reads in experimental data files from a Blackchirp
    Experiment, providing access to all of the data and settings associated with
    the experiment. The CP-FTMW and/or LIF data associated with the experiment are
    loaded into ``BCFTMW`` and ``BCLIF`` objects which are used to access and process
    the experimental data.

    Essentially, the ``BCExperiment`` class directly stores the data shown on the
    Header, Aux Data, and Log tabs, and its ``ftmw`` and ``lif`` properties contain
    the data shown on the CP-FTMW and LIF tabs in the Blackchirp program.

    Args:
        path: Path to data storage folder or experiment folder
        num: Experiment number (only required if path is a data storage folder)


    Raises:
        FileNotFoundError: If no ``version.csv`` file is found

    Example:
        To load data, pass the experiment number and/or the path to the
        folder containing the data. The path should point either to the base
        Blackchirp data storage folder or to the specific folder of the desired
        experiment.

        For example, if the data storage location is ``/home/user/blackchirp``, then
        to load experiment 347 (``/home/user/blackchirp/experiments/0/0/347``)::

            $ e347 = BCExperiment('/home/user/blackchirp',347)

        The experiment number is only required if pointing to a data storage folder.
        The same data could be accessed with::

            $ e347 = BCExperiment('/home/user/blackchirp/experiments/0/0/347')

        If, instead, folder 347 has been copied into the same directory as your python
        script, the experiment can be accessed with::

            $ e347 = BCExperiment('./347')

        Each csv file in the Experiment folder is loaded as a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html>`_. These
        DataFrames are each stored as a variable with the same name as the csv file.
        For example, to see the contents of hardware.csv::

            $ e437.hardware

                            key   subKey
            0             AWG.0  virtual
            1           Clock.0    fixed
            2           Clock.1    fixed
            3   FtmwDigitizer.0  virtual
            4  PulseGenerator.0  virtual


    Attributes:
        num (int): Experiment number
        path (str): Experiment path
        version (DataFrame): Contents of version.csv.
            Information about the Blackchirp version used to acquire the data
        header (DataFrame): Contents of header.csv.
            General experimental parameters.
        objectives (DataFrame): Contents of objectives.csv.
            Information about the goals of the experiment (FTMW type, LIF, etc)
        log (DataFrame): Contents of log.csv.
            All messages printed to Blackchirp's log during the experiment.
        hardware (DataFrame): Contents of hardware.csv.
            All hardware items and their implementation keys.
        clocks (DataFrame): Contents of clocks.csv
            Configurations of all clocks at each experimental step.
        auxdata (DataFrame, optional): Contents of auxdata.csv (if present).
            Data shown on `Aux Data <user_guide/rolling-aux-data.html>`_ plots
            during the experiment.
        chirps (DataFrame, optional): Contents of chirps.csv (if present).
            Details of all chirps associated with a CP-FTMW acquisition.
        ftmw (BCFTMW, optional): Contents of fid directory.
            This object provides an interface for accessing CP-FTMW data.
        lif (BCLIF, optional): Contents of lif directory.
            This object provides an interface for accessing LIF data.


    """

    def __init__(self, path: str = ".", num: int = None):
        if num is not None:
            self._mil = num // 1000000
            self._th = num // 1000
            self.num = num
        self.path = path
        self._sep = ";"

        if os.path.exists(os.path.join(path, "version.csv")):
            self.path = path
        elif num is not None:  # maybe this is a savePath
            testpath = os.path.join(
                path, f"experiments/{self._mil}/{self._th}/{self.num}"
            )
            if os.path.exists(os.path.join(testpath, "version.csv")):
                self.path = testpath
            else:
                raise FileNotFoundError(f"Could not find Blackchirp data at {testpath}")
        else:
            raise FileNotFoundError(
                f"Could not find Blackchirp data at {os.path.abspath(path)}"
            )

        # read separator character from version file
        with open(os.path.join(self.path, "version.csv"), "r") as v:
            l = v.readline()
            self._sep = l.strip()
        self.version = pd.read_csv(
            os.path.join(self.path, "version.csv"), sep=self._sep, header=1
        )

        self.header = pd.read_csv(
            os.path.join(self.path, "header.csv"),
            sep=self._sep,
            header=0,
            dtype={
                "ObjKey": str,
                "ArrayKey": str,
                "ArrayIndex": "Int64",
                "ValueKey": str,
                "Value": str,
                "Units": str,
            },
        )
        self.num = int(
            self.header.query(
                "ObjKey == 'Experiment' and ValueKey == 'Number'"
            ).Value.iloc[0]
        )

        self.objectives = pd.read_csv(
            os.path.join(self.path, "objectives.csv"), sep=self._sep, header=0
        )
        self.log = pd.read_csv(
            os.path.join(self.path, "log.csv"), sep=self._sep, header=0
        )
        self.hardware = pd.read_csv(
            os.path.join(self.path, "hardware.csv"), sep=self._sep, header=0
        )
        self.clocks = pd.read_csv(
            os.path.join(self.path, "clocks.csv"), sep=self._sep, header=0
        )

        try:
            self.auxdata = pd.read_csv(
                os.path.join(self.path, "auxdata.csv"), sep=self._sep, header=0
            )
        except FileNotFoundError:
            pass

        try:
            self.chirps = pd.read_csv(
                os.path.join(self.path, "chirps.csv"), sep=self._sep, header=0
            )
        except FileNotFoundError:
            pass

        if os.path.exists(os.path.join(self.path, "fid")):
            self.ftmw = BCFTMW(self.path, self._sep)

        if os.path.exists(os.path.join(self.path, "lif")):
            self.lif = BCLIF(self.path, self._sep)
