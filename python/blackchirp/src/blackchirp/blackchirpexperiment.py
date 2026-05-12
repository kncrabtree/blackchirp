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
        FileNotFoundError: If no ``version.csv`` file is found at the
            resolved experiment path.
        KeyError: If ``header.csv`` does not contain the
            ``Experiment / Number`` row required to identify the
            experiment.

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

        Each csv file in the Experiment folder is loaded as a
        `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html>`_.
        These DataFrames are each stored as a variable with the same name as the csv
        file. For example, to see the contents of hardware.csv::

            $ e347.hardware

                                          key                       driver
            0                          AWG.Ka                   VirtualAwg
            1                    Clock.virtual                   FixedClock
            2          FtmwDigitizer.virtual       VirtualFtmwDigitizer
            3       FlowController.Default       PythonFlowController
            4   PulseGenerator.Default       VirtualPulseGenerator

        The ``hardware.csv`` file from older experiments may use the
        legacy column header ``subKey`` instead of ``driver``. Both
        forms are normalised to ``driver`` on read so downstream code
        can use a single column name.


    Attributes:
        num (int): Experiment number
        path (str): Experiment path
        version (pd.DataFrame): Contents of version.csv.
            Information about the Blackchirp version used to acquire the data
        header (pd.DataFrame): Contents of header.csv.
            General experimental parameters.
        objectives (pd.DataFrame): Contents of objectives.csv.
            Information about the goals of the experiment (FTMW type, LIF, etc)
        log (pd.DataFrame): Contents of log.csv.
            All messages printed to Blackchirp's log during the experiment.
        hardware (pd.DataFrame): Contents of hardware.csv.
            All hardware items and their driver identities. The second
            column is exposed as ``driver`` regardless of whether the
            on-disk header was ``driver`` or ``subKey``.
        clocks (pd.DataFrame, optional): Contents of clocks.csv.
            Configurations of all clocks at each experimental step.
            Required for CP-FTMW experiments and ``None`` for LIF-only
            acquisitions, which do not produce a ``clocks.csv``.
        auxdata (pd.DataFrame, optional): Contents of auxdata.csv (if present).
            Data shown on `Aux Data <user_guide/rolling-aux-data.html>`_ plots
            during the experiment.
        chirps (pd.DataFrame, optional): Contents of chirps.csv (if present).
            Details of all chirps associated with a CP-FTMW acquisition.
        markers (pd.DataFrame, optional): Contents of markers.csv (if present).
            Marker channel configuration for the experiment's pulse pattern.
            Columns: ``Channel``, ``Name``, ``Role``, ``TimingMode``,
            ``StartUs``, ``EndUs``, ``Enabled``.
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
        elif num is not None:
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

        with open(os.path.join(self.path, "version.csv"), "r") as v:
            line = v.readline()
            self._sep = line.strip()
        self.version = pd.read_csv(
            os.path.join(self.path, "version.csv"),
            sep=self._sep,
            header=1,
            keep_default_na=False,
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
            keep_default_na=False,
        )
        num_rows = self.header.query("ObjKey == 'Experiment' and ValueKey == 'Number'")
        if num_rows.empty:
            raise KeyError(
                "header.csv is missing the required 'Experiment / Number' row"
            )
        self.num = int(num_rows.Value.iloc[0])

        self.objectives = pd.read_csv(
            os.path.join(self.path, "objectives.csv"),
            sep=self._sep,
            header=0,
            keep_default_na=False,
        )
        self.log = pd.read_csv(
            os.path.join(self.path, "log.csv"),
            sep=self._sep,
            header=0,
            keep_default_na=False,
        )
        self.hardware = pd.read_csv(
            os.path.join(self.path, "hardware.csv"),
            sep=self._sep,
            header=0,
            keep_default_na=False,
        )
        if "subKey" in self.hardware.columns and "driver" not in self.hardware.columns:
            self.hardware = self.hardware.rename(columns={"subKey": "driver"})
        try:
            self.clocks = pd.read_csv(
                os.path.join(self.path, "clocks.csv"),
                sep=self._sep,
                header=0,
                keep_default_na=False,
            )
        except FileNotFoundError:
            self.clocks = None

        try:
            self.auxdata = pd.read_csv(
                os.path.join(self.path, "auxdata.csv"),
                sep=self._sep,
                header=0,
                keep_default_na=False,
            )
        except FileNotFoundError:
            pass

        try:
            self.chirps = pd.read_csv(
                os.path.join(self.path, "chirps.csv"),
                sep=self._sep,
                header=0,
                keep_default_na=False,
            )
        except FileNotFoundError:
            pass

        try:
            self.markers = pd.read_csv(
                os.path.join(self.path, "markers.csv"),
                sep=self._sep,
                header=0,
                keep_default_na=False,
            )
        except FileNotFoundError:
            pass

        ftmw_type_rows = self.objectives.query("key == 'FtmwType'")
        ftmw_type = (
            str(ftmw_type_rows.value.iloc[0]) if not ftmw_type_rows.empty else ""
        )

        if os.path.exists(os.path.join(self.path, "fid")):
            self.ftmw = BCFTMW(self.path, self._sep, ftmw_type)

        if os.path.exists(os.path.join(self.path, "lif")):
            self.lif = BCLIF(self.path, self._sep, self.header)

        if hasattr(self, "ftmw") and self.clocks is None:
            raise FileNotFoundError(
                f"clocks.csv is required for CP-FTMW experiments but was not "
                f"found at {os.path.join(self.path, 'clocks.csv')!r}"
            )

    def header_unique_keys(self) -> set[str]:
        """Fetch all unique ObjKeys in experiment header

        Returns:
            Set of unique header keys

        """

        return set(self.header.ObjKey.tolist())

    def header_rows(
        self, objKey: str = None, valKey: str = None, arrKey: str = None
    ) -> pd.DataFrame:
        """Fetch rows from the header file matching conditions

        Filters rows in the header according to ObjKey, ValueKey, and ArrayKey.
        Any combination of these (or none) may be specified to filter.

        Args:
            objKey: Object key in header
            valKey: Value key in header
            arrKey: Array key in header

        Returns:
            DataFrame with matching rows. May be empty (use ``.empty`` /
            ``len()`` to test) — an empty result is not an error here.

        """

        df = self.header
        if objKey is not None:
            df = df.query(f"ObjKey == '{objKey}'")
        if valKey is not None:
            df = df.query(f"ValueKey == '{valKey}'")
        if arrKey is not None:
            df = df.query(f"ArrayKey == '{arrKey}'")

        return df

    def header_value(
        self, objKey: str, valKey: str, idx: int = 0, arrKey: str = None
    ) -> str:
        """Fetch one value from header

        The ``objKey`` and ``valKey`` (and ``arrKey``, if specified) are used
        to filter the header. The ``idx`` value then selects which matching
        row to return.

        Args:
            objKey: Object key in header
            valKey: Value key in header
            idx: Row number to return (optional)
            arrKey: Array key in header (optional)

        Returns:
            Matching value as a string.

        Raises:
            KeyError: If no row matches the supplied filter, or if ``idx``
                is past the end of the matching rows.
        """

        df = self.header_rows(objKey, valKey, arrKey)
        if len(df) <= idx:
            raise KeyError(
                f"No header row at index {idx} for ObjKey={objKey!r}, "
                f"ValueKey={valKey!r}, ArrayKey={arrKey!r}"
            )

        return df.Value.iloc[idx]

    def header_unit(
        self, objKey: str, valKey: str, idx: int = 0, arrKey: str = None
    ) -> str:
        """Fetch one unit value from header

        The ``objKey`` and ``valKey`` (and ``arrKey``, if specified) are used
        to filter the header. The ``idx`` value then selects which matching
        row to return.

        Args:
            objKey: Object key in header
            valKey: Value key in header
            idx: Row number to return (optional)
            arrKey: Array key in header (optional)

        Returns:
            Unit string for the matching row. An empty string is returned
            when the row exists but its ``Units`` cell is empty (a value
            with no associated unit is a legitimate state).

        Raises:
            KeyError: If no row matches the supplied filter, or if ``idx``
                is past the end of the matching rows.
        """

        df = self.header_rows(objKey, valKey, arrKey)
        if len(df) <= idx:
            raise KeyError(
                f"No header row at index {idx} for ObjKey={objKey!r}, "
                f"ValueKey={valKey!r}, ArrayKey={arrKey!r}"
            )

        out = df.Units.iloc[idx]
        if out != out:
            return ""

        return out
