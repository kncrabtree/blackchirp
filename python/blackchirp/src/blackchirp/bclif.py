from __future__ import annotations

import pandas as pd
import os


class BCLIF:
    def __init__(self, path: str, sep: str):
        self.path = path
        self._sep = sep
        self.lifparams = pd.read_csv(
            os.path.join(self.path, "lif/lifparams.csv"), sep=self._sep, header=0
        )
        self.proc = pd.read_csv(
            os.path.join(self.path, "lif/processing.csv"),
            sep=self._sep,
            header=0,
            index_col=0,
        )
