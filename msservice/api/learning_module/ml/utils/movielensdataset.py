# movielensdatataset.py
# Contact: X ( x@gmail.com )

from dataset import Dataset
from zipfile import ZipFile
import numpy as np
import pandas as pd

class MovieLensDataset(Dataset):
    """
    """

    def _filter_by_range(self, obj, start, end):
        return np.array(obj)[:,start:end]

    def _shuffle(self, obj, perm):
        return np.array(obj)[:,perm]
        
    def _read_process(self, filname, sep="\t"):
        """
        """
        col_names = ["user", "item", "rate", "st"]
        df = pd.read_csv(filname, sep=sep, header=None, names=col_names)
        df["user"] -= 1
        df["item"] -= 1
        for col in ("user", "item"):
            df[col] = df[col].astype(np.int32)
        df["rate"] = df["rate"].astype(np.float32)
        return df

    def _process(self):
        """
        """
        folder = ""
        with ZipFile(self.workdir + self.filenames["100k"], "r") as z:
            data = z.read(z.namelist()[0] + 'ua.base')
            df = self._read_process(data, sep="\t")
            rows = len(df)
            df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
            # split_index = int(rows * 0.9)
            # df_train = df[0:split_index]
            # df_validation = df[split_index:].reset_index(drop=True)

            self.train = [[df["user"], df["item"]], df["rate"]]
            self.test = [[df_test["user"], df_test["item"]], df_test["rate"]]
