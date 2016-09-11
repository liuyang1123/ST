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
        col_names = ["user", "item", "rate", "timestamp"]
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
            z.extract('ml-100k/ua.base', self.workdir)

            data = open(self.workdir + 'ml-100k/ua.base')
            df = self._read_process(data, sep="\t")
            data.close()

            rows = len(df)
            split_index = int(rows * 0.9)

            df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)

            df_train = df[0:split_index]
            df_test = df[split_index:].reset_index(drop=True)

            self.train = [[df_train["user"].values, df_train["item"].values],
                          df_train["rate"].values]

            self.test = [[df_test["user"].values, df_test["item"].values],
                         df_test["rate"].values]

            self._num_examples_train = self.train[0][0].shape[0]
            self._num_examples_test = self.test[0][0].shape[0]
            self._num_examples_validation = 0
