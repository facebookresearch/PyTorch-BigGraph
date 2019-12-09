#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import gzip
import shutil
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Iterable
from urllib.parse import urlparse
from urllib.request import urlretrieve

from tqdm import tqdm


def extract_gzip(gzip_path: Path, remove_finished: bool = False) -> str:
    print(f"Extracting {gzip_path}")
    if gzip_path.suffix != ".gz":
        raise RuntimeError("Not a gzipped file")
    fpath = gzip_path.with_suffix("")

    if fpath.exists():
        print("Found a file that indicates that the input data "
              "has already been extracted, not doing it again.")
        print(f"This file is: {fpath}")
        return fpath

    with fpath.open("wb") as out_bf, gzip.GzipFile(gzip_path) as zip_f:
        shutil.copyfileobj(zip_f, out_bf)
    if remove_finished:
        gzip_path.unlink()

    return fpath


def extract_tar(fpath: Path) -> None:
    # extract file
    with tarfile.open(fpath, "r:gz") as tar:
        tar.extractall(path=fpath.parent)


def gen_bar_updater(pbar: tqdm) -> Callable[[int, int, int], None]:
    def bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, root: Path, filename: Optional[str] = None) -> str:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under.
                        If None, use the basename of the URL
    """

    root = root.expanduser()
    if filename is None:
        filename = Path(urlparse(url).path).name
    fpath = root / filename
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    # downloads file
    if fpath.is_file():
        print(f"Using downloaded and verified file: {fpath}")
    else:
        try:
            print(f"Downloading {url} to {fpath}")
            urlretrieve(
                url, str(fpath),
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            print(f"Failed to download from url: {url}")

    return fpath


class EdgelistReader(ABC):
    @abstractmethod
    def read(self, path: Path) -> Iterable[Tuple[str, str, Optional[str]]]:
        """Read rows from a path. Returns (lhs, rhs, rel)."""
        pass


class TSVEdgelistReader(EdgelistReader):
    def __init__(self,
                 lhs_col: int,
                 rhs_col: int,
                 rel_col: int,
    ):
        self.lhs_col, self.rhs_col, self.rel_col = lhs_col, rhs_col, rel_col

    def read(self, path: Path):
        with path.open("rt") as tf:
            for line_num, line in enumerate(tf, start=1):
                try:
                    words = line.split()
                    lhs_word = words[self.lhs_col]
                    rhs_word = words[self.rhs_col]
                    rel_word = words[self.rel_col] if self.rel_col is not None else None
                    yield lhs_word, rhs_word, rel_word
                except IndexError:
                    raise RuntimeError(
                        f"Line {line_num} of {path} has only {len(words)} words"
                    ) from None


ParquetCol = Union[int, str]
class ParquetEdgelistReader(EdgelistReader):
    def __init__(self,
                 lhs_col: ParquetCol,
                 rhs_col: ParquetCol,
                 rel_col: Optional[ParquetCol],
    ):
        """Reads edgelists from a Parquet file.

        col arguments can either be the column name or the offset of the col.
        """
        self.lhs_col, self.rhs_col, self.rel_col = lhs_col, rhs_col, rel_col

    def _get_col(self, tf, col):
        if isinstance(col, str):
            return col
        elif isinstance(col, int):
            return tf.columns[col]
        else:
            raise RuntimeError(f"Unknown column type: {col}")

    def read(self, path: Path):
        import parquet
        with path.open() as tf:
            columns = [self._get_col(col) for col in (self.lhs_col, self.rhs_col)]
            if self.rel_col is not None:
                columns.append(self._get_col(self.rel_col))
            for row in parquet.reader(tf, columns=columns):
                if self.rel_col is not None:
                    yield row
                else:
                    yield (row[0], row[1], None)
