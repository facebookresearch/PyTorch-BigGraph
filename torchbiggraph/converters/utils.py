#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import os
import tarfile
import shutil
import urllib.request
from typing import Callable, Optional

from tqdm import tqdm


def extract_gzip(gzip_path: str, remove_finished: bool = False) -> str:
    print('Extracting %s' % gzip_path)
    fpath, ext = os.path.splitext(gzip_path)
    if ext != ".gz":
        raise RuntimeError("Not a gzipped file")

    with open(fpath, "wb") as out_bf, gzip.GzipFile(gzip_path) as zip_f:
        shutil.copyfileobj(zip_f, out_bf)
    if remove_finished:
        os.unlink(gzip_path)

    return fpath


def extract_tar(fpath: str) -> None:
    # extract file
    root = os.path.dirname(fpath)
    with tarfile.open(fpath, "r:gz") as tar:
        tar.extractall(path=root)


def gen_bar_updater(pbar: tqdm) -> Callable[[int, int, int], None]:
    def bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, root: str, filename: Optional[str] = None) -> str:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under.
                        If None, use the basename of the URL
    """

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    if not os.path.exists(root):
        os.makedirs(root)

    # downloads file
    if os.path.isfile(fpath):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            print('Failed to download from url: ' + url)

    return fpath
