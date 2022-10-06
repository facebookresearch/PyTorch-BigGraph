#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import gzip
import shutil
import tarfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

from tqdm import tqdm


def extract_gzip(gzip_path: Path, remove_finished: bool = False) -> Path:
    print(f"Extracting {gzip_path}")
    if gzip_path.suffix != ".gz":
        raise RuntimeError("Not a gzipped file")
    fpath = gzip_path.with_suffix("")

    if fpath.exists():
        print(
            "Found a file that indicates that the input data "
            "has already been extracted, not doing it again."
        )
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
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=fpath.parent)


def gen_bar_updater(pbar: tqdm) -> Callable[[int, int, int], None]:
    def bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, root: Path, filename: Optional[str] = None) -> Path:
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
                url,
                str(fpath),
                reporthook=gen_bar_updater(tqdm(unit="B", unit_scale=True)),
            )
        except OSError:
            print(f"Failed to download from url: {url}")

    return fpath
