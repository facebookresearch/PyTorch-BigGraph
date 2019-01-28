#!/usr/bin/env python3

import gzip
import os
import tarfile
import urllib

from tqdm import tqdm


def extract_gzip(gzip_path, remove_finished=False):
    print('Extracting {}'.format(gzip_path))
    fpath, ext = os.path.splitext(gzip_path)
    if ext != ".gz":
        raise RuntimeError("Not a gzipped file")

    with open(fpath, 'wb') as out_f, gzip.GzipFile(gzip_path) as zip_f:
        out_f.write(zip_f.read())
    if remove_finished:
        os.unlink(gzip_path)

    return fpath


def extract_tar(fpath):
    # extract file
    root = os.path.dirname(fpath)
    with tarfile.open(fpath, "r:gz") as tar:
        tar.extractall(path=root)


def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root, filename=None):
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
