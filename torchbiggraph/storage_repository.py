import logging
import pathlib
from abc import abstractmethod
from contextlib import AbstractContextManager
from io import TextIOWrapper
from pathlib import Path
from types import TracebackType
from typing import IO
from typing import Optional, Type, Union

import h5py

from torchbiggraph.plugin import PluginRegistry
from torchbiggraph.util import url_path, run_external_cmd

logger = logging.getLogger("torchbiggraph")


class Constants:
    GET = "hadoop fs -get {remote_path} {local_path}"
    PUT = "hadoop fs -put -f {local_path} {remote_path}"
    TEST_FILE = "hadoop fs -test -e {remote_path}"
    REMOVE = "hadoop fs -rm {remote_path}"
    H5 = "h5"
    RELOAD = 'reload'
    WRITE_MODES = ['w', 'x', 'a']
    READ_MODE = 'r'


class LocalFileContextManager(AbstractContextManager):

    def __init__(self, path: Path, mode: str, **kwargs) -> None :
        self._path: Path = path
        self.mode = mode
        self.kwargs = kwargs

    def __enter__(self) -> Union[h5py.File, TextIOWrapper]:
        self._file = self.get_resource(str(self._path), self.mode)
        return self._file

    def __exit__(self, exception_type: Optional[Type[BaseException]],
                 exception_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> bool:
        self._file.close()
        self._file = None
        return True

    @staticmethod
    def get_resource(filepath: str, mode: str) -> Union[h5py.File, IO[str]]:
        # get file handler
        if filepath.split(".")[-1] == Constants.H5:
            # check if the file being read is a valid h5 file
            if 'r' in mode:
                if Path(filepath).exists() and not h5py.is_hdf5(filepath):
                    raise ValueError('Invalid .h5 file', filepath)
            return h5py.File(filepath, mode)
        else:
            return open(filepath, mode)


class HDFSFileContextManager(AbstractContextManager):
    def __init__(self, path: Path, mode: str, **kwargs) -> None :
        self._path: Path = path
        self.mode = mode
        self.kwargs = kwargs

    def __enter__(self) -> Union[h5py.File, TextIOWrapper]:
        reload = True
        if Constants.RELOAD in self.kwargs:
            reload = self.kwargs[Constants.RELOAD]

        if Constants.READ_MODE in self.mode:
            self.get_from_hdfs(reload)

        self._file = LocalFileContextManager.get_resource(str(self._path), self.mode)
        return self._file


    def __exit__(self, exception_type: Optional[Type[BaseException]],
                 exception_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> bool:
        self._file.close()
        self._file = None

        if any(ext in self.mode for ext in Constants.WRITE_MODES):
            self.put_to_hdfs()

        return True

    @staticmethod
    def get_hdfs_path(path) -> str:
        return str(path).replace(str(Path.cwd()), '', 1)

    @staticmethod
    def hdfs_file_exists(path: str) -> bool:
        rcode = 1
        try:
            rcode, output, errors = run_external_cmd(Constants.TEST_FILE.format(remote_path=path))
        except:
            pass

        return rcode == 0

    def get_from_hdfs(self, reload):
        local_path = self._path
        hdfs_loc = self.get_hdfs_path(local_path)

        if hdfs_loc != str(local_path):
            # check if hdfs file exists before running get command
            if self.hdfs_file_exists(hdfs_loc):
                local_path.parent.mkdir(parents=True, exist_ok=True)

                if local_path.exists():
                    if reload:
                        local_path.unlink()
                        run_external_cmd(Constants.GET.format(remote_path=hdfs_loc, local_path=str(local_path)))
                    else:
                        logger.info(f"Skip get: reload is {reload} and local file exists : {local_path}")
                else:
                    run_external_cmd(Constants.GET.format(remote_path=hdfs_loc, local_path=str(local_path)))
            else:
                raise FileNotFoundError(f"File {hdfs_loc} not found.")
        else:
            logger.info(f"identical local and hdfs path. Skipping get {str(local_path)}")

    def put_to_hdfs(self):
        local_loc_str = str(self._path)
        hdfs_loc = self.get_hdfs_path(local_loc_str)

        if self._path.exists():
            if hdfs_loc != local_loc_str:
                run_external_cmd(Constants.PUT.format(local_path=local_loc_str, remote_path=hdfs_loc))
            else:
                logger.info('identical local and hdfs path. Skipping put ..', local_loc_str)
        else:
            raise FileNotFoundError("File " + local_loc_str + " not found.")


class AbstractPath(type(pathlib.Path())):
    @abstractmethod
    def __init__(self, path: Union[str, Path]):
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self, key: str) -> 'AbstractPath':
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parent(self) -> 'AbstractPath':
        raise NotImplementedError

    @abstractmethod
    def resolve(self, strict: bool = False) -> 'AbstractPath':
        raise NotImplementedError

    @abstractmethod
    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, **kwargs):
        pass

    @abstractmethod
    def cleardir(self) -> None:
        raise NotImplementedError

CUSTOM_PATH = PluginRegistry[AbstractPath]()

@CUSTOM_PATH.register_as("")
@CUSTOM_PATH.register_as("file")
class LocalPath(AbstractPath):
    def __init__(self, path: Union[str, Path]):
        _tmp_path = ''
        if isinstance(path, str):
            _tmp_path = path
        elif isinstance(path, Path):
            _tmp_path = str(path)

        self._path = Path(url_path(_tmp_path))

    def __truediv__(self, key: str):
        return LocalPath(self._path / key)

    def __str__(self):
        return str(self._path)

    @property
    def parent(self):
        return LocalPath(self._path.parent)

    def resolve(self, strict: bool = False) -> 'LocalPath':
        resolved_path = self._path.resolve(strict)
        return LocalPath(resolved_path)

    def cleardir(self) -> None:
        pass

    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, **kwargs):
        return LocalFileContextManager(self._path, mode, **kwargs)


@CUSTOM_PATH.register_as("hdfs")
class HDFSPath(AbstractPath):
    def __init__(self, path: Union[str, Path]):
        _tmp_path = ''
        if isinstance(path, str):
            _tmp_path = path
        elif isinstance(path, Path):
            _tmp_path = str(path)

        self._path = Path(url_path(_tmp_path))

    def __truediv__(self, key: str) -> 'HDFSPath':
        return HDFSPath(self._path / key)

    def __str__(self):
        return str(self._path)

    @property
    def parent(self) -> 'HDFSPath':
        return HDFSPath(self._path.parent)

    def resolve(self, strict: bool = False) -> 'HDFSPath':
        resolved_path = Path('./' + str(self._path)).resolve(strict)
        return HDFSPath(resolved_path)

    def cleardir(self) -> None:

        resolved_path = self._path.resolve(strict=True)

        if not resolved_path.is_dir():
            raise ValueError(f"Not a directory: {resolved_path}")
        else:
            logger.info(f"Clean directory : {resolved_path}")
            for file in resolved_path.iterdir():
                if file.is_file():
                    logger.info(f"Deleting file : {file}")
                    file.unlink()

    def unlink(self) -> None:
        _hdfs_path = HDFSFileContextManager.get_hdfs_path(self._path)

        if HDFSFileContextManager.hdfs_file_exists(_hdfs_path):
            run_external_cmd(Constants.REMOVE.format(remote_path=str(_hdfs_path)))
        else:
            logger.info('hdfs file not found. Skipping : {}'.format(_hdfs_path))
        self._path.unlink()

    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, **kwargs):
        return HDFSFileContextManager(self._path, mode, **kwargs)