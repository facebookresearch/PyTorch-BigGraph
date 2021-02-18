import shutil
import tempfile
from contextlib import AbstractContextManager
from io import TextIOWrapper
from pathlib import Path
from unittest import TestCase, main
import h5py

from torchbiggraph.storage_repository import CUSTOM_PATH
from torchbiggraph.storage_repository import LocalPath, HDFSPath, HDFSFileContextManager, LocalFileContextManager
from torchbiggraph.util import run_external_cmd, url_scheme

HDFS_TEST_PATH = '<valid hdfs path>'


def _touch_file(name: str):
    file_path = HDFS_TEST_PATH + "/" + name
    run_external_cmd("hadoop fs -touchz " + file_path)
    return file_path


class TestLocalFileContextManager(TestCase):
    def setUp(self):
        self.resource_dir = Path(__file__).parent.absolute() / 'resources'

    def test_get_resource_valid_h5(self):
        filepath_h5 = self.resource_dir / 'edges_0_0.h5'
        file_path = str(filepath_h5)
        self.assertIs(type(LocalFileContextManager.get_resource(file_path, 'r')), h5py.File)

    def test_get_resource_invalid_h5(self):
        filepath_h5 = self.resource_dir / 'invalidFile.h5'
        file_path = str(filepath_h5)

        with self.assertRaises(ValueError):
            LocalFileContextManager.get_resource(str(file_path), 'r')

    def test_get_resource_valid_text_file(self):
        filepath_txt = self.resource_dir / 'text.txt'
        file_path = str(filepath_txt)
        self.assertIs(type(LocalFileContextManager.get_resource(str(file_path), 'r')), TextIOWrapper)



class TestHDFSFileContextManager(TestCase):
    def setUp(self):
        self.resource_dir = Path(__file__).parent.absolute() / 'resources'
        run_external_cmd("hadoop fs -mkdir -p " + HDFS_TEST_PATH)

    def tearDown(self):
        run_external_cmd("hadoop fs -rm -r " + HDFS_TEST_PATH)

    def test_prepare_hdfs_path(self):
        actual = HDFSFileContextManager.get_hdfs_path(Path.cwd() / '/some/path')
        expected = '/some/path'
        self.assertEqual(str(expected), actual)

    def test_hdfs_file_exists(self):
        valid_path = _touch_file('abc')
        self.assertTrue(HDFSFileContextManager.hdfs_file_exists(valid_path))

    def test_hdfs_file_doesnt_exists(self):
        invalid_path = HDFS_TEST_PATH + "/invalid_loc"
        self.assertFalse(HDFSFileContextManager.hdfs_file_exists(invalid_path))

    def test_get_from_hdfs_valid(self):
        valid_hdfs_file = _touch_file('valid.file')
        local_file = Path(str(Path.cwd()) + valid_hdfs_file)
        file_ctx = HDFSFileContextManager(local_file, 'r')

        # valid path
        file_ctx.get_from_hdfs(reload=True)
        self.assertTrue(Path(file_ctx._path).exists())

    def test_get_from_hdfs_valid_dont_reload(self):
        valid_hdfs_file = _touch_file('valid.file')
        local_file = Path(str(Path.cwd()) + valid_hdfs_file)
        file_ctx = HDFSFileContextManager(local_file, 'r')

        # valid path
        file_ctx.get_from_hdfs(reload=False)
        self.assertTrue(Path(file_ctx._path).exists())

    def test_get_from_hdfs_invalid(self):
        invalid_hdfs_file = Path('./' + HDFS_TEST_PATH + "/invalid_loc").resolve()
        file_ctx = HDFSFileContextManager(invalid_hdfs_file, 'r')

        # invalid path
        with self.assertRaises(FileNotFoundError):
            file_ctx.get_from_hdfs(reload=True)

    def test_put_to_hdfs(self):
        local_file_name = 'test_local.file'
        local_file = Path(str(Path.cwd()) + HDFS_TEST_PATH + '/' + local_file_name)
        file_ctx = HDFSFileContextManager(local_file, 'w')

        # clean up local
        if local_file.exists():
            local_file.unlink()

        # invalid path
        with self.assertRaises(FileNotFoundError):
            file_ctx.put_to_hdfs()

        # create local file
        local_file.touch()
        file_ctx.put_to_hdfs()
        self.assertTrue(HDFSFileContextManager.hdfs_file_exists(HDFS_TEST_PATH + '/' + local_file_name))


class TestLocalPath(TestCase):
    def setUp(self):
        self.resource_dir = Path(__file__).parent.absolute() / 'resources'

    def test_init(self):
        path = LocalPath(Path.cwd())
        self.assertIs(type(path), LocalPath)

        path = LocalPath('some/path')
        self.assertIs(type(path), LocalPath)

    def test_stem_suffix(self):
        path = LocalPath('some/path/name.txt')
        self.assertTrue(path.stem == 'name')
        self.assertTrue(path.suffix == '.txt')
        self.assertIsInstance(path.stem, str)

    def test_name(self):
        path = LocalPath('some/path/name')
        self.assertTrue(path.name == 'name')
        self.assertIsInstance(path.name, str)

    def test_resolve(self):
        path = LocalPath('some/path/name')
        actual = path.resolve(strict=False)
        expected = Path.cwd() / Path(str(path))
        self.assertTrue(str(actual) == str(expected))

    def test_exists(self):
        invalid_path = LocalPath('some/path/name')
        self.assertFalse(invalid_path.exists())

        valid_path = LocalPath(Path(__file__))
        self.assertTrue(valid_path.exists())

    def test_append_path(self):
        path = LocalPath('/some/path/name')
        actual = path / 'storage_manager.py'
        expected = '/some/path/name/storage_manager.py'
        self.assertTrue(str(actual) == expected)

    def test_open(self):
        file_path = Path(__file__)
        with file_path.open('r') as fh:
            self.assertGreater(len(fh.readlines()), 0)

    def test_mkdir(self):
        path = LocalPath(self.resource_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

    def test_with_plugin_empty_scheme(self):
        local_path = '/some/path/file.txt'
        actual_path = CUSTOM_PATH.get_class(url_scheme(local_path))(local_path)
        expected_path = '/some/path/file.txt'
        self.assertEqual(str(actual_path), str(expected_path))

    def test_with_plugin_file_scheme(self):
        local_path = 'file:///some/path/file.txt'
        actual_path = CUSTOM_PATH.get_class(url_scheme(local_path))(local_path)
        expected_path = '/some/path/file.txt'
        self.assertEqual(str(expected_path), str(actual_path))


class TestHDFSDataPath(TestCase):

    def setUp(self):
        self.resource_dir = Path(__file__).parent.absolute() / 'resources'
        run_external_cmd("hadoop fs -mkdir -p " + HDFS_TEST_PATH)

    def tearDown(self):
        run_external_cmd("hadoop fs -rm -r " + HDFS_TEST_PATH)

    def test_delete_valid(self):
        valid_path = _touch_file('abc.txt')
        local_temp_dir = str(Path.cwd()) + '/' + 'axp'

        # create resolved path based on the hdfs path
        remote_path = HDFSPath(valid_path).resolve(strict=False)
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path.touch()
        remote_path.unlink()

        # remove local path
        shutil.rmtree(local_temp_dir, ignore_errors=True)

    def test_delete_invalid(self):
        invalid_path = HDFSPath(HDFS_TEST_PATH + '/invalid.file')
        with self.assertRaises(FileNotFoundError):
            invalid_path.unlink()

    def test_open(self):
        filepath_h5 = self.resource_dir / 'edges_0_0.h5'
        hdfs = HDFSPath(filepath_h5).resolve(strict=False)
        with hdfs.open('r') as fh:
            self.assertEqual(len(fh.keys()), 3)
            self.assertIsInstance(fh, AbstractContextManager)

    def test_open_reload_False(self):
        filepath_h5 = self.resource_dir / 'edges_0_0.h5'
        hdfs = HDFSPath(filepath_h5).resolve(strict=False)
        with hdfs.open('r', reload=False) as fh:
            self.assertEqual(len(fh.keys()), 3)
            self.assertIsInstance(fh, AbstractContextManager)

    def test_name(self):
        hdfs = HDFSPath('/some/path/file.txt')
        self.assertEqual(hdfs.name, 'file.txt')

    def test_with_plugin(self):
        hdfs_path = 'hdfs:///some/path/file.txt'
        actual_path = CUSTOM_PATH.get_class(url_scheme(hdfs_path))(hdfs_path).resolve(strict=False)
        expected_path = Path.cwd() / 'some/path/file.txt'
        self.assertEqual(str(actual_path), str(expected_path))

    def test_append_path(self):
        path = HDFSPath('/some/path/name')
        actual = path.resolve(strict = False) / 'storage_manager.py'
        expected = str(Path.cwd() / 'some/path/name/storage_manager.py')
        self.assertEqual(expected, str(actual))

    def test_stem_suffix(self):
        path = HDFSPath('some/path/name.txt')
        self.assertTrue(path.stem == 'name')
        self.assertTrue(path.suffix == '.txt')
        self.assertIsInstance(path.stem, str)

    def test_cleardir(self):
        # create empty files

        tempdir = tempfile.mkdtemp()
        Path(tempdir + 'file1.txt').touch()
        Path(tempdir + 'file2.txt').touch()
        Path(tempdir + 'file3.txt').touch()

        dirpath = HDFSPath(tempdir)
        dirpath.cleardir()

        self.assertFalse(any(dirpath.iterdir()))


if __name__ == "__main__":
    main()
