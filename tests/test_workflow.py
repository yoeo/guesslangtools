from csv import DictReader
from pathlib import Path
from random import randint
from secrets import token_bytes
from shutil import rmtree
from tempfile import mkdtemp
from unittest.mock import patch
import warnings
from zipfile import ZipFile

from guesslangtools.common import Config, File, absolute
from guesslangtools.app import run_workflow
from guesslangtools.workflow.source_files import Status


DATA_PATH = Path(__file__).parent.joinpath('data')
FILES_PER_LANG_PER_REPO = (5, 10)


# Helpers


def check_files():
    path = absolute(File.EXTRACTED_FILES)
    assert path.exists()

    languages = Config.languages
    files = {lang: 0 for lang in languages}

    with path.open() as csv_file:
        for item in DictReader(csv_file):
            if not item['status'] == Status.EXTRACTED.value:
                continue

            language = item['language']
            path_elements = ('files', item['usage'], item['extract_to'])
            extracted_path = absolute(*path_elements)
            ext = extracted_path.suffix.lstrip('.')

            assert extracted_path.exists()
            assert ext in languages[language]

            files[language] += 1

    assert all(count == 30 for count in files.values())


def copy_dataset(_, destination):
    source = 'repositories_dataset.tar.xz'
    source_path = DATA_PATH.joinpath(source)
    destination_path = Path(destination)
    destination_path.write_bytes(source_path.read_bytes())
    return True, 200


def create_repository(_, destination):
    with ZipFile(destination, 'w') as myzip:
        myzip.writestr('data/file.txt', 'text')
        for lang, exts in Config.languages.items():
            myzip.writestr(f'src/{lang}/file.txt', 'text {lang}')
            ext = exts[0]
            for index in range(randint(*FILES_PER_LANG_PER_REPO)):
                filename = f'src/{lang}/unicode_{index:02}.{ext}'
                content = f'{destination}:{lang}:{index:02}'
                myzip.writestr(filename, content)

                if index % 3 == 0:
                    filename = f'src/{lang}/binary_{index:02}.{ext}'
                    content = token_bytes(30)
                    myzip.writestr(filename, content)

    return True, 200


# Tests


def setup_function(_):
    tempdir = mkdtemp(suffix='_gesslangtools_unittest')
    print(f'Temporary config directory: {tempdir}')
    Config.setup(
        cache_dir=tempdir,
        nb_repositories=10,
        nb_train=10,
        nb_valid=10,
        nb_test=10,
    )

    assert Config.cache_dir == tempdir


def teardown_function(_):
    assert Config.cache_dir
    assert Config.cache_dir.endswith('_gesslangtools_unittest')

    rmtree(Config.cache_dir)


@patch(
    'guesslangtools.workflow.repositories_dataset.download_file',
    copy_dataset)
@patch(
    'guesslangtools.workflow.compressed_repositories.download_file',
    create_repository)
def test_workflow():
    warnings.filterwarnings('error', module='pandas')

    run_workflow()
    check_files()
