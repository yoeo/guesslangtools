from csv import DictReader
import logging
from pathlib import Path
from random import randint
from secrets import token_bytes
import shlex
from shutil import rmtree
from subprocess import check_output, STDOUT, CompletedProcess
from tempfile import mkdtemp
from unittest.mock import patch
import warnings
from zipfile import ZipFile

from guesslangtools.common import Config, File, absolute
from guesslangtools.app import run_workflow
from guesslangtools.workflow.source_files import Status


DATA_PATH = Path(__file__).parent.joinpath('data')
FILES_PER_LANG_PER_REPO = (5, 10)
GIT_SETUP_COMMANDS = """
    git init .
    git checkout -B master
    git add .
    git commit -m "init"
    git rm *
"""


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


def create_repository(command, *_, **__):
    # Generate repository files
    destination = Path(command[-1])
    destination.mkdir()

    data_dir = destination.joinpath('data')
    data_dir.mkdir()
    data_dir.joinpath('file.txt').write_text('text')

    src_dir = destination.joinpath('src')
    src_dir.mkdir()

    for lang, exts in Config.languages.items():
        lang_dir = src_dir.joinpath(lang)
        lang_dir.mkdir()
        lang_dir.joinpath('file.txt').write_text('text {lang}')

        ext = exts[0]
        for index in range(randint(*FILES_PER_LANG_PER_REPO)):
            content = f'{destination}:{lang}:{index:02}'
            lang_dir.joinpath(f'unicode_{index:02}.{ext}').write_text(content)

            if index % 3 == 0:
                noise = token_bytes(30)
                lang_dir.joinpath(f'binary_{index:02}.{ext}').write_bytes(noise)

    # Setup git shallow repository
    for line in GIT_SETUP_COMMANDS.strip().splitlines():
        git_command = shlex.split(line.strip())
        check_output(git_command, cwd=destination, stderr=STDOUT)

    return CompletedProcess(command, 0, stdout=b'')


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
    'guesslangtools.workflow.compressed_repositories.run',
    create_repository)
def test_workflow(caplog):
    caplog.set_level(logging.INFO)
    warnings.filterwarnings('error', module='pandas')

    run_workflow()
    check_files()
