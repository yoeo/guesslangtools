from csv import DictReader
from io import BytesIO
import logging
from pathlib import Path
from random import randint
from secrets import token_bytes, token_hex
import shlex
from shutil import rmtree
from subprocess import check_output, STDOUT, CompletedProcess
from tarfile import TarFile, TarInfo
from tempfile import mkdtemp
from unittest.mock import patch
import warnings
from zipfile import ZipFile

from guesslangtools.common import Config, File, absolute
from guesslangtools.app import run_workflow
from guesslangtools.workflow.repositories_dataset import DATASET_FILENAME
from guesslangtools.workflow.source_files import Status


REPO_PER_LANG = 10
FILES_PER_LANG_PER_REPO = (5, 10)

REPO_LIST_HEADERS = (
    'ID,Host Type,Name with Owner,Description,Fork,Created Timestamp,'
    'Updated Timestamp,Last pushed Timestamp,Homepage URL,Size,Stars Count,'
    'Language,Issues enabled,Wiki enabled,Pages enabled,Forks Count,Mirror URL,'
    'Open Issues Count,Default branch,Watchers Count,UUID,'
    'Fork Source Name with Owner,License,Contributors Count,Readme filename,'
    'Changelog filename,Contributing guidelines filename,License filename,'
    'Code of Conduct filename,Security Threat Model filename,'
    'Security Audit filename,Status,Last Synced Timestamp,SourceRank,'
    'Display Name,SCM type,Pull requests enabled,Logo URL,Keywords'
)
REPO_LINE = ',GitHub,{full_name},,false,,,,,,,{lang},,,,,,,,,,,,,,,,,,,,,,,,,,,'

GIT_SETUP_COMMANDS = """
    git init .
    git checkout -B master
    git add .
    git commit -m "init"
    git rm *
"""


# Helpers


def generate_dataset(_, destination):
    csv_lines = [REPO_LIST_HEADERS]
    for lang in Config.languages:
        for pos in range(REPO_PER_LANG):
            full_name = f'{token_hex()[:10]}/{token_hex()}'
            csv_lines.append(REPO_LINE.format(full_name=full_name, lang=lang))

    csv_bytes = '\n'.join(csv_lines).encode()
    with TarFile.open(destination, 'w:gz') as tar_file:
        tar_info = TarInfo(DATASET_FILENAME)
        tar_info.size = len(csv_bytes)
        tar_file.addfile(tar_info, BytesIO(csv_bytes))

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
    generate_dataset)
@patch(
    'guesslangtools.workflow.github_repositories.run',
    create_repository)
def test_workflow(caplog):
    caplog.set_level(logging.INFO)
    warnings.filterwarnings('error', module='pandas')

    run_workflow()
    check_files()
