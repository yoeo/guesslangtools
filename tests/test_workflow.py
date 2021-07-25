from csv import DictReader
from io import BytesIO
import logging
from pathlib import Path
from secrets import token_bytes
import shlex
from shutil import rmtree
from subprocess import check_output, STDOUT
from tarfile import TarFile, TarInfo
from tempfile import mkdtemp
from unittest.mock import patch
import warnings

from pytest import fixture

from guesslangtools.common import Config, File
from guesslangtools.app import run_workflow
from guesslangtools.workflow.repositories_dataset import DATASET_FILENAME
from guesslangtools.workflow.source_files import Status


REPO_PER_LANG = 3
FILES_PER_LANG_PER_REPO = 3
FILES_PER_LANG_PER_DATASET = 3  # < (files per repo * total repo) / 3 datasets
FILES_PER_LANG = FILES_PER_LANG_PER_DATASET * 3  # 3 datasets

REPO_LIST_HEADERS = (
    'ID,Host Type,Name with Owner,Description,Fork,Created Timestamp,'
    'Updated Timestamp,Last pushed Timestamp,Homepage URL,Size,Stars Count,'
    'Language,Issues enabled,Wiki enabled,Pages enabled,Forks Count,'
    'Mirror URL,Open Issues Count,Default branch,Watchers Count,UUID,'
    'Fork Source Name with Owner,License,Contributors Count,Readme filename,'
    'Changelog filename,Contributing guidelines filename,License filename,'
    'Code of Conduct filename,Security Threat Model filename,'
    'Security Audit filename,Status,Last Synced Timestamp,SourceRank,'
    'Display Name,SCM type,Pull requests enabled,Logo URL,Keywords'
)
REPO_LINE = (
    ',GitHub,{full_name},,false,,,,,,,{lang},,,,,,,,,,,,,,,,,,,,,,,,,,,'
)

GIT_SETUP_COMMANDS = """
    git init .
    git checkout -B master
    git add .
    git commit -m "init"
    git rm *
"""


# Helpers


def generate_dataset(config):
    destination = config.absolute(File.COMPRESSED_DATASET)
    csv_lines = [REPO_LIST_HEADERS]
    for lang, ext in config.extensions.items():
        for pos in range(REPO_PER_LANG):
            full_name = f'lang_{ext}/repo_{pos:02}'
            csv_lines.append(REPO_LINE.format(full_name=full_name, lang=lang))

    csv_bytes = '\n'.join(csv_lines).encode()
    with TarFile.open(destination, 'w:gz') as tar_file:
        tar_info = TarInfo(DATASET_FILENAME)
        tar_info.size = len(csv_bytes)
        tar_file.addfile(tar_info, BytesIO(csv_bytes))

    return True, 200


def create_repositories(_, rows, config, *__, **___):
    for item in rows:
        # Generate repository files
        path = config.repositories_dir.joinpath(item['repository_dirname'])
        path.mkdir()

        data_dir = path.joinpath('data')
        data_dir.mkdir()
        data_dir.joinpath('file.txt').write_text('text')

        src_dir = path.joinpath('src')
        src_dir.mkdir()

        for lang, ext in config.extensions.items():
            lang_dir = src_dir.joinpath(lang)
            lang_dir.mkdir()

            # Create plain text file => should be ignored
            lang_dir.joinpath('file.txt').write_text('text {lang}')

            # Create binary file => should be ignored
            noise = token_bytes(30)
            lang_dir.joinpath(f'binary_00.{ext}').write_bytes(noise)

            # Create source file => should be kept
            for index in range(FILES_PER_LANG_PER_REPO):
                content = f'{path}:{lang}:{index:02}'
                filename = f'unicode_{index:02}.{ext}'
                lang_dir.joinpath(filename).write_text(content)

            # Create source file with exact name => should be kept
            for filename, languages in config.file_mapping.items():
                if lang in languages:
                    content = f'{path}:{lang}:{filename}'
                    lang_dir.joinpath(filename).write_text(content)

        # Setup git shallow repository
        for line in GIT_SETUP_COMMANDS.strip().splitlines():
            git_command = shlex.split(line.strip())
            check_output(git_command, cwd=path, stderr=STDOUT)

        item['repository_is_empty'] = not any(path.iterdir())
        yield item


def check_files(config):
    path = config.absolute(File.EXTRACTED_FILES)
    assert path.exists()

    files = {lang: 0 for lang in config.languages}
    with path.open() as csv_file:
        for item in DictReader(csv_file):
            if not item['status'] == Status.EXTRACTED.value:
                continue

            lang = item['language']
            path_elements = ('files', item['usage'], item['extract_to'])
            extracted_path = config.absolute(*path_elements)
            ext = extracted_path.suffix.lstrip('.')

            assert extracted_path.exists()
            assert ext == config.extensions[lang]

            files[lang] += 1

    for lang, nb_files in files.items():
        message = f'Lang {lang}, expected: {FILES_PER_LANG}, found: {nb_files}'
        assert nb_files == FILES_PER_LANG, message


# Test

@fixture()
def config():
    tempdir = mkdtemp(prefix='guesslangtools_unittest_')
    print(f'Temporary config directory: {tempdir}')
    config = Config(
        cache_dir=tempdir,
        nb_repositories=REPO_PER_LANG,
        nb_train=FILES_PER_LANG_PER_DATASET,
        nb_valid=FILES_PER_LANG_PER_DATASET,
        nb_test=FILES_PER_LANG_PER_DATASET,
    )
    assert config.cache_path == Path(tempdir).absolute()
    try:
        yield config
    finally:
        rmtree(config.cache_path)


@patch(
    'guesslangtools.workflow.repositories_dataset.download',
    generate_dataset,
)
@patch(
    'guesslangtools.workflow.github_repositories.pool_map',
    create_repositories,
)
def test_workflow(config, caplog):
    caplog.set_level(logging.INFO)
    warnings.filterwarnings('error', module='pandas')

    run_workflow(config)
    check_files(config)
