from collections import OrderedDict
import csv
from enum import Enum, auto
from functools import partial
import json
import logging
from pathlib import Path
import random
from subprocess import run, check_output, PIPE, CalledProcessError
from typing import Dict, List, Any, Tuple, Optional
from uuid import uuid4

import chardet
import pandas as pd

from guesslangtools.common import (
    Config,
    File,
    cached,
    pool_map,
    LOG_STEP,
    CSV_FIELD_LIMIT,
    MAX_FILES_PER_REPOSITORY_PER_LANGUAGE,
)


LOGGER = logging.getLogger(__name__)

MIN_REPOSITORIES = 3
MIN_ENCODING_CONFIDENCE = 0.5
MIN_SPLIT_RATIO = 0.15

GIT_LIST_FILES = ['git', 'ls-tree', '-r', 'HEAD']
GIT_RESET_FILES = ['git', 'checkout', 'HEAD']
GIT_DISABLE_GC = ['git', 'config', 'gc.auto', '0']
GIT_RESET_FILES = ['timeout', '600', 'git', 'checkout', 'HEAD']
GIT_EMPTY_FILE_KEY = '0xe69de29bb2d1d6434b8b29ae775ad8c2e48c5391'

AVAILABLE_FILES_COLUMNS = [
    'extract_to',
    'dedup_key',
    'filename',
    'language',
    'rank',
    'repository_dirname',
    'repository_language',
]

EXTRACTED_FILES_COLUMNS = [
    'extract_to',
    'filename',
    'language',
    'rank',
    'repository_dirname',
    'repository_language',
    'usage',
    'status',
]

random.seed()


class Status(Enum):
    """File extraction status"""

    def _generate_next_value_(name: str, *_: Any) -> str:  # type: ignore
        # Mypy requires a @staticmethod here but this is a "magic" function
        # that doesn't work as a static method.
        return name

    PENDING = auto()
    SELECTED = auto()
    EXTRACTED = auto()
    DISCARDED = auto()


# Always (re-)generates File.AVAILABLE_FILES
@cached(File.DEDUPLICATED_FILES)
def list_all(config: Config) -> None:
    LOGGER.info('List source files from repositories')
    LOGGER.info('This operation might take several minutes...')

    # Start or resume files listing
    repo = config.load_csv(File.DOWNLOADED_REPOSITORIES)
    try:
        files = config.load_csv(File.AVAILABLE_FILES)
    except IOError:
        files = pd.DataFrame([], columns=AVAILABLE_FILES_COLUMNS)

    # Find repositories that have not been processed yet
    mask = ~repo['repository_dirname'].isin(files['repository_dirname'])
    new_repo = repo[mask]
    LOGGER.info(f'{len(new_repo)} newly downloaded repositories')

    # Show the number of deleted repositories
    nb_repo_before = len(files['repository_dirname'].unique())
    mask = files['repository_dirname'].isin(repo['repository_dirname'])
    files = files[mask]
    nb_repo_after = len(files['repository_dirname'].unique())
    nb_removed = nb_repo_before - nb_repo_after
    LOGGER.info(f'{nb_removed} deleted repositories')

    # List unprocessed repositories files
    total = len(new_repo)
    rows = (dict(repo) for _, repo in new_repo.iterrows())

    output_path = config.absolute(File.AVAILABLE_FILES)
    write_headers = not output_path.exists()
    csv.field_size_limit(CSV_FIELD_LIMIT)
    with output_path.open('a') as output:
        writer = csv.DictWriter(output, fieldnames=AVAILABLE_FILES_COLUMNS)
        if write_headers:
            writer.writeheader()

        for index, result in enumerate(pool_map(_list_files, rows, config)):
            for item in result:
                writer.writerow(item)

            if index % LOG_STEP == 0:
                LOGGER.info(f'--> Processed {index} / {total} repositories...')
        LOGGER.info(f'--> Processed {total} / {total} repositories!')

    LOGGER.info(f'Created file: {output_path}')


def _list_files(item: Dict[str, str], config: Config) -> List[Dict[str, Any]]:
    max_files = MAX_FILES_PER_REPOSITORY_PER_LANGUAGE
    repository_language = item['repository_language']
    repository_dirname = item['repository_dirname']
    repository_path = config.repositories_dir.joinpath(repository_dirname)

    files = _repository_files(repository_path)
    random.shuffle(files)

    result = []
    file_counter = {lang: 0 for lang in config.languages}
    for filename, dedup_key in files:
        if dedup_key == GIT_EMPTY_FILE_KEY:
            continue

        if filename.endswith('/'):
            continue

        lang = _find_language(
            filename,
            repository_language,
            config.ext_mapping,
            config.file_mapping,
        )

        if not lang or file_counter[lang] >= max_files:
            continue

        file_counter[lang] += 1
        extracted_ext = config.extensions[lang]
        extract_to_filename = f'{uuid4()}.{extracted_ext}'
        result.append(
            {
                'language': lang,
                'repository_language': repository_language,
                'repository_dirname': repository_dirname,
                'filename': filename,
                'dedup_key': dedup_key,
                'extract_to': extract_to_filename,
                'rank': file_counter[lang],
            }
        )

    return result


def _repository_files(repository_path: Path) -> List[Tuple[str, str]]:
    try:
        raw_result = check_output(
            GIT_LIST_FILES, cwd=repository_path, stderr=PIPE
        )
    except CalledProcessError:
        LOGGER.warning(f'Cannot list files from repository {repository_path}')
        return []

    result = raw_result.decode().strip()
    if not result:
        return []

    compressed_files = []
    for info in result.split('\n'):
        try:
            _, _, dedup, filename = info.strip().split(maxsplit=3)
        except ValueError:  # empty filename
            continue

        dedup_key = f'0x{dedup}'
        compressed_files.append((filename, dedup_key))
    return compressed_files


def _find_language(
    filename: str,
    repository_language: str,
    ext_mapping: Dict[str, List[str]],
    file_mapping: Dict[str, List[str]],
) -> Optional[str]:
    path = Path(filename)

    basename = path.name
    languages = file_mapping.get(basename, [])
    if len(languages) == 1:
        return languages[0]
    elif repository_language in languages:
        return repository_language

    ext = path.suffix.lstrip('.').lower()
    languages = ext_mapping.get(ext, [])
    if len(languages) == 1:
        return languages[0]
    elif repository_language in languages:
        return repository_language

    return None


@cached(File.DEDUPLICATED_FILES)
def deduplicate(config: Config) -> None:
    df = config.load_csv(File.AVAILABLE_FILES)
    df.drop_duplicates(subset='dedup_key', inplace=True)
    df.sort_values(by='rank', inplace=True)

    LOGGER.info('Files available by language:')
    for lang in config.languages:
        nb_files = len(df[df['language'] == lang])
        LOGGER.info(f'--> {lang}: {nb_files}')

    config.save_csv(df, File.DEDUPLICATED_FILES)


@cached(File.FILES_SPLIT_BY_USAGE)
def split(config: Config) -> None:
    LOGGER.info('Split repositories by usage: train, valid & test')
    LOGGER.info('This operation should take few seconds...')

    files = config.load_csv(File.DEDUPLICATED_FILES)
    files = files.drop('dedup_key', axis=1)
    repo_columns = ['repository_language', 'repository_dirname']

    repo = files[repo_columns].drop_duplicates()
    repo = repo.sample(frac=1).reset_index(drop=True)
    repo.loc[:, 'usage'] = ''

    LOGGER.info(f'Total downloaded repositories: {len(repo)}')

    total_files = (
        config.nb_train_files_per_language
        + config.nb_valid_files_per_language
        + config.nb_test_files_per_language
    )
    valid_ratio = config.nb_valid_files_per_language / total_files
    valid_ratio = max(valid_ratio, MIN_SPLIT_RATIO)

    test_ratio = config.nb_test_files_per_language / total_files
    test_ratio = max(test_ratio, MIN_SPLIT_RATIO)

    repositories = {}
    for lang in config.languages:
        by_language = repo[repo['repository_language'] == lang]
        total = len(by_language)
        if total < MIN_REPOSITORIES:
            raise RuntimeError(
                f'Need more than {MIN_REPOSITORIES}, '
                f'only {total} repositories usable for language {lang}'
            )

        nb_test = max(int(total*test_ratio), 1)
        nb_valid = max(int(total*valid_ratio), 1)
        nb_test_valid = nb_test + nb_valid

        test = by_language[:nb_test]
        test['usage'].values[:] = 'test'
        repositories[f'{lang}/test'] = test

        valid = by_language[nb_test:nb_test_valid]
        valid['usage'].values[:] = 'valid'
        repositories[f'{lang}/valid'] = valid

        train = by_language[nb_test_valid:]
        train['usage'].values[:] = 'train'
        repositories[f'{lang}/train'] = train

        LOGGER.info(
            f'{lang} nb repositories, train: {total-nb_test_valid}, '
            f'valid: {nb_valid}, test: {nb_test}'
        )

    for name, repository in repositories.items():
        if not len(repository):
            LOGGER.error(f'No repositories available for {name}')
            raise RuntimeError(f'No repositories for category: {name}')

    repo = pd.concat(repositories.values())
    files = pd.merge(files, repo, on=repo_columns)
    config.save_csv(files, File.FILES_SPLIT_BY_USAGE)


def extract(config: Config) -> None:
    LOGGER.info('Extract selected files')
    LOGGER.info('This operation might take a lot of time...')

    train_path = config.extracted_files_dir.joinpath('train')
    valid_path = config.extracted_files_dir.joinpath('valid')
    test_path = config.extracted_files_dir.joinpath('test')

    train_path.mkdir(exist_ok=True)
    valid_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    # Load list of files to extract
    source = config.load_csv(File.FILES_SPLIT_BY_USAGE)

    # Load list of processed files
    try:
        files = config.load_csv(File.EXTRACTED_FILES)
    except IOError:
        files = pd.DataFrame([], columns=EXTRACTED_FILES_COLUMNS)

    df = pd.merge(source, files, how='outer', on=list(source.columns))
    df.loc[df['status'].isnull(), 'status'] = Status.PENDING.value

    # Flag existing files
    is_pending = df['status'] == Status.PENDING.value
    file_exists = df.apply(partial(_destination_exists, config), axis=1)
    df.loc[(is_pending & file_exists), 'status'] = Status.DISCARDED.value

    while True:
        selected = _choose_files_to_extract(config, df)
        LOGGER.info(f'{len(selected)} files to extract')

        if not len(selected):
            break

        result = _extract_files(config, selected)

        result_extracted = result[result['status'] == Status.EXTRACTED.value]
        mask = df['extract_to'].isin(result_extracted['extract_to'])
        df.loc[mask, 'status'] = Status.EXTRACTED.value

        result_discarded = result[result['status'] == Status.DISCARDED.value]
        mask = df['extract_to'].isin(result_discarded['extract_to'])
        df.loc[mask, 'status'] = Status.DISCARDED.value

        extracted = df[df['status'] == Status.EXTRACTED.value]
        discarded = df[df['status'] == Status.DISCARDED.value]

        LOGGER.info(
            f'Processed {len(result)} files: {len(result_extracted)} '
            f'extracted, {len(result_discarded)} discarded'
        )

        LOGGER.info(f'{len(extracted)} total files extracted')
        LOGGER.info(f'{len(discarded)} total files discarded')

    config.save_csv(df, File.EXTRACTED_FILES)

    LOGGER.info(f'The training files are located in {train_path}')
    LOGGER.info(f'The validation files are located in {valid_path}')
    LOGGER.info(f'The test files are located in {test_path}')


def _destination_exists(config: Config, item: Dict[str, str]) -> bool:
    usage = item['usage']
    extract_to = item['extract_to']
    return config.extracted_files_dir.joinpath(usage, extract_to).exists()


def _choose_files_to_extract(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    usage_info = {
        'train': config.nb_train_files_per_language,
        'valid': config.nb_valid_files_per_language,
        'test': config.nb_test_files_per_language,
    }

    files = []
    mask_pending = df['status'] == Status.PENDING.value
    mask_extracted = df['status'] == Status.EXTRACTED.value

    for lang in config.languages:
        mask_lang = df['language'] == lang

        for usage, nb_files in usage_info.items():
            mask_usage = df['usage'] == usage

            nb_extracted = len(df[mask_lang & mask_usage & mask_extracted])
            nb_files_to_keep = max(nb_files-nb_extracted, 0)

            pending = df[mask_lang & mask_usage & mask_pending]
            kept = pending[:nb_files_to_keep]
            files.append(kept)

            LOGGER.info(
                f'{lang}/{usage}, pending: {len(pending)} files, '
                f'kept: {len(kept)} files'
            )

            if len(kept) < nb_files_to_keep:
                LOGGER.warning(
                    f'{lang}/{usage} minimum required: '
                    f'{nb_files_to_keep} files'
                )

    chosen = pd.concat(files)
    return chosen


def _extract_files(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('repository_dirname')
    nb_groups = len(grouped)
    simple_groups = (
        (dirname, [dict(repo_info) for _, repo_info in items.iterrows()])
        for dirname, items in grouped
    )

    results = []
    pool = pool_map(
        _extract_from_repository, simple_groups, config, multiplier=2
    )
    for index, grouped_results in enumerate(pool, 1):
        results.append(grouped_results)
        if index % LOG_STEP == 0:
            LOGGER.info(f'--> Processed {index} / {nb_groups} repositories...')
    LOGGER.info(f'--> Processed {nb_groups} / {nb_groups} repositories!')

    flattened = (file_info for result in results for file_info in result)
    final_result = pd.DataFrame(flattened)
    return final_result


def _extract_from_repository(
    params: Tuple[str, List[Dict[str, str]]],
    config: Config,
) -> List[Dict[str, str]]:
    repository_dirname, items = params
    repository_path = config.repositories_dir.joinpath(repository_dirname)

    result = run(GIT_DISABLE_GC, stdout=PIPE, stderr=PIPE, cwd=repository_path)
    if result.returncode != 0:
        LOGGER.debug(f'Failed to disable GC in {repository_path}')

    filenames = set(item['filename'] for item in items)
    command = GIT_RESET_FILES + list(filenames)
    result = run(command, stdout=PIPE, stderr=PIPE, cwd=repository_path)
    if result.returncode != 0:
        LOGGER.debug(f'Failed to reset files from {repository_path}')

    return [_move_file(config, repository_path, item) for item in items]


def _move_file(
    config: Config, repository_path: Path, item: Dict[str, str]
) -> Dict[str, Any]:
    usage = item['usage']
    filename = item['filename']
    basename = item['extract_to']
    ko = {'extract_to': basename, 'status': Status.DISCARDED.value}
    ok = {'extract_to': basename, 'status': Status.EXTRACTED.value}

    source = repository_path.joinpath(filename)
    try:
        content = source.read_bytes()
        source.unlink()
    except OSError:
        LOGGER.debug(f'Unreachable file {source}')
        return ko

    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        LOGGER.debug(f'Non UTF-8 text {repository_path}: {filename}')
        detected = chardet.detect(content)
        if detected['confidence'] < MIN_ENCODING_CONFIDENCE:
            LOGGER.debug(f'Bad Encoding {repository_path}: {filename}')
            return ko

        try:
            text = content.decode(detected['encoding'])
        except (UnicodeDecodeError, LookupError):
            LOGGER.debug(f'Bad Encoding {repository_path}: {filename}')
            return ko

    destination = config.extracted_files_dir.joinpath(usage, basename)
    destination.write_text(text)
    return ok


def finalize(config: Config) -> None:
    items = config.extensions.items()
    lang_ext = OrderedDict(sorted(items, key=_lang_name))
    language_filename = config.absolute('languages.json')
    with language_filename.open('w') as output:
        json.dump(lang_ext, output, indent=2)

    LOGGER.info('Dataset successfully generated')
    LOGGER.info('To train Guesslang with this dataset:')
    LOGGER.info(f'* copy {language_filename} into guesslang/data/ directory')
    LOGGER.info(
        f'* run $ guesslang --train {config.cache_path} /path/to/new_model'
    )


def _lang_name(value: Tuple[str, str]) -> str:
    return value[0].lower()
