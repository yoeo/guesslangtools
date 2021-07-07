from contextlib import suppress
from enum import Enum, auto
from itertools import groupby
import logging
from operator import itemgetter
from pathlib import Path
import random
import shutil
from subprocess import run, check_output, PIPE, CalledProcessError
from typing import Dict, List, Any, Tuple, Optional
from uuid import uuid4
from zipfile import ZipFile, BadZipFile

import chardet
import pandas as pd

from guesslangtools.common import (
    Config, File, cached, load_csv, save_csv, pool_map
)


LOGGER = logging.getLogger(__name__)

MIN_REPOSITORIES = 3
MIN_ENCODING_CONFIDENCE = 0.5
MIN_SPLIT_RATIO = 0.15

GIT_LIST_FILES = ['git', 'ls-tree', '-r', 'HEAD']
GIT_RESET_FILES = ['git', 'checkout', 'HEAD']
GIT_DISABLE_GC = ['git', 'config', 'gc.auto', '0']
GIT_RESET_FILES = ['timeout', '600', 'git', 'checkout', 'HEAD']

AVAILABLE_FILES_COLUMNS = [
    'extract_to',
    'dedup_key',
    'filename',
    'language',
    'rank',
    'repository_dirname',
    'repository_language',  # FIXME
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
@cached(File.FILES_SPLIT_BY_USAGE)
def list_all() -> None:
    LOGGER.info('List source files from repositories')
    LOGGER.info('This operation might take several minutes...')

    repo = load_csv(File.DOWNLOADED_REPOSITORIES)
    try:
        files = load_csv(File.AVAILABLE_FILES)
    except IOError:
        files = pd.DataFrame([], columns=AVAILABLE_FILES_COLUMNS)

    mask = ~repo['repository_dirname'].isin(files['repository_dirname'])
    new_repo = repo[mask]
    LOGGER.info(f'{len(new_repo)} newly downloaded repositories')

    nb_repo_before = len(files['repository_dirname'].unique())
    mask = files['repository_dirname'].isin(repo['repository_dirname'])
    files = files[mask]
    nb_repo_after = len(files['repository_dirname'].unique())
    nb_removed = nb_repo_before - nb_repo_after
    LOGGER.info(f'{nb_removed} deleted repositories')

    new_files = _list_files_by_language(new_repo)
    df = pd.concat([files, new_files], axis=0, sort=False)

    df.drop_duplicates(subset='dedup_key', inplace=True)
    df.sort_values(by='rank', inplace=True)

    LOGGER.info('Files available by language:')
    for language in Config.languages:
        nb_files = len(df[df['language'] == language])
        LOGGER.info(f'--> {language}: {nb_files}')

    save_csv(df, File.AVAILABLE_FILES)


def _list_files_by_language(repo: pd.DataFrame) -> pd.DataFrame:
    languages = Config.languages
    nb_files_limit = Config.max_files_per_repository_per_language
    ext_lang, ambiguous = _analyse_languages(languages)
    total_repo = len(repo)

    results = []
    rows = (dict(item) for _, item in repo.iterrows())
    args = (languages, ext_lang, ambiguous, nb_files_limit)
    for index, result in enumerate(pool_map(_select_files, rows, *args)):
        if result:
            results.append(result)

        if index % Config.step == 0:
            LOGGER.info(f'--> Processed {index} / {total_repo} repositories...')
    LOGGER.info(f'--> Processed {total_repo} / {total_repo} repositories!')

    LOGGER.info('Saving source files info')
    flattened = (file_info for result in results for file_info in result)

    output_data = pd.DataFrame(flattened)
    return output_data


def _select_files(
    item: Dict[str, str],
    languages: Dict[str, List[str]],
    ext_lang: Dict[str, str],
    ambiguous: Dict[str, List[str]],
    nb_files_limit: int,
) -> List[Dict[str, Any]]:
    repository_language = item['repository_language']
    repository_dirname = item['repository_dirname']

    files = _list_compressed_files(repository_dirname)
    random.shuffle(files)

    output_items = []
    files_per_lang = {lang: 0 for lang in languages}
    for filename, dedup_key in files:
        lang = _find_language(
            filename, ext_lang, ambiguous, repository_language
        )

        if not lang or files_per_lang[lang] >= nb_files_limit:
            continue

        ext = languages[lang][0]
        extract_to_filename = f'{uuid4()}.{ext}'

        files_per_lang[lang] += 1
        output_items.append({
            'language': lang,
            'repository_language': repository_language,
            'repository_dirname': repository_dirname,
            'filename': filename,
            'dedup_key': dedup_key,
            'extract_to': extract_to_filename,
            'rank': files_per_lang[lang],
        })
    return output_items


def _list_compressed_files(repository_dirname: str) -> List[Tuple[str, str]]:
    repository_path = Config.repositories_dir.joinpath(repository_dirname)

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
        _, _, dedup, filename = info.strip().split(maxsplit=3)
        dedup_key = f'0x{dedup}'
        compressed_files.append((filename, dedup_key))
    return compressed_files


def _analyse_languages(
    languages: Dict[str, List[str]]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    key = itemgetter(0)
    couples = ((ext, lang) for lang, exts in languages.items() for ext in exts)
    groups = groupby(sorted(couples, key=key), key=key)
    ext_lang_data = ((ext, [lang for _, lang in info]) for ext, info in groups)
    ext_languages = dict(ext_lang_data)

    ambiguous = {}
    ext_lang = {}
    for ext, langs in ext_languages.items():
        if len(ext_languages[ext]) == 1:
            ext_lang[ext] = langs[0]
        elif len(ext_languages[ext]) > 1:
            ambiguous[ext] = langs

    if ambiguous:
        LOGGER.warning(f'Ambiguous extensions found: {ambiguous}')

    for ext, langs in ambiguous.items():
        for lang in langs:
            if languages[lang][0] == lang:
                raise RuntimeError(
                    f'Ambiguous extension {ext} cannot be the first'
                    'file extension defined for language {lang}')

    return ext_lang, ambiguous


def _find_language(
    filename: str,
    ext_lang: Dict[str, str],
    ambiguous: Dict[str, List[str]],
    repository_language: str,
) -> Optional[str]:
    if filename.endswith('/'):
        return None

    ext = Path(filename).suffix.lstrip('.').lower()
    if ext in ext_lang:
        return ext_lang[ext]

    elif repository_language in ambiguous.get(ext, []):
        return repository_language

    return None


@cached(File.FILES_SPLIT_BY_USAGE)
def split() -> None:
    LOGGER.info('Split repositories by usage: train, valid & test')
    LOGGER.info('This operation should take few seconds...')

    files = load_csv(File.AVAILABLE_FILES)
    files = files.drop('dedup_key', axis=1)
    repo_columns = ['repository_language', 'repository_dirname']

    repo = files[repo_columns].drop_duplicates()
    repo = repo.sample(frac=1).reset_index(drop=True)
    repo.loc[:, 'usage'] = ''

    LOGGER.info(f'Total downloaded repositories: {len(repo)}')

    total_files = (
        Config.nb_train_files_per_language
        + Config.nb_valid_files_per_language
        + Config.nb_test_files_per_language
    )
    valid_ratio = Config.nb_valid_files_per_language / total_files
    valid_ratio = max(valid_ratio, MIN_SPLIT_RATIO)

    test_ratio = Config.nb_test_files_per_language / total_files
    test_ratio = max(test_ratio, MIN_SPLIT_RATIO)

    repositories = {}
    for language in Config.languages:
        by_language = repo[repo['repository_language'] == language]
        total = len(by_language)
        if total < MIN_REPOSITORIES:
            raise RuntimeError(
                f'Need more than {MIN_REPOSITORIES}, '
                f'only {total} repositories usable for language {language}'
            )

        nb_test = max(int(total*test_ratio), 1)
        nb_valid = max(int(total*valid_ratio), 1)
        nb_test_valid = nb_test + nb_valid

        test = by_language[:nb_test]
        test['usage'].values[:] = 'test'
        repositories[f'{language}/test'] = test

        valid = by_language[nb_test:nb_test_valid]
        valid['usage'].values[:] = 'valid'
        repositories[f'{language}/valid'] = valid

        train = by_language[nb_test_valid:]
        train['usage'].values[:] = 'train'
        repositories[f'{language}/train'] = train

        LOGGER.info(
            f'{language} nb repositories, train: {total-nb_test_valid}, '
            f'valid: {nb_valid}, test: {nb_test}'
        )

    for name, repository in repositories.items():
        if not len(repository):
            LOGGER.error(f'No repositories available for {name}')
            raise RuntimeError(f'No repositories for category: {name}')

    repo = pd.concat(repositories.values())
    files = pd.merge(files, repo, on=repo_columns)
    save_csv(files, File.FILES_SPLIT_BY_USAGE)


def extract() -> None:
    LOGGER.info('Extract selected files')
    LOGGER.info('This operation might take a lot of time...')

    train_path = Config.extracted_files_dir.joinpath('train')
    valid_path = Config.extracted_files_dir.joinpath('valid')
    test_path = Config.extracted_files_dir.joinpath('test')

    train_path.mkdir(exist_ok=True)
    valid_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    # Load list of files to extract
    source = load_csv(File.FILES_SPLIT_BY_USAGE)

    # Load list of processed files
    try:
        files = load_csv(File.EXTRACTED_FILES)
    except IOError:
        files = pd.DataFrame([], columns=EXTRACTED_FILES_COLUMNS)

    df = pd.merge(source, files, how='outer', on=list(source.columns))
    df.loc[df['status'].isnull(), 'status'] = Status.PENDING.value

    # Flag existing files
    is_pending = df['status'] == Status.PENDING.value
    file_exists = df.apply(_destination_exists, axis=1)
    df.loc[(is_pending & file_exists), 'status'] = Status.DISCARDED.value

    while True:
        selected = _choose_files_to_extract(df)
        LOGGER.info(f'{len(selected)} files to extract')

        if not len(selected):
            break

        result = _extract_files(selected)

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

    save_csv(df, File.EXTRACTED_FILES)

    LOGGER.info(f'The training files are located in {train_path}')
    LOGGER.info(f'The validation files are located in {valid_path}')
    LOGGER.info(f'The test files are located in {test_path}')


def _destination_exists(item: Dict[str, str]):
    usage = item['usage']
    extract_to = item['extract_to']
    return Config.extracted_files_dir.joinpath(usage, extract_to).exists()


def _choose_files_to_extract(df: pd.DataFrame) -> pd.DataFrame:
    usage_info = {
        'train': Config.nb_train_files_per_language,
        'valid': Config.nb_valid_files_per_language,
        'test': Config.nb_test_files_per_language,
    }

    files = []
    mask_pending = df['status'] == Status.PENDING.value
    mask_extracted = df['status'] == Status.EXTRACTED.value

    for lang in Config.languages:
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
                    f'{lang}/{usage} minimum required: {nb_files_to_keep} files'
                )

    chosen = pd.concat(files)
    return chosen


def _extract_files(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('repository_dirname')
    nb_groups = len(grouped)
    simple_groups = (
        (dirname, [dict(repo_info) for _, repo_info in items.iterrows()])
        for dirname, items in grouped
    )

    results = []
    pool = pool_map(_extract_from_repository, simple_groups, multiplier=2)
    for index, grouped_results in enumerate(pool, 1):
        results.append(grouped_results)
        if index % Config.step == 0:
            LOGGER.info(f'--> Processed {index} / {nb_groups} repositories...')
    LOGGER.info(f'--> Processed {nb_groups} / {nb_groups} repositories!')

    flattened = (file_info for result in results for file_info in result)
    final_result = pd.DataFrame(flattened)
    return final_result


def _extract_from_repository(
    params: Tuple[str, List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    repository_dirname, items = params
    repository_path = Config.repositories_dir.joinpath(repository_dirname)

    result = run(GIT_DISABLE_GC, stdout=PIPE, stderr=PIPE, cwd=repository_path)
    if result.returncode != 0:
        LOGGER.debug(f'Failed to disable GC in {repository_path}')

    filenames = set(item['filename'] for item in items)
    command = GIT_RESET_FILES + list(filenames)
    result = run(command, stdout=PIPE, stderr=PIPE, cwd=repository_path)
    if result.returncode != 0:
        LOGGER.debug(f'Failed to reset files from {repository_path}')

    return [_move_file(repository_path, item) for item in items]


def _move_file(repository_path: Path, item: Dict[str, str]) -> Dict[str, Any]:
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

    destination = Config.extracted_files_dir.joinpath(usage, basename)
    destination.write_text(text)
    return ok
