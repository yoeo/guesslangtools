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
    Config, File, cached, load_csv, save_csv, pool_imap
)


LOGGER = logging.getLogger(__name__)

CRC_BITS = 32
MIN_ENCODING_CONFIDENCE = 0.5
MIN_SPLIT_RATIO = 0.15

GIT_LIST_FILES_COMMAND = ['git', 'ls-tree', '-r', 'HEAD']
GIT_RESET_FILES = ['git', 'checkout', 'HEAD']

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


# (re-)Generates File.AVAILABLE_FILES
@cached(File.FILES_SPLIT_BY_USAGE)
def list_all() -> None:
    LOGGER.info('List source files from repositories')
    LOGGER.info('This operation might take few minutes...')

    columns = [
        'extract_to', 'filename', 'language', 'rank', 'repository_filename',
        'dedup_key',
    ]

    repo = load_csv(File.DOWNLOADED_REPOSITORIES)
    try:
        files = load_csv(File.AVAILABLE_FILES)
    except IOError:
        files = pd.DataFrame([], columns=columns)

    mask = ~repo['repository_filename'].isin(files['repository_filename'])
    new_repo = repo[mask]
    LOGGER.info('%s newly downloaded repositories', len(new_repo))

    nb_repo_before = len(files.repository_filename.unique())
    mask = files['repository_filename'].isin(repo['repository_filename'])
    files = files[mask]
    nb_repo_after = len(files.repository_filename.unique())
    nb_removed = nb_repo_before - nb_repo_after
    LOGGER.info('%s deleted repositories', nb_removed)

    new_files = _list_files_by_language(new_repo)
    df = pd.concat([files, new_files], axis=0, sort=False)

    df.drop_duplicates(subset='dedup_key', inplace=True)
    df.sort_values(by='rank', inplace=True)

    LOGGER.info('Files available by language:')
    for language in Config.languages:
        nb_files = len(df[df['language'] == language])
        LOGGER.info('--> %s: %s', language, nb_files)

    save_csv(df, File.AVAILABLE_FILES)


def _list_files_by_language(repo: pd.DataFrame) -> pd.DataFrame:
    languages = Config.languages
    nb_files_limit = Config.max_files_per_repository_per_language
    ext_lang, ambiguous = _analyse_languages(languages)

    results = []
    rows = (item for _, item in repo.iterrows())
    args = (languages, ext_lang, ambiguous, nb_files_limit)
    for index, result in enumerate(pool_imap(_select_files, rows, *args)):
        if result:
            results.append(result)

        if index % Config.step == 0:
            LOGGER.info('--> Processed %s repositories...', index)

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
    repository_filename = item['repository_filename']

    files = _list_compressed_files(repository_filename)
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
            'repository_filename': repository_filename,
            'filename': filename,
            'dedup_key': dedup_key,
            'extract_to': extract_to_filename,
            'rank': files_per_lang[lang],
        })
    return output_items


def _list_compressed_files(repository_filename: str) -> List[Tuple[str, str]]:
    compressed_files = []
    zip_filename = Config.repositories_dir.joinpath(repository_filename)
    with suppress(CalledProcessError):
        result = check_output(GIT_LIST_FILES_COMMAND, cwd=zip_filename)
        for info in result.decode().strip().split('\n'):
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
        LOGGER.warning('Ambiguous extensions found: %s', ambiguous)

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
    columns = ['repository_language', 'repository_filename']

    repo = files[columns].drop_duplicates()
    repo = repo.sample(frac=1).reset_index(drop=True)
    repo.loc[:, 'usage'] = ''

    LOGGER.info('Total downloaded repositories: %s', len(repo))

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
        if total < 3:
            raise RuntimeError(
                f'Need more than 3 repositories for language {language}')

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
            '%s nb repositories, train: %s, valid: %s, test: %s',
            language, total-nb_test_valid, nb_valid, nb_test)

    for name, repository in repositories.items():
        if not len(repository):
            LOGGER.error('No repositories available for %s', name)
            raise RuntimeError(f'No repositories for category: {name}')

    repo = pd.concat(repositories.values())
    files = pd.merge(files, repo, on=columns)
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

    source = load_csv(File.FILES_SPLIT_BY_USAGE)
    columns = [
        'extract_to', 'filename', 'language', 'rank', 'repository_filename',
        'repository_language', 'usage', 'status']
    try:
        files = load_csv(File.EXTRACTED_FILES)
    except IOError:
        files = pd.DataFrame([], columns=columns)

    df = pd.merge(source, files, how='outer', on=list(source.columns))
    df.loc[df['status'].isnull(), 'status'] = Status.PENDING.value

    while True:
        selected = _choose_files_to_extract(df)
        LOGGER.info('%s files to extract', len(selected))

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
            'Processed %s files: %s extracted, %s discarded',
            len(result), len(result_extracted), len(result_discarded))

        LOGGER.info('%s total files extracted', len(extracted))
        LOGGER.info('%s total files discarded', len(discarded))

    save_csv(df, File.EXTRACTED_FILES)

    LOGGER.info('The training files are located in %s', train_path)
    LOGGER.info('The validation files are located in %s', valid_path)
    LOGGER.info('The test files are located in %s', test_path)


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
                '%s/%s, pending: %s files,  kept: %s files',
                lang, usage, len(pending), len(kept))

            if len(kept) < nb_files_to_keep:
                LOGGER.warning(
                    '%s/%s minimum required: %s files',
                    lang, usage, nb_files_to_keep)

    chosen = pd.concat(files)
    return chosen


def _extract_files(input_data: pd.DataFrame) -> pd.DataFrame:
    results = []
    grouped = input_data.groupby('repository_filename')

    pool = pool_imap(_extract_from_repository, grouped)
    for index, grouped_results in enumerate(pool, 1):
        results.append(grouped_results)
        if index % Config.step == 0:
            LOGGER.info('--> Processed %s repositories...', index)

    flattened = (file_info for result in results for file_info in result)
    final_result = pd.DataFrame(flattened)
    return final_result


def _extract_from_repository(
    grouped_args: Tuple[str, pd.DataFrame],
) -> pd.DataFrame:
    repository_filename, items = grouped_args
    zip_filename = Config.repositories_dir.joinpath(repository_filename)

    filenames = set(items['filename'])
    command = GIT_RESET_FILES + list(filenames)
    result = run(command, stdout=PIPE, cwd=zip_filename)
    if result.returncode != 0:
        LOGGER.debug(f'Failed to reset files from {zip_filename}')

    def extract_file(item: Dict[str, str]) -> Dict[str, Any]:
        usage = item['usage']
        filename = item['filename']
        basename = item['extract_to']
        destination = Config.extracted_files_dir.joinpath(usage, basename)

        ko = {'extract_to': basename, 'status': Status.DISCARDED.value}
        ok = {'extract_to': basename, 'status': Status.EXTRACTED.value}

        if destination.exists():
            LOGGER.debug('File already extracted %s', destination)
            return ko

        source = zip_filename.joinpath(filename)
        content = source.read_bytes()
        source.unlink()

        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            LOGGER.debug('Non UTF-8 text %s: %s', zip_filename, filename)
            detected = chardet.detect(content)
            if detected['confidence'] < MIN_ENCODING_CONFIDENCE:
                LOGGER.debug('Bad Encoding %s: %s', zip_filename, filename)
                return ko

            try:
                text = content.decode(detected['encoding'])
            except (UnicodeDecodeError, LookupError):
                LOGGER.debug('Bad Encoding %s: %s', zip_filename, filename)
                return ko

        destination.write_text(text)
        return ok

    result = items.apply(extract_file, axis=1)
    return result
