from contextlib import suppress
from functools import wraps
from http.client import IncompleteRead
import json
import logging
from multiprocessing import get_context, cpu_count
from pathlib import Path
import signal
from ssl import SSLError
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Callable,
    Iterator,
    Iterable,
    TypeVar,
    Optional,
    cast,
)

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)
Function = TypeVar('Function', bound=Callable[..., Any])

NULL_PATH = Path('/dev/null')
LANGUAGES_FILENAME = 'languages.json'
CHUNK_SIZE = 1024
TIMEOUT = 30


class File:
    """Cache files."""
    COMPRESSED_DATASET = '01_repositories_dataset.tar.gz'
    DATASET = '02_repositories_dataset.csv'
    SHRUNK_DATASET = '03_shrunk_repositories_dataset.csv'
    ALTERED_DATASET = '04_altered_repositories_dataset.csv'

    SELECTED_REPOSITORIES = '05_selected_repositories.csv'
    PREPARED_REPOSITORIES = '06_prepare_repositories_to_download.csv'
    DOWNLOADED_REPOSITORIES = '07_downloaded_repositories.csv'

    AVAILABLE_FILES = '08_available_files.csv'
    FILES_SPLIT_BY_USAGE = '09_files_split_by_usage.csv'
    EXTRACTED_FILES = '10_extracted_files.csv'


class Config:
    """Runtime configuration."""
    nb_train_files_per_language = 0
    nb_valid_files_per_language = 0
    nb_test_files_per_language = 0
    nb_repositories_per_language = 0
    cache_dir = ''

    max_files_per_repository_per_language = 1000
    bypass_cache = False
    languages: Dict[str, List[str]] = {}
    step = 100
    repositories_dir = NULL_PATH
    extracted_files_dir = NULL_PATH

    @classmethod
    def setup(
        cls,
        cache_dir: str,
        nb_repositories: int,
        nb_train: int,
        nb_valid: int,
        nb_test: int,
    ) -> None:
        """Set configuration."""
        cls.nb_train_files_per_language = nb_train
        cls.nb_valid_files_per_language = nb_valid
        cls.nb_test_files_per_language = nb_test
        cls.nb_repositories_per_language = nb_repositories
        cls.cache_dir = cache_dir
        cls.repositories_dir = absolute('repositories')
        cls.extracted_files_dir = absolute('files')

        Path(cls.cache_dir).mkdir(exist_ok=True)
        Path(cls.repositories_dir).mkdir(exist_ok=True)
        Path(cls.extracted_files_dir).mkdir(exist_ok=True)

        root_path = Path(__file__).parent
        languages_path = root_path.joinpath('data', LANGUAGES_FILENAME)
        with languages_path.open() as languages_file:
            cls.languages = json.load(languages_file)


def absolute(*path_parts: str) -> Path:
    """Create an absolute path."""
    return Path(Config.cache_dir, *path_parts).absolute()


def cached(location: str) -> Callable[[Function], Function]:
    """Decorator: run a function only if the cache file doesn't exist."""

    def wrapper(func: Function) -> Function:

        @wraps(func)
        def wrapped(*args: Any, **kw: Any) -> Any:

            if not Config.cache_dir:
                raise RuntimeError('Cache directory not set')

            path = absolute(location)
            if Config.bypass_cache:
                _remove_from_cache(path)

            if path.exists():
                LOGGER.info(f'Found in the cache: {path}')
                return

            Config.bypass_cache = True
            try:
                result = func(*args, **kw)
                LOGGER.info(f'Created cache file: {path}')
                return result
            except (Exception, KeyboardInterrupt):
                _remove_from_cache(path)
                raise

        return cast(Function, wrapped)

    return wrapper


def requires(location: str) -> Callable[[Function], Function]:
    """Decorator: run a function only if the cache file doesn't exist."""

    def wrapper(func: Function) -> Function:

        @wraps(func)
        def wrapped(*args: Any, **kw: Any) -> Any:

            path = absolute(location)
            if not path.exists():
                LOGGER.error(f'Cache file missing: {path}')
                raise RuntimeError(f'Requires cache file {path}')

            LOGGER.info(f'Found in the cache: {path}')
            result = func(*args, **kw)
            return result

        return cast(Function, wrapped)

    return wrapper


def _remove_from_cache(path: Path) -> None:
    if path.is_file():
        path.unlink()
        LOGGER.info(f'Removed cache file: {path}')


def download_file(url: str, destination: Path) -> Tuple[bool, int]:
    """Download a file."""

    response = requests.get(url, stream=True, timeout=TIMEOUT)
    if not response.ok:
        LOGGER.warning(f'Cannot download {url}: {response.status_code}')
        return False, response.status_code

    try:
        with destination.open('wb') as repo_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                repo_file.write(chunk)
    except (IncompleteRead, SSLError) as error:
        LOGGER.warning(f'Download cancelled {url}: {error}')
        _remove_if_possible(destination)
        return False, -2
    except requests.RequestException as error:
        LOGGER.warning(f'Download failed  {url}: {error}')
        _remove_if_possible(destination)
        return False, -1
    except (Exception, KeyboardInterrupt):
        _remove_if_possible(destination)
        raise

    return True, response.status_code


def _remove_if_possible(path: Path) -> None:
    with suppress(IOError):
        path.unlink()


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file."""

    fullname = absolute(filename)
    return pd.read_csv(fullname)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to a CSV file."""

    fullname = absolute(filename)
    df.to_csv(fullname, index=False)


def pool_map(
    method: Function,
    items: Iterable[Any],
    *method_args: Any,
    multiplier: Optional[int] = None,
    **method_kw: Any,
) -> Iterator[Any]:
    """Run a function with multiprocessing."""

    processes = multiplier * cpu_count() if multiplier else None
    iterable = ((method, item, method_args, method_kw) for item in items)
    context = get_context('spawn')
    with context.Pool(processes, initializer=_initializer) as pool:
        for result in pool.imap_unordered(_apply, iterable):
            yield result


def _apply(
    composite: Tuple[Function, Any, Tuple[Any, ...], Dict[str, Any]]
) -> Any:
    """Generic function wrapper"""
    method, item, other_args, keywords = composite
    return method(item, *other_args, **keywords)


def _initializer() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def backup(filename: str) -> None:
    """Create a backup of a given file"""

    current_path = absolute(filename)
    backup_path = absolute(f'{filename}.bkp')

    with suppress(IOError):
        backup_path.unlink()

    current_path.replace(backup_path)
    LOGGER.info(f'Backup: {current_path} to {backup_path}')
