from contextlib import suppress
from functools import wraps
from http.client import IncompleteRead
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
from yaml import safe_load


LOGGER = logging.getLogger(__name__)

Function = TypeVar('Function', bound=Callable[..., Any])

LANGUAGES_FILENAME = 'languages.yaml'
CHUNK_SIZE = 1024
TIMEOUT = 30
CSV_FIELD_LIMIT = 10 * 1024 * 1024  # 1O MiB
MAX_FILES_PER_REPOSITORY_PER_LANGUAGE = 1000
LOG_STEP = 100


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
    DEDUPLICATED_FILES = '09_deduplicated_files.csv'
    FILES_SPLIT_BY_USAGE = '10_files_split_by_usage.csv'
    EXTRACTED_FILES = '11_extracted_files.csv'


class Config:
    """Runtime configuration."""

    def __init__(
        self,
        cache_dir: str,
        nb_repositories: int,
        nb_train: int,
        nb_valid: int,
        nb_test: int,
    ) -> None:
        """Setup configuration."""
        self.bypass_cache = False
        self.nb_train_files_per_language = nb_train
        self.nb_valid_files_per_language = nb_valid
        self.nb_test_files_per_language = nb_test
        self.nb_repositories_per_language = nb_repositories
        self.cache_path = Path(cache_dir).absolute()
        self.repositories_dir = self.absolute('repositories')
        self.extracted_files_dir = self.absolute('files')

        self.cache_path.mkdir(exist_ok=True)
        self.repositories_dir.mkdir(exist_ok=True)
        self.extracted_files_dir.mkdir(exist_ok=True)

        root_path = Path(__file__).parent
        languages_path = root_path.joinpath('data', LANGUAGES_FILENAME)
        content = languages_path.read_text()
        language_info = safe_load(content)
        self.languages = list(language_info)
        self.alias_mapping = self._map_values(language_info, 'aliases', False)
        self.file_mapping = self._map_values(language_info, 'files')
        self.ext_mapping = self._map_values(language_info, 'extensions')

        self.extensions = {}
        for lang, info in language_info.items():
            ext = info['extensions'][0]
            languages = self.ext_mapping[ext]
            if len(languages) > 1:
                raise RuntimeError(
                    f'"{ext}" is used by multiple languages {languages}. '
                    f'Please change the first extension of {lang}'
                )
            self.extensions[lang] = ext

    def absolute(self, *path_parts: str) -> Path:
        """Create an absolute path."""
        return self.cache_path.joinpath(*path_parts).absolute()

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV file."""
        fullname = self.absolute(filename)
        return pd.read_csv(fullname)

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame to a CSV file."""
        fullname = self.absolute(filename)
        df.to_csv(fullname, index=False)

    def backup(self, filename: str) -> None:
        """Create a backup of a given file"""
        current_path = self.absolute(filename)
        backup_path = self.absolute(f'{filename}.bkp')

        with suppress(IOError):
            backup_path.unlink()

        current_path.replace(backup_path)
        LOGGER.info(f'Backup: {current_path} to {backup_path}')

    @staticmethod
    def remove_from_cache(path: Path) -> None:
        if path.is_file():
            path.unlink()
            LOGGER.info(f'Removed cache file: {path}')

    @staticmethod
    def _map_values(
        language_info: Dict[str, Dict[str, List[str]]],
        fieldname: str,
        duplicates_ok: bool = True,
    ) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for lang, info in language_info.items():
            for value in info[fieldname]:
                result.setdefault(value, []).append(lang)

        # Check mapping
        for value, languages in result.items():
            if len(languages) > 1:
                message = (
                    f'Checking {fieldname}: "{value}" is associated with '
                    f'more than one language: {languages}'
                )
                if duplicates_ok:
                    LOGGER.warning(message)
                else:
                    raise RuntimeError(message)

        return result


def cached(location: str) -> Callable[[Function], Function]:
    """Decorator: run a function only if the cache file doesn't exist."""

    def wrapper(func: Function) -> Function:

        @wraps(func)
        def wrapped(config: Config, *args: Any, **kw: Any) -> Any:
            path = config.absolute(location)
            if config.bypass_cache:
                config.remove_from_cache(path)

            if path.exists():
                LOGGER.info(f'Found in the cache: {path}')
                return

            config.bypass_cache = True
            try:
                result = func(config, *args, **kw)
                if path.exists():
                    LOGGER.info(f'Created cache file: {path}')
                return result
            except (Exception, KeyboardInterrupt):
                config.remove_from_cache(path)
                raise

        return cast(Function, wrapped)

    return wrapper


def requires(location: str) -> Callable[[Function], Function]:
    """Decorator: run a function only if the cache file doesn't exist."""

    def wrapper(func: Function) -> Function:

        @wraps(func)
        def wrapped(config: Config, *args: Any, **kw: Any) -> Any:

            path = config.absolute(location)
            if not path.exists():
                LOGGER.error(f'Cache file missing: {path}')
                raise RuntimeError(f'Requires cache file {path}')

            LOGGER.info(f'Found in the cache: {path}')
            result = func(config, *args, **kw)
            return result

        return cast(Function, wrapped)

    return wrapper


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
