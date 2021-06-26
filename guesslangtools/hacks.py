from contextlib import suppress
import logging
import time
from typing import List

import pandas as pd
import requests

from guesslangtools.common import (
    absolute, File, Config, requires, load_csv, save_csv, backup
)


LOGGER = logging.getLogger(__name__)

DATASET_BASENAME = 'alt_dataset.csv'

GITHUB_API_URL = 'https://api.github.com/search/repositories'
GITHUB_DELAY = 2.1  # GITHUB API rate limit is 30 request/min
MAX_PAGES = 100
PER_PAGE = 100


@requires(File.SELECTED_REPOSITORIES)
def show_repositories_distribution() -> None:
    LOGGER.info('Loading repositories info')
    LOGGER.info('This operation should take few seconds...')

    selected = load_csv(File.SELECTED_REPOSITORIES)
    count = selected.repository_language.value_counts()

    pd.set_option('display.max_rows', None)
    print(count)


@requires(File.ALTERED_DATASET)
@requires(File.SELECTED_REPOSITORIES)
def select_more_repositories(languages: List[str]) -> None:
    LOGGER.info('Choose more repositories per language')
    LOGGER.info('This operation might take several minutes...')

    output_path = absolute(File.SELECTED_REPOSITORIES)

    input_data = load_csv(File.ALTERED_DATASET)
    known = load_csv(File.SELECTED_REPOSITORIES)

    mask = ~input_data['repository_name'].isin(known['repository_name'])
    repositories = input_data[mask]
    shuffled = repositories.sample(frac=1).reset_index(drop=True)

    max_repositories = Config.nb_repositories_per_language

    selected_list = []
    for language in languages:
        if language not in Config.languages:
            LOGGER.error(f'Unknown language {language}')
            raise RuntimeError(f'Unknown language {language}')

        pending = shuffled[shuffled['repository_language'] == language]
        nb_known = len(known[known['repository_language'] == language])
        nb_pending = len(pending)
        nb_required = max(max_repositories-nb_known, 0)
        nb_selected = min(nb_pending, nb_required)
        total = nb_known + nb_selected

        LOGGER.info(
            f'{language}: repositories per language: {max_repositories}, '
            f'pending: {nb_pending}, known: {nb_known}, '
            f'selected: {nb_selected}, total: {total}'
        )

        if total < max_repositories:
            LOGGER.warning(
                f'{language}, not enough repositories, '
                f'required: {max_repositories}'
            )

        if nb_selected == 0:
            continue

        selected = pending[:nb_selected]
        selected_list.append(selected)

    if not selected_list:
        LOGGER.error('No repository found')
        raise RuntimeError('No repository found')

    backup(File.SELECTED_REPOSITORIES)
    with suppress(IOError):
        backup(File.PREPARED_REPOSITORIES)

    new_repositories = pd.concat(selected_list)
    united = known.append(new_repositories)
    united.to_csv(output_path, index=False)


@requires(File.SELECTED_REPOSITORIES)
@requires(File.PREPARED_REPOSITORIES)
def select_only_downloaded_repo() -> None:
    downloaded_repo = (path.name for path in Config.repositories_dir.glob('*'))
    selected = load_csv(File.SELECTED_REPOSITORIES)
    prepared = load_csv(File.PREPARED_REPOSITORIES)

    LOGGER.info(f'{len(selected)} repositories previously selected')

    repo = pd.DataFrame(downloaded_repo, columns=['repository_filename'])
    mask = prepared['repository_filename'].isin(repo['repository_filename'])
    prepared = prepared[mask]
    mask = selected['repository_name'].isin(prepared['repository_name'])
    selected = selected[mask]

    LOGGER.info(f'{len(selected)} downloaded repositories selected')

    backup(File.SELECTED_REPOSITORIES)
    backup(File.PREPARED_REPOSITORIES)
    save_csv(selected, File.SELECTED_REPOSITORIES)
    save_csv(prepared, File.PREPARED_REPOSITORIES)


@requires(File.SELECTED_REPOSITORIES)
def merge_to_selected_repositories(filename: str) -> None:
    selected = load_csv(File.SELECTED_REPOSITORIES)
    listed = load_csv(filename)

    selected = pd.concat([listed, selected])
    selected = selected.drop_duplicates('repository_name')

    backup(File.SELECTED_REPOSITORIES)
    save_csv(selected, File.SELECTED_REPOSITORIES)
    with suppress(IOError):
        backup(File.PREPARED_REPOSITORIES)


def download_github_repo_list(
    token: str, language: str, filename: str
) -> None:
    LOGGER.info(f'Listing repositories for language {language}')
    with open(filename, 'w') as output:
        output.write('repository_name,repository_language\n')

        known_repos = set()
        for page in range(1, MAX_PAGES+1):
            LOGGER.info(f'Processing page {page: 3} / {MAX_PAGES}')
            url = (
                f'{GITHUB_API_URL}?access_token={token}&q=language:{language}'
                f'&per_page={PER_PAGE}&page={page}&sort=updated&order=desc'
            )
            response = requests.get(url)
            items = response.json().get('items', [])
            if not response.ok or not items:
                LOGGER.info('No more repositories to retrieve')
                break

            for repo in items:
                repo_name = repo['full_name']
                repo_id = repo['id']
                if repo_id in known_repos:
                    print(repo_id)
                    continue

                known_repos.add(repo_id)
                output.write(f'{repo_name},{language}\n')

            time.sleep(GITHUB_DELAY)
    LOGGER.info(f'{len(known_repos)} repositories saved in {filename}')
