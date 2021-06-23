from contextlib import suppress
import logging
from typing import List

import pandas as pd

from guesslangtools.common import (
    absolute, File, Config, requires, load_csv, save_csv, backup
)


LOGGER = logging.getLogger(__name__)

DATASET_BASENAME = 'alt_dataset.csv'


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
    LOGGER.info('This operation might take few minutes...')

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
