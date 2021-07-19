import logging
from subprocess import run, PIPE
from typing import Dict, Any

import pandas as pd

from guesslangtools.common import File, Config, cached, pool_map, LOG_STEP


LOGGER = logging.getLogger(__name__)

# Using "none" as credentials to generate an authentication error
# when the repository is not accessible
REPOSITORY_DOWNLOAD_URL = 'https://none:none@github.com/{}/{}.git'
REPOSITORY_BASENAME = '{}___{}'

GIT_CLONE_ERROR = b'Authentication failed'
GIT_CLONE_TIMEOUT = 10
GIT_CLONE_COMMAND = [
    'timeout',
    str(GIT_CLONE_TIMEOUT),
    'git',
    'clone',
    '--no-checkout',
    '--filter=blob:none',
    '--depth=1'
]


@cached(File.SELECTED_REPOSITORIES)
def select(config: Config) -> None:
    LOGGER.info('Choose repositories per language')
    LOGGER.info('This operation might take several minutes...')

    input_data = config.load_csv(File.ALTERED_DATASET)
    shuffled = input_data.sample(frac=1).reset_index(drop=True)

    max_repositories = config.nb_repositories_per_language

    selected_list = []
    for lang in config.languages:
        filtered = shuffled[shuffled['repository_language'] == lang]
        nb_found = len(filtered)
        nb_selected = min(nb_found, max_repositories)

        LOGGER.info(
            f'{lang} repositories, found: {nb_found}, kept: {nb_selected}'
        )

        if nb_selected < max_repositories:
            LOGGER.warning(
                f'{lang}, not enough repositories, '
                f'required: {max_repositories}'
            )

        if nb_selected == 0:
            continue

        selected = filtered[:nb_selected]
        selected_list.append(selected)

    if not selected_list:
        LOGGER.error('No repository found')
        raise RuntimeError('No repository found')

    output_path = config.absolute(File.SELECTED_REPOSITORIES)
    united = pd.concat(selected_list)
    united.to_csv(output_path, index=False)


@cached(File.PREPARED_REPOSITORIES)
def prepare(config: Config) -> None:
    LOGGER.info('Prepare repositories download')
    LOGGER.info('This operation should take few seconds...')

    input_data = config.load_csv(File.SELECTED_REPOSITORIES)
    input_data.loc[:, 'repository_dirname'] = ''
    input_data.loc[:, 'repository_url'] = ''

    output_data = input_data.apply(_add_download_info, axis=1)
    output_path = config.absolute(File.PREPARED_REPOSITORIES)
    output_data.to_csv(output_path, index=False)


def _add_download_info(item: Dict[str, str]) -> Dict[str, str]:
    user, project = item['repository_name'].split('/')
    dirname = REPOSITORY_BASENAME.format(user, project)

    item['repository_url'] = REPOSITORY_DOWNLOAD_URL.format(user, project)
    item['repository_dirname'] = dirname
    return item


@cached(File.DOWNLOADED_REPOSITORIES)
def download(config: Config) -> None:
    LOGGER.info('Download chosen repositories')
    LOGGER.info('This operation might take a lot of time...')

    input_data = config.load_csv(File.PREPARED_REPOSITORIES)

    input_data.loc[:, 'repository_is_empty'] = True
    rows = (dict(row) for _, row in input_data.iterrows())
    result_rows = []
    total = len(input_data)
    for step, row in enumerate(pool_map(_clone_repository, rows, config), 1):
        result_rows.append(row)
        if step % LOG_STEP == 0:
            LOGGER.info(f'--> Processed {step} / {total} repositories...')
    LOGGER.info(f'--> Processed {total} / {total} repositories!')

    data = pd.DataFrame(result_rows)

    LOGGER.info('Removing empty repositories')
    data = data[~data['repository_is_empty']]
    LOGGER.info(f'Kept {len(data)} non empty repositories')

    fieldnames = ['repository_language', 'repository_dirname']
    output_data = data[fieldnames]
    output_path = config.absolute(File.DOWNLOADED_REPOSITORIES)
    output_data.to_csv(output_path, index=False)


def _clone_repository(item: Dict[str, Any], config: Config) -> Dict[str, str]:
    url = item['repository_url']
    path = config.repositories_dir.joinpath(item['repository_dirname'])

    if not path.exists():
        LOGGER.debug(f'Downloading {url}')
        command = GIT_CLONE_COMMAND + [url, str(path)]
        result = run(command, stdout=PIPE, stderr=PIPE)
        if result.returncode != 0 or GIT_CLONE_ERROR in result.stdout:
            path.mkdir(exist_ok=True)

    item['repository_is_empty'] = not any(path.iterdir())
    return item
