import csv
import logging
from pathlib import Path
import tarfile
from typing import Dict

import pandas as pd

from guesslangtools.common import (
    absolute, File, cached, download_file, load_csv
)


LOGGER = logging.getLogger(__name__)

# Open source projects dataset: https://zenodo.org/record/3626071/
DATASET_FILENAME = 'repositories-1.2.0-2018-03-12.csv'
DATASET_URL = (
    'https://zenodo.org/record/3626071/files/'
    'libraries-1.6.0-2020-01-12.tar.gz?download=1'
)
CSV_FIELD_LIMIT = 10 * 1024 * 1024  # 1O MiB

PKG_ROOT = Path(__file__).parent.parent
SQL_DATASET_PATH = PKG_ROOT.joinpath('data', 'sql_dataset.csv')


@cached(File.COMPRESSED_DATASET)
def download() -> None:
    LOGGER.info('Retrieving repositories dataset (8GB)')
    LOGGER.info('This operation might take a lot of time...')

    destination = absolute(File.COMPRESSED_DATASET)
    download_file(DATASET_URL, destination)


@cached(File.DATASET)
def extract() -> None:
    LOGGER.info('Extracting repositories list file')
    LOGGER.info('This operation might take few minutes...')

    compressed_filename = absolute(File.COMPRESSED_DATASET)
    with tarfile.open(compressed_filename) as tar:
        tar.extract(DATASET_FILENAME, path=absolute('.'))

    extracted_file = absolute(DATASET_FILENAME)
    extracted_file.rename(absolute(File.DATASET))


@cached(File.SHRUNK_DATASET)
def shrink() -> None:
    LOGGER.info('Shrink repositories list file')
    LOGGER.info('This operation might take few minutes...')

    input_path = absolute(File.DATASET)
    output_path = absolute(File.SHRUNK_DATASET)

    # The input dataset is too huge to be fully loaded into memory
    csv.field_size_limit(CSV_FIELD_LIMIT)
    with input_path.open() as input_file, output_path.open('w') as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = ['repository_name', 'repository_language']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in reader:
            if _ignore(item):
                continue

            smaller_item = {
                'repository_name': item['Name with Owner'],
                'repository_language': item['Language'],
            }
            writer.writerow(smaller_item)


def _ignore(item: Dict[str, str]) -> bool:
    return (
        item['Fork'] == 'true'
        or item['Host Type'] != 'GitHub'
        or not item['Name with Owner']
    )


@cached(File.ALTERED_DATASET)
def alter() -> None:
    LOGGER.info('Alter repositories list file')
    LOGGER.info('This operation might take few minutes...')

    output_path = absolute(File.ALTERED_DATASET)

    df = load_csv(File.SHRUNK_DATASET)

    # Set repositories with no language as Markdown repositories.
    # Because most of Github repositories have a Readme.md file.
    mask = df['repository_language'].isnull()
    df.loc[mask, 'repository_language'] = 'Markdown'

    # There are too few repositories tagged as SQL repositories.
    # To mitigate this problem, a list of known repositories are flagged as
    # SQL repositories.
    sql_df = pd.read_csv(SQL_DATASET_PATH)
    mask = df['repository_name'].isin(sql_df['repository_name'])
    df.loc[mask, 'repository_language'] = 'SQL'

    df.to_csv(output_path, index=False)
