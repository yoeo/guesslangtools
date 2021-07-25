#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from contextlib import suppress
import logging.config
from typing import Dict, Any

from guesslangtools import hacks
from guesslangtools import utils
from guesslangtools.common import Config
from guesslangtools.app import run_workflow


LOGGING_CONFIG: Dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s: %(message)s',
            'datefmt': '%H:%M:%S',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console'],
    },
}

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description='Guesslang data preparation tool')
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='display debug messages',
    )

    # Setup to generate Guesslang training, validation and test datasets
    parser.add_argument(
        'CACHE_DIR',
        help='directory where the generated content will be stored',
    )
    parser.add_argument(
        '--nb-repo',
        type=int,
        default=8000,
        help='number of repositories per language',
    )
    parser.add_argument(
        '--nb-train-files',
        type=int,
        default=27000,
        help='number of training files per language',
    )
    parser.add_argument(
        '--nb-valid-files',
        type=int,
        default=4000,
        help='number of validation files per language',
    )
    parser.add_argument(
        '--nb-test-files',
        type=int,
        default=4000,
        help='number of testing files per language',
    )

    # Utils to analyse Guesslang model performances
    parser.add_argument(
        '--util-prediction-confidence',
        action='store_true',
        default=False,
        help='plot the prediction probabilies distribution for each language',
    )
    parser.add_argument(
        '--util-confusion-matrix',
        metavar='GUESSLANG_TEST_REPORT_FILENAME',
        help='show languages that Guesslange confuses with others',
    )
    parser.add_argument(
        '--util-less-training-files',
        type=int,
        metavar='NB_FILES_PER_LANGUAGE',
        help='extract a subset of the training files dataset',
    )

    # Hacks to use when you don't have enough files for some language
    parser.add_argument(
        '--hack-repo-dist',
        action='store_true',
        default=False,
        help='show the number of selected repositories per languages',
    )
    parser.add_argument(
        '--hack-download-repo-list',
        nargs=3,
        type=str,
        # To get a Github token, check https://developer.github.com/v3/oauth/
        metavar=('GITHUB_TOKEN', 'LANGUAGE', 'REPO_LIST_FILENAME'),
        help='download a list or repository names from Github for a language',
    )
    parser.add_argument(
        '--hack-merge-repo-list',
        metavar='REPO_LIST_FILENAME',
        help='merge downloaded repository names to the selected repositories',
    )
    parser.add_argument(
        '--hack-add-repo',
        nargs='+',
        metavar='LANGUAGE',
        help='select more repositories for the listed languages',
    )
    parser.add_argument(
        '--hack-only-use-downloaded-repo',
        action='store_true',
        default=False,
        help='only use the repositories that have already been downloaded',
    )

    args = parser.parse_args()
    items = vars(args).items()
    util_args = any(val for name, val in items if name.startswith('util_'))
    hack_args = any(val for name, val in items if name.startswith('hack_'))

    log_level = 'DEBUG' if args.debug else 'INFO'
    LOGGING_CONFIG['root']['level'] = log_level
    logging.config.dictConfig(LOGGING_CONFIG)

    config = Config(
        cache_dir=args.CACHE_DIR,
        nb_repositories=args.nb_repo,
        nb_train=args.nb_train_files,
        nb_valid=args.nb_valid_files,
        nb_test=args.nb_test_files,
    )

    with suppress(KeyboardInterrupt):
        if util_args:
            run_utils(config, args)
        elif hack_args:
            run_hacks(config, args)
        else:
            run_workflow(config)


def run_utils(config: Config, args: Namespace) -> None:
    if args.util_prediction_confidence:
        utils.plot_prediction_confidence(config)

    if args.util_confusion_matrix:
        utils.show_confusion_matrix(config, args.util_confusion_matrix)

    if args.util_less_training_files:
        utils.shring_training_dataset(config, args.util_less_training_files)


def run_hacks(config: Config, args: Namespace) -> None:
    if args.hack_repo_dist:
        hacks.show_repositories_distribution(config)

    if args.hack_add_repo:
        hacks.select_more_repositories(config, args.hack_add_repo)

    if args.hack_download_repo_list:
        hacks.download_github_repo_list(config, *args.hack_download_repo_list)

    if args.hack_merge_repo_list:
        hacks.merge_to_selected_repositories(config, args.hack_merge_repo_list)

    if args.hack_only_use_downloaded_repo:
        hacks.select_only_downloaded_repo(config)


if __name__ == '__main__':
    main()
