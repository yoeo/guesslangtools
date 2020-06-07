#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from contextlib import suppress
import logging.config
from typing import Dict, Any

from guesslangtools import hacks
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
        '-d', '--debug', action='store_true',
        help='display debug messages')

    parser.add_argument(
        'CACHE_DIR',
        help='directory where the generated content will be stored')
    parser.add_argument(
        '--nb-train-files', type=int, default=27000,
        help='number of training files per language')
    parser.add_argument(
        '--nb-valid-files', type=int, default=4000,
        help='number of validation files per language')
    parser.add_argument(
        '--nb-test-files', type=int, default=4000,
        help='number of testing files per language')
    parser.add_argument(
        '--nb-repo', type=int, default=4000,
        help='number of repositories per language')

    parser.add_argument(
        '--hack-repo-dist', action='store_true', default=False,
        help='show the number of selected repositories per languages')
    parser.add_argument(
        '--hack-add-repo', nargs='+', metavar='LANGUAGE',
        help='select more repositories for the listed languages')
    parser.add_argument(
        '--hack-only-downloaded-repo', action='store_true', default=False,
        help='only use the repositories that have already been downloaded')

    args = parser.parse_args()
    items = vars(args).items()
    hack_args = any(val for name, val in items if name.startswith('hack_'))

    log_level = 'DEBUG' if args.debug else 'INFO'
    LOGGING_CONFIG['root']['level'] = log_level
    logging.config.dictConfig(LOGGING_CONFIG)

    Config.setup(
        cache_dir=args.CACHE_DIR,
        nb_repositories=args.nb_repo,
        nb_train=args.nb_train_files,
        nb_valid=args.nb_valid_files,
        nb_test=args.nb_test_files,
    )

    with suppress(KeyboardInterrupt):
        if hack_args:
            run_hacks(args)
        else:
            run_workflow()


def run_hacks(args: Namespace) -> None:
    if args.hack_repo_dist:
        hacks.show_repositories_distribution()

    if args.hack_add_repo:
        hacks.select_more_repositories(args.hack_add_repo)

    if args.hack_only_downloaded_repo:
        hacks.select_only_downloaded_repo()


if __name__ == '__main__':
    main()
