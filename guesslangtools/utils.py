import logging
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Iterator
import webbrowser

import guesslang
import pandas as pd
from matplotlib import pyplot as plt

from guesslangtools.common import Config, LOG_STEP


LOGGER = logging.getLogger(__name__)

MIN_CONFUSION_RATIO = 2/100
EPSILON = 1e-16
INDEX_TEMPLATE_PATH = Path(__file__).parent.joinpath(
    'data', 'utils', 'confusion_matrix_template.html'
)

Report = Dict[str, Dict[str, int]]
ReportGroup = Dict[str, int]
ReportGraph = Dict[str, List[Dict[str, Any]]]


def plot_prediction_confidence(config: Config) -> None:
    utils_path = _setup_utils(config)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_path = utils_path.joinpath(f'prediction_confidence_{timestamp}')
    output_path.mkdir(exist_ok=True)
    report_path = output_path.joinpath('report.csv')

    test_path = config.extracted_files_dir.joinpath('test')

    LOGGER.info('Running tests')
    full_df = pd.DataFrame(_run_tests(test_path))
    full_df.to_csv(report_path, index=False)

    LOGGER.info('Saving figures')
    plt.rcParams['figure.figsize'] = (20, 10)
    labels = sorted(full_df['_label_'].drop_duplicates())
    for label in labels:
        df = full_df.copy()
        df = df[df['_label_'] == label].drop(columns=['_label_'])

        medians = df.median(axis=0).sort_values(ascending=False)
        df = df[medians.index]
        df.plot.box(
            title=f'{label}: prediction probabilities',
            showfliers=False,
            rot=90,
        )
        plt.savefig(output_path.joinpath(f'{label}.png'))
        plt.close()

    LOGGER.info('Done')


def _setup_utils(config: Config) -> Path:
    utils_path = Path(config.cache_dir, 'utils').absolute()
    utils_path.mkdir(exist_ok=True)
    return utils_path


def _run_tests(test_path: Path) -> Iterator[Dict[str, Any]]:
    guess = guesslang.Guess()
    for index, filename in enumerate(test_path.iterdir(), 1):
        ext = filename.suffix.strip('.')
        if ext not in guess._extension_map:
            continue  # File type not supported

        label = guess._extension_map[ext]
        content = filename.read_text()
        result = {'_label_': label} | dict(guess.probabilities(content))
        yield result

        if index % LOG_STEP == 0:
            LOGGER.info(f'Processed {index} files')

    LOGGER.info('Processed all files')


def show_confusion_matrix(config: Config, filename: str) -> None:
    report = json.loads(Path(filename).read_text())
    graph_data = _build_graph(report)
    index_path = _prepare_resources(config, graph_data)
    webbrowser.open(str(index_path))


def _build_graph(report: Report) -> ReportGraph:
    groups = _build_groups(report)
    return {
        'nodes':  [
            {'name': label, 'group': groups[label]} for label in report
        ],
        'links': [
            {
                'source': index_label,
                'target': index_prediction,
                'value': value / (sum(predictions.values()) or EPSILON),
            }
            for index_label, (_, predictions) in enumerate(report.items())
            for index_prediction, (_, value) in enumerate(predictions.items())
        ]
    }


def _build_groups(report: Report) -> ReportGroup:
    total = {
        label: sum(predictions.values())
        for label, predictions in report.items()
    }

    shares = {
        label: [
            prediction
            for prediction, value in predictions.items()
            if (value / total[label]) > MIN_CONFUSION_RATIO
        ]
        for label, predictions in report.items()
    }

    mutual_shares = {
        label: [
            prediction
            for prediction in predictions
            if label in shares[prediction]
        ]
        for label, predictions in shares.items()
    }

    current_groups = [set(items) for items in mutual_shares.values()]
    groups: List[Set[str]] = []
    for current_group in current_groups:
        for group in groups:
            if any(item for item in group if item in current_group):
                group.update(current_group)
                break
        else:
            groups.append(current_group)

    default: Set[str] = set()
    joined_groups = [default]
    for group in groups:
        if len(group) > 1:
            joined_groups.append(group)
        else:
            default.update(group)

    group_map = {
        key: position
        for position, group in enumerate(joined_groups)
        for key in group
    }

    return group_map


def _prepare_resources(config: Config, graph_data: ReportGraph) -> Path:
    utils_path = _setup_utils(config)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_path = utils_path.joinpath(f'confusion_matrix_{timestamp}')
    output_path.mkdir(exist_ok=True)
    index_file = output_path.joinpath('index.html')

    template_content = INDEX_TEMPLATE_PATH.read_text()
    data_json = json.dumps(graph_data, indent=2, sort_keys=True)
    content = template_content.replace('__DATA__', data_json)
    index_file.write_text(content)

    LOGGER.info(f'Report graph available at {index_file}')
    return index_file
