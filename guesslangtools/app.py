from guesslangtools.common import Config
from guesslangtools.workflow import repositories_dataset
from guesslangtools.workflow import github_repositories
from guesslangtools.workflow import source_files


def run_workflow(config: Config) -> None:
    repositories_dataset.download(config)
    repositories_dataset.extract(config)
    repositories_dataset.shrink(config)
    repositories_dataset.alter(config)

    github_repositories.select(config)
    github_repositories.prepare(config)
    github_repositories.download(config)

    source_files.list_all(config)
    source_files.deduplicate(config)
    source_files.split(config)
    source_files.extract(config)
    source_files.finalize(config)
