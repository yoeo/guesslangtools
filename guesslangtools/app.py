from guesslangtools.workflow import repositories_dataset
from guesslangtools.workflow import compressed_repositories
from guesslangtools.workflow import source_files


def run_workflow() -> None:
    repositories_dataset.download()
    repositories_dataset.extract()
    repositories_dataset.shrink()
    repositories_dataset.alter()

    compressed_repositories.select()
    compressed_repositories.prepare()
    compressed_repositories.download()

    source_files.list_all()
    source_files.split()
    source_files.extract()
