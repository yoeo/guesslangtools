# GuesslangTools [![Build Status](https://github.com/yoeo/guesslangtools/actions/workflows/python-package.yml/badge.svg)](https://github.com/yoeo/guesslangtools/actions)

![Guesslangtools](guesslangtools/data/guesslangtools.png)

A training dataset generator for Guesslang's deep learning model.

## Description

GuesslangTools purpose is to find and download **a million** source code files.
These files are used to train, evaluate and test
[Guesslang](https://github.com/yoeo/guesslang),
a deep learning programming language detection tool.

The files are retrieved from more than **100k** public open source
GitHub repositories.

### Workflow

The million source code files used to feed Guesslang are generated as follows:

1. Download Github open source repositories information from the
[Libraries.io Open Source Repository and Dependency Metadata](https://zenodo.org/record/3626071/files/libraries-1.6.0-2020-01-12.tar.gz?download=1).
2. Randomly select the repositories that will be used to create
  Guesslang's training, validation and test datasets.
3. Download each selected repository.
4. Extract some source code files from the downloaded repositories.

This workflow is fully automated but takes several hours to complete,
especially the download part.
Fortunately, it can be stopped and resumed at any moment.

### Constraints

GuesslangTools ensures that:

* Each source code file in the datasets is unique.
* There are no empty files.
* Only text files are retrieved, binary files are skipped.
* All the files are converted to UTF-8 encoding.
* Each selected repository is associated to only one dataset
  (training, validation or test),
  therefore files from a training repository can only be in
  the training dataset. Same for the validation and test datasets.

## Usage

### Prerequisite

* GuesslangTools requires Python 3.7 or later.
* At least 16GB of total system memory is recommended.
* At least 150GB of free storage space is recommended.

### Installation

You can install GuesslangTools from the source code by running:

```bash
pip install .
```

### Execution

You can run Guesslang tools on a terminal as follows:

```bash
gltool /path/to/generated_datasets/
```

Several options and hacks are available to fine tune the size and
the diversity of the generated datasets. To list all the options, please run:

```bash
gltool --help
```

## License and credits

* [Guesslang documentation](https://guesslang.readthedocs.io/en/latest/)

* [Guesslang on Github](https://github.com/yoeo/guesslang)

* Guesslang icon created with
  [AndroidAssetStudio](https://github.com/romannurik/AndroidAssetStudio)

* Repository dataset downloaded from
  [Libraries.io Open Source Repository and Dependency Metadata](https://zenodo.org/record/1196312/files/Libraries.io-open-data-1.2.0.tar.gz)

* SQL repositories dataset retrieve from [The Public Git Archive](https://github.com/src-d/datasets/tree/master/PublicGitArchive)

* GuesslangTools — Copyright (c) 2021 Y. SOMDA, [MIT License](LICENSE)
