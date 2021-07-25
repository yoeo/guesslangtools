#!/usr/bin/env python3

import ast
from pathlib import Path
import re

from setuptools import setup, find_packages


def version(base_module: str) -> str:
    version_pattern = r'__version__\s+=\s+(.*)'
    init_path = Path(Path(__file__).parent, base_module, '__init__.py')
    found = re.search(version_pattern, init_path.read_text())

    if not found:
        raise RuntimeError(f'Version not found in {init_path}')

    repr_value = found.group(1)
    return format(ast.literal_eval(repr_value))


if __name__ == '__main__':
    # Avoids calling `setup(...)` inside Guesslangtools
    # process pools with `spawn` context,
    # when running the unit tests with `python setup.py test`

    setup(
        # Package info
        name="guesslangtools",
        author="Y. SOMDA",
        version=version('guesslangtools'),
        url="http://github.com/yoeo/guesslangtools",
        description="Guesslang tools python package",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
        ],
        # Install
        python_requires='>=3',
        platforms='any',
        packages=find_packages(exclude=['tests']),
        install_requires=Path('requirements.txt').read_text(),
        zip_safe=True,
        include_package_data=True,
        # Test
        setup_requires=['pytest-runner'],
        tests_require='pytest',
        # Execute
        entry_points={
            'console_scripts': ['gltool = guesslangtools.__main__:main']
        },
    )
