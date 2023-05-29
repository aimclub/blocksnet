# Masterplanning

![Your logo](https://sun9-46.userapi.com/impf/aUFBStH0x_6jN9UhgwrKN1WN4hZ9Y2HMMrXT2w/NuzVobaGlZ0.jpg?size=1590x400&quality=95&crop=0,0,1878,472&sign=9d33baa41a86de35d951d4bbd8011994&type=cover_group)

[![Azure](https://dev.azure.com/scikit-learn/scikit-learn/_apis/build/status/scikit-learn.scikit-learn?branchName=main)](https://dev.azure.com/scikit-learn/scikit-learn/_build/latest?definitionId=1&branchName=main)
[![CirrusCI](https://img.shields.io/cirrus/github/scikit-learn/scikit-learn/main?label=Cirrus%20CI)](https://circleci.com/gh/scikit-learn/scikit-learn)
[![Codecov](https://codecov.io/gh/scikit-learn/scikit-learn/branch/main/graph/badge.svg?token=Pk8G9gg3y9)](https://codecov.io/gh/scikit-learn/scikit-learn)
[![Nightly wheels](https://github.com/scikit-learn/scikit-learn/workflows/Wheel%20builder/badge.svg?event=schedule)](https://github.com/scikit-learn/scikit-learn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/scikit-learn/)
[![PyPi](https://img.shields.io/pypi/v/scikit-learn)](https://pypi.org/project/scikit-learn)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/21369/scikit-learn/scikit-learn.svg)](https://zenodo.org/badge/latestdoi/21369/scikit-learn/scikit-learn)

## The purpose of the project

**Masterplanning** is an open-source library for generating master plan requirements for urban areas. While achieving the main goal of generating requirements, the library provides more than that:

- Simple yet detailed **city information model** based on urban blocks accessibility in graph.
- **Provision assessment** based on normative requirements.
- Urban territory **parameters balance** based on city development **concept**.

## Table of Contents

- [Masterplanning](#masterplanning)
  - [The purpose of the project](#the-purpose-of-the-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Example](#example)
  - [Documentation](#documentation)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Contacts](#contacts)
  - [Citation](#citation)

## Installation

*Masterplanning* can be installed with `git clone`:

```
git clone https://github.com/iduprojects/masterplanning
```

Then use the library by importing classes from `masterplan_tools`. For example:

```
from masterplan_tools.City_model.city_model import CityModel
```

For more detailed use case see our [example notebook](examples/workflow.ipynb).

## Example

[Example notebook](examples/workflow.ipynb) includes following steps of using the library:

1. **City information model creation** - how to create city information graph model, containing blocks as nodes and distances between blocks as edges.
2. **Service type provision evaluation** - service type provision assessment on city information model.
3. **Balancing urban territory parameters** - balancing urban territory parameters for requirements generation.

## Documentation

We have a [GitBook page](https://iduprojects.gitbook.io/masterplanning/)

## License

The project has [BSD-3-Clause license](./LICENSE.md)

## Acknowledgments

The library was developed as the main part of the ITMO University project #622280 **"Machine learning algorithms library for the tasks of generating value-oriented requirements for urban areas master planning"**

## Contacts

You can contact us through telegram or email:

- [IDU](https://idu.itmo.ru/en/) - Institute of Design and Urban Studies

## Citation

@article{"name",
  title = {},
  author = {},
  journal = {},
  year = {},
  issn = {},
  doi = {}}

bibtex-ссылку удобно брать с google scholar
