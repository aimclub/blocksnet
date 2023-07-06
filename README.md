# Masterplanning

![Your logo](https://sun9-46.userapi.com/impf/aUFBStH0x_6jN9UhgwrKN1WN4hZ9Y2HMMrXT2w/NuzVobaGlZ0.jpg?size=1590x400&quality=95&crop=0,0,1878,472&sign=9d33baa41a86de35d951d4bbd8011994&type=cover_group)

[![PythonVersion](https://img.shields.io/badge/python-3.10-blue)](https://pypi.org/project/masterplan_tools/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

*masterplan_tools* can be installed with `pip`:

1. `pip install git+https://github.com/iduprojects/masterplanning`

Then use the library by importing classes from `masterplan_tools`. For example:

```python
from masterplan_tools import CityModel
```

For more detailed use case see our [example notebook](examples/workflow.ipynb).

## Example

[Example notebook](examples/workflow.ipynb) includes following steps of using the library:

1. **City information model creation** - how to create city information graph model, containing blocks as nodes and distances between blocks as edges.
2. **Service type provision evaluation** - service type provision assessment on city information model.
3. **Balancing urban territory parameters** - balancing urban territory parameters for requirements generation.

## Documentation

We have a [GitBook page](https://iduprojects.gitbook.io/masterplanning/)

## Developing

To start developing the library, one must perform following actions:

1. Clone repository (`git clone https://github.com/iduprojects/masterplanning`)
2. (optionally) create a virtual environment as the library demands exact packages versions: `python -m venv venv` and activate it.
3. Install the library in editable mode: `python -m pip install -e '.[dev]' --config-settings editable_mode=strict`
4. Install pre-commit hooks: `pre-commit install`
5. Create a new branch based on **develop**: `git checkout -b develop <new_branch_name>`
6. Add changes to the code
7. Make a commit, push the new branch and create a pull-request into **develop**

Editable installation allows to keep the number of re-installs to the minimum. A developer will need to repeat step 3 in case of adding new files to the library.

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
