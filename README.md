# BlocksNet

<!-- logo-start -->

![BlocksNet logo](https://i.ibb.co/QC9XD07/blocksnet.png)

<!-- logo-end -->

<!-- description-start -->

[![Documentation Status](https://github.com/aimclub/blocksnet/actions/workflows/documentation.yml/badge.svg?branch=main)](https://aimclub.github.io/blocksnet/)
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://pypi.org/project/blocksnet/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Readme ru](https://img.shields.io/badge/lang-ru-yellow.svg)](README-RU.rst)

**BlocksNet** is an open-source library that includes methods of modeling urbanized areas for the generation of
value-oriented master planning requirements. The library provides tools for generating an information city model based
on the accessibility of urban blocks. The library also provides tools for working with the information city model,
which allows one: to assess urban network metrics such as connectivity and centrality, to calculate service type
provision based on regulatory requirements and to obtain optimal requirements for master planning of territories.

<!-- description-end -->

<!-- features-start -->

## Features

BlocksNet — a library for modeling urban development scenarios (e.g. creating a master plan), supporting the following
tools:

- Method for generating a layer of urban blocks is the division of the territory into the smallest elements for the
  analysis of the urban area - blocks. The method of generating a layer of urban blocks is based on clustering
  algorithms taking into account additional data on land use.
- Intermodal graph generator and accessibility matrix calculator based on [IduEdu](https://github.com/DDonnyy/IduEdu)
  library.
- The Universal Information City Model is used to further analyze urban areas and to obtain information on the
  accessibility of urban blocks. The City Model includes aggregated information on services and buildings, intermodal
  accessibility, service types hierarchy, and urban blocks.
- Method for accessing the connectivity of the blocks based on intermodal accessibility.
- Methods for assessing urban provision of different types of services with regard to normative requirements and value
  attitudes of the population. The estimation of provisioning is performed by iterative algorithm on graphs, as well as
  by solving linear optimization problem.
- Method for computing the function for evaluating the optimality of master planning projects based on the value
  attitudes of the population and systems of external limitations. The method is based on solving an optimization
  problem: it is necessary to find an optimal development to increase the provision. The problem is solved with the help
  of simulated annealing algorithm, user scenarios support is added.
- Method for identifying vacant areas based on open-data.
- Land use prediction based on services within blocks.
- Centrality and diversity assessments, spacematrix morphotypes identification method, integration metric assessment
  etc.

Main differences from existing solutions:

- The method of generating a layer of **urban blocks** considers the type of land use, which makes it possible to define
  limitations for the development of the territory in the context of master planning.
- The universal information **city model** can be built on open data; the smallest spatial unit for analysis is a block,
  which makes it possible to analyze on a city scale.
- **Provision assessment** takes into account the competition element created between residents and services.
- **Services optimization** algorithm based on simulated annealing supports user-defined **scenarios**.
- Support for different regulatory requirements.
- Pretty easy to use out of the box. The library is aimed to help students, so it balances between being friendly to
  non-programmers as well as useful and respective for advanced possible users and contributors.

<!-- features-end -->

<!-- installation-start -->

## Installation

**BlocksNet** can be installed with `pip`:

```bash
pip install blocksnet
```

There are various extras to install:

- `ml` - machine learning related packages (`torch`, `catboost`, etc).
- `opt` - optimization problem related packages (`optuna`, `pymoo`).
- `full` - all packages required by blocksnet methods (`ml` + `opt` extras).
- `ipynb` - jupyter notebook visualization packages (`matplotlib`, etc).
- `tests` (DEVELOPMENT ONLY) - pytest related packages.
- `docs` (DEVELOPMENT ONLY) - sphinx documentation related packages.
- `dev` (DEVELOPMENT ONLY) - development related packages and all extras.

We recommend installing `full` and `ipynb` extras:

```bash
pip install blocksnet[full,ipynb]
```

<!-- installation-end -->

Read the [installation](https://aimclub.github.io/blocksnet/installation) section of the documentation on how to get started with blocksnet in different environments.

<!-- use-start -->

## How to use

Use the library by importing functions and classes from `blocksnet` modules:

```python
import pandas as pd
from blocksnet.analysis.network import mean_accessibility

acc_mx = pd.DataFrame([
   [0,7,5],
   [9,0,3],
   [5,4,0]
])

mean_acc_df = mean_accessibility(acc_mx)
mean_acc_df
```

```
>>> mean_acc_df
   mean_accessibility
0                 4.0
1                 4.0
2                 3.0
```

<!-- use-end -->

For more detailed use cases please see our [examples](examples).

## Data

Before running the examples, one can use the data from the [blocksnet-data](tests/data) repository releases and place it in the
`examples/data` directory, for example:

```
examples
| - data
|   | - saint_petersburg
|   |   | - blocks.pickle
|   |   | - accessibility_matrix_drive.pickle
|   |   | - ...
|- ...
```

## Examples

The examples are presented and structured in [examples](examples) directory. Same examples can be viewed in the [documentation](https://aimclub.github.io/blocksnet/examples).
Since the library is focused on operating with urban blocks spatial layer, the [pipeline](examples/pipeline.ipynb) example is highly recommended for reading.

## Documentation

Detailed information and description of BlocksNet is available in the
[documentation](https://aimclub.github.io/blocksnet).

## Project Structure

The latest version of the library is available in the `main` branch.

The repository includes the following directories and modules:

- [`blocksnet`](blocksnet) - library code:

  - [`analysis`](blocksnet/analysis) - urban blocks spatial layer assessment metrics.
  - [`blocks`](blocksnet/blocks) - methods to generate, aggregate and process urban blocks.
  - [`config`](blocksnet/config) - config to handle logging and meta information (service types, land use relations, etc).
  - [`enums`](blocksnet/enums) - enums used within the code base.
  - [`machine_learning`](blocksnet/machine_learning) - machine learning strategies for different packages. Requires `ml` extra (`pip install blocksnet[ml]`).
  - [`optimization`](blocksnet/optimization) - optimization related methods. Requires the `opt` extra (`pip install blocksnet[opt]`).
  - [`preprocessing`](blocksnet/preprocessing) - data imputing and preprocessing.
  - [`relations`](blocksnet/relations) - methods to handle network graphs and accessibility matrices.
  - [`synthesis`](blocksnet/synthesis) - methods to generate master-planning requirements using different heuristics.
  - [`utils`](blocksnet/utils) - validation related utils.

- [`tests`](https://github.com/aimclub/blocksnet/tree/main/tests) - `pytest` testing.
- [`examples`](examples) - library usage examples. Required packages can be installed with `ipynb` extra (`pip install blocksnet[ipynb]`).
- [`docs`](docs) - documentation structure.

## Contributing

Contributors are welcomed! To participate in development, please, read the [contributing](https://aimclub.github.io/blocksnet/blocksnet/contributing) section of the documentation.

## License

The project has [BSD-3-Clause license](./LICENSE).

<!-- acknowledgments-start -->

## Acknowledgments

The library was developed as the main part of the ITMO University project #622280 “Machine learning algorithms
library for the tasks of generating value-oriented requirements for urban areas master planning”

This research is financially supported by the Foundation for National Technology Initiative's Projects Support as a part
of the roadmap implementation for the development of the high-tech field of Artificial Intelligence for the period up to
2030 (agreement 70-2021-00187)

<!-- acknowledgments-end -->

<!-- contacts-start -->

## Contacts

You can contact us:

- [NCCR](https://actcognitive.org/o-tsentre/kontakty) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/en/contacts/contacts.htm) - Institute of Design and Urban Studies
- [Tatiana Churiakova](https://t.me/tanya_chk) - project manager
- [Vasilii Starikov](https://t.me/vasilstar) - lead software engineer

Also, you are welcomed to our [issues](https://github.com/aimclub/blocksnet/issues) section!

<!-- contacts-end -->

<!-- publications-start -->

## Publications

- [Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S. Digital Master Plan as a tool for generating
  territory development requirements // International Conference on Advanced Research in Technologies, Information,
  Innovation and Sustainability 2023 – ARTIIS 2023](https://link.springer.com/chapter/10.1007/978-3-031-48855-9_4)

<!-- publications-end -->
