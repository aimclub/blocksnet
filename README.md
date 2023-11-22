# BlocksNet

![Your logo](https://i.ibb.co/QC9XD07/blocksnet.png)

[![Documentation Status](https://readthedocs.org/projects/blocknet/badge/?version=latest)](https://blocknet.readthedocs.io/en/latest/?badge=latest)
[![PythonVersion](https://img.shields.io/badge/python-3.10-blue)](https://pypi.org/project/blocksnet/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Readme_ru](https://img.shields.io/badge/lang-ru-yellow.svg)](README-RU.md)

[![Example on Vologda city: Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mk0Bs0s-JNn8QdYfWy1uSmyptrBG1Ch_?usp=sharing)


**BlocksNet** is an open-source library that includes methods of modeling urbanized areas for the generation of value-oriented master planning requirements. The library is provided for generating a universal information city model based on the accessibility of urban blocks. The library also provides tools for working with the information city model, which allow: to generate a layer of urban blocks, to calculate provisioning based on regulatory requirements, to obtain optimal requirements for master planning of territories.

## BlocksNet Features

BlocksNet — a library for modeling urban development scenarios (e.g. creating a master plan), supporting the following tools:

1. The method of generating a layer of urban blocks is the division of the territory into the smallest elements for the analysis of the urban area - blocks. The method of generating a layer of urban blocks is based on clustering algorithms taking into account additional data on land use.
2. The Universal Information City Model is used to further analyze urban areas and to obtain information on the accessibility of urban blocks. The City Model includes aggregated information on services, intermodal accessibility and urban blocks.
3. Methods for assessing urban provision of different types of services with regard to normative requirements and value attitudes of the population. The estimation of provisioning is performed by iterative algorithm on graphs, as well as by solving linear optimization problem.
4. A method for computing the function for evaluating the optimality of master planning projects based on the value attitudes of the population and systems of external limitations. The method is based on solving an optimization problem: it is necessary to find an optimal development to increase the provision. The problem is solved with the help of genetic algorithm, user scenarios support is added.

Main differences from existing solutions:

- The method of generating a layer of **urban blocks** considers the type of land use, which makes it possible to define limitations for the development of the territory in the context of master planning.
- The universal information **city model** can be built on open data; the smallest spatial unit for analysis is a block, which makes it possible to analyze on a city scale.
- Not only normative documents are taken into account when assessing **provisioning**, but also the value attitudes of the population.
- Genetic algorithm for optimization of development supports user-defined **scenarios**.
- Support for different regulatory requirements.

## Installation

**BlocksNet** can be installed with `pip`:

```
pip install git+https://github.com/iduprojects/blocksnet
```

## How to use

Then use the library by importing classes from `blocksnet`:

```
from blocksnet import CityModel
```

Next, use the necessary functions and modules:

```
city_model = CityModel(
  blocks=aggregated_blocks,
  accessibility_matrix=accessibility_matrix,
  services=services
)
city_model.visualize()
```

For more detailed use case see our [examples](#examples) below.

## Data

Before running the examples, you must download the [input data](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP) and place it in the `examples/data` directory. You can use your own data, but it must follow the structure described in the [specification](https://blocknet.readthedocs.io/en/latest/index.html).

## Examples

Next examples will help to get used to the library:

1. [City blocks generating](examples/1%20blocks_cutter.ipynb) - city blocks generating according to landuse and buildings clustering.
2. [Aggregating city blocks information](examples/2%20data_getter.ipynb) - how to fill blocks with aggregated information and also generate the accessibility matrix between blocks.
3. [City model creation](examples/3%20city_model.ipynb) - how to create the **city model** and visualize it (to make sure it is real and correct).
4. [Linear optimization provision assessment](examples/3a%20city_model%20lp_provision.ipynb) - how to assess provision of certain city service type.
5. [Iterative algorithm provision assessment](examples/3b%20city_model%20iterative_provision.ipynb) - another example of how to assess provision, but using different iterative method.
6. [Genetic algorithm master plan optimization](examples/3d%20city_model%20genetic.ipynb) - how to optimize the search for master planning requirements for a specific area or the entire city in a specific scenario.
7. [Balancing territory parameters](examples/3c%20city_model%20balancer.ipynb) - how to increase certain territory population without decreasing the quality of life of the city.

We advice to start with [city model creation](examples/3%20city_model.ipynb) example, if you downloaded the [input data](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP) we prepared.

## Documentation

Detailed information and description of BlocksNet is available in [documentation](https://blocknet.readthedocs.io/en/latest/).

## Project Structure

The latest version of the library is available in the main branch.

The repository includes the following directories and modules:

- [**blocksnet**](https://github.com/iduprojects/blocksnet/tree/main/blocksnet) - directory with the framework code:
  - preprocessing - preprocessing module
  - models - model classes
  - method - library tool methods
  - utils - module for static units of measure
- [data](https://github.com/iduprojects/blocksnet/tree/main/data) - directory with data for experiments and tests
- [tests](https://github.com/iduprojects/blocksnet/tree/main/tests) - directory with units of measurement and integration tests
- [examples](https://github.com/iduprojects/blocksnet/tree/main/examples) - directory with examples of how the methods work
- [docs](https://github.com/iduprojects/blocksnet/tree/main/docs) - directory with RTD documentation

## Developing

To start developing the library, one must perform following actions:

1. Clone repository (`git clone https://github.com/iduprojects/blocksnet`)
2. (optionally) create a virtual environment as the library demands exact packages versions: `python -m venv venv` and activate it.
3. Install the library in editable mode: `python -m pip install -e '.[dev]' --config-settings editable_mode=strict`
4. Install pre-commit hooks: `pre-commit install`
5. Create a new branch based on **develop**: `git checkout -b develop <new_branch_name>`
6. Add changes to the code
7. Make a commit, push the new branch and create a pull-request into **develop**

Editable installation allows to keep the number of re-installs to the minimum. A developer will need to repeat step 3 in case of adding new files to the library.

## License

The project has [BSD-3-Clause license](./LICENSE)

## Acknowledgments

The library was developed as the main part of the ITMO University project #622280 **"Machine learning algorithms library for the tasks of generating value-oriented requirements for urban areas master planning"**

## Contacts

You can contact us:

- [NCCR](https://actcognitive.org/o-tsentre/kontakty) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/en/contacts/contacts.htm) - Institute of Design and Urban Studies
- [Tatiana Churiakova](https://t.me/tanya_chk) - project manager

## Publications on library tools

Published:

1. [Morozov A. S. et al. Assessing the transport connectivity of urban territories, based on intermodal transport accessibility // Frontiers in Built Environment. – 2023. – Т. 9. – С. 1148708.](https://www.frontiersin.org/articles/10.3389/fbuil.2023.1148708/full)
2. [Morozov A. et al. Assessment of Spatial Inequality Through the Accessibility of Urban Services // International Conference on Computational Science and Its Applications. – Cham : Springer Nature Switzerland, 2023. – С. 270-286.](https://link.springer.com/chapter/10.1007/978-3-031-36808-0_18)

Accepted:

1. Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S. Digital Master Plan as a tool for generating territory development requirements // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
2. Natykin M.V., Budenny S., Zakharenko N. and Mityagin S.A. Comparison of solution methods the maximal covering location problem of public spaces for teenagers in the urban environment // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
3. Natykin M.V., Morozov A., Starikov V. and Mityagin S.A. A method for automatically identifying vacant area in the current urban environment based on open source data // 12th International Young Scientists Conference in Computational Science – YSC 2023
4. Kontsevik G., Churiakova T., Markovskiy V., Antonov A. and Mityagin S. Urban blocks modelling method // 12th International Young Scientists Conference in Computational Science – YSC 2023
