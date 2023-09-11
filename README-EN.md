# BlockNet

![Your logo](https://i.ibb.co/jTVHdkp/background-without-some.png)

[![Documentation Status](https://readthedocs.org/projects/blocknet/badge/?version=latest)](https://blocknet.readthedocs.io/en/latest/?badge=latest)
[![PythonVersion](https://img.shields.io/badge/python-3.10-blue)](https://pypi.org/project/masterplan_tools/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Readme_ru](https://img.shields.io/badge/lang-ru-yellow.svg)](README.md)

## The purpose of the project

**BlockNet** is an open-source library for generating master plan requirements for urban areas. While achieving the main goal of generating requirements, the library provides more than that:

- Simple yet detailed **city information model** based on city blocks accessibility in graph.
- A **city blocks** generating method.
- Optimized **master plan** generating method provided by **genetic algorithm** according to certain city development **scenario**.
- Fast **provision assessment** method based on normative requirements and linear optimization algorithm.
- Urban territory **parameters balance** based on city development **scenario**.

## Table of Contents

- [BlockNet](#blocknet)
  - [The purpose of the project](#the-purpose-of-the-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Documentation](#documentation)
  - [Developing](#developing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Contacts](#contacts)
  - [Citation](#citation)

## Installation

_masterplan_tools_ can be installed with `pip`:

1. `pip install git+https://github.com/iduprojects/masterplanning`

Then use the library by importing classes from `masterplan_tools`. For example:

```python
from masterplan_tools import CityModel
```

For more detailed use case see our [examples](#examples) below.

## Examples

Before running the examples, please, download the [input data](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP) and place it in `examples/data` directory. You are free to use your own data, but it should match specification classes. Next examples will help to get used to the library:

1. [City blocks generating](examples/1%20blocks_cutter.ipynb) - city blocks generating according to landuse and buildings clustering.
2. [Aggregating city blocks information](examples/2%20data_getter.ipynb) - how to fill blocks with aggregated information and also generate the accessibility matrix between blocks.
3. [City model creation](examples/3%20city_model.ipynb) - how to create the **city model** and visualize it (to make sure it is real and correct).
4. [Linear optimization provision assessment](examples/3a%20city_model%20lp_provision.ipynb) - how to assess provision of certain city service type.
5. [Iterative algorithm provision assessment](examples/3b%20city_model%20iterative_provision.ipynb) - another example of how to assess provision, but using different iterative method.
6. [Genetic algorithm master plan optimization](examples/3d%20city_model%20genetic.ipynb) - how to generate optimized master plans for certain territory or the whole city according to certain scenario.
7. [Balancing territory parameters](examples/3c%20city_model%20balancer.ipynb) - how to increase certain territory population without decreasing the quality of life of the city.

We advice to start with [city model creation](examples/3%20city_model.ipynb) example, if you downloaded the [input data](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP) we prepared.

## Documentation

We have a [documentation](https://blocknet.readthedocs.io/en/latest/), but our [examples](#examples) will explain the use cases cleaner.

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

The project has [BSD-3-Clause license](./LICENSE)

## Acknowledgments

The library was developed as the main part of the ITMO University project #622280 **"Machine learning algorithms library for the tasks of generating value-oriented requirements for urban areas master planning"**

## Contacts

You can contact us:

- [NCCR](https://actcognitive.org/o-tsentre/kontakty) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/en/contacts/contacts.htm) - Institute of Design and Urban Studies
- [Tanya Churyakova](https://t.me/tanya_chk) - project manager

## Citation

Published:

1. [Morozov A. S. et al. Assessing the transport connectivity of urban territories, based on intermodal transport accessibility // Frontiers in Built Environment. – 2023. – Т. 9. – С. 1148708.](https://www.frontiersin.org/articles/10.3389/fbuil.2023.1148708/full)
2. [Kontsevik G. et al. Assessment of Spatial Inequality in Agglomeration Planning // International Conference on Computational Science and Its Applications. – Cham : Springer Nature Switzerland, 2023. – С. 256-269.](https://link.springer.com/chapter/10.1007/978-3-031-36808-0_17)
3. [Morozov A. et al. Assessment of Spatial Inequality Through the Accessibility of Urban Services // International Conference on Computational Science and Its Applications. – Cham : Springer Nature Switzerland, 2023. – С. 270-286.](https://link.springer.com/chapter/10.1007/978-3-031-36808-0_18)
4. [Судакова В. В. Символический капитал территории как ресурс ревитализации: методики выявления // Вестник Омского государственного педагогического университета. Гуманитарные исследования. – 2023. – №. 2 (39). – С. 45-49](<https://vestnik-omgpu.ru/volume/2023-2-39/vestnik_2(39)2023_45-49.pdf>)

Accepted:

1. Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S. Digital Master Plan as a tool for generating territory development requirements // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
2. Natykin M.V., Budenny S., Zakharenko N. and Mityagin S.A. Comparison of solution methods the maximal covering location problem of public spaces for teenagers in the urban environment // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
3. Natykin M.V., Morozov A., Starikov V. and Mityagin S.A. A method for automatically identifying vacant area in the current urban environment based on open source data // 12th International Young Scientists Conference in Computational Science – YSC 2023
4. Kontsevik G., Churiakova T., Markovskiy V., Antonov A. and Mityagin S. Urban blocks modelling method // 12th International Young Scientists Conference in Computational Science – YSC 2023
