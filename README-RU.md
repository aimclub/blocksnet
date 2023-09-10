# UrbanBlockNet

![Your logo](https://psv4.userapi.com/c236331/u6931256/docs/d54/bf3e6a5a3aeb/background-without-some.png?extra=0UhxWRG5hnl9wMXt_xuNBJnKPk28rqvDqW990UqdJJjJ0VnbhDq9qKd7UQawD2-QVz1QMP_ekK4Iw0e6oa1vPVYtwcgeQcAZ0FyTXaGT38JxBvhU5v46AwiQza1Q25Xsnb52wSvF_bqdRirFZyg)

[![PythonVersion](https://img.shields.io/badge/python-3.10-blue)](https://pypi.org/project/masterplan_tools/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Readme_en](https://img.shields.io/badge/lang-en-red.svg)](README.md)


**UrbanBlockNet** – это библиотека с открытым исходным кодом для генерации требований к мастер-планированию городских территорий. Библиотека предназначена для формирования универсальной информационной модели города, основанной на дсотупности городских кварталов. В бибилотеке также представлены инструменты для работы с информационной моделью города, которые позволяют: генерировать слой городских кварталов, рассчитывать обеспеченность на основе нормативных требований, а также составлять оптимальные требования к мастер-планированию территорий.

## UrbanBlockNet Features
UrbanBlockNet — это библиотека для моделирования развития города, поддерживающая следующие инструменты:

1. Метод генерации городских кварталов, который позволяет разделять территорию на наименьшие элементы для анализа городской территории - кварталы. Метод генерации городских кварталов основан на алгоритмах кластеризации с учетом дополнительных данных о землпепользовании.
2. Универсальная информационная модель города, которая может использоваться для анализа городских территорий и доступности городских кварталов. Универсальаня модель города включает аггрегированную информацию о сервисах, интермодальной доступности и городских кварталах.
3. Методы оценки городской обеспеченности различными типами сервисов, которые учитывают нормативные требования и покаызвают ... что-то про ценностно-ориентированные и качество жизни граждан. Оценка обеспеченности производится при помощи итеративного алгоритма на графах, а также при помощи решения задачи линейной оптимизации.
4. Метод генерации оптимальных требований к мастер-планированию территории с целью определения дальнейшего развития. Метод генерации основан на решении задачи оптимизации, необходимо найти оптимальную застройку, чтобы повысить обеспеченность. Задача решеается при помощи генетического алгоритма, добавлена поддержка пользовательских сценариев.

Основные отличия от существующих решений:

- сетевая модель - на графах кварталов
- Метод генерации кварталов учитывет LANDUSE
- При оценки обеспеченности учитываются не только нормативные документы, но и ценностно-ориентированные функции населения. 
- Генетический алгоритм для оптимизации застройки поддерживает пользовательские сценарии.
- Поддержка различных нормативных требований.

## Установка

Чтобы установить библиотеку **UrbanBlockNet** необходимо использовать `pip`:

```
pip install git+https://github.com/iduprojects/masterplanning
```

## How to Use

Then the necessary classes are imported from the library:

```
from masterplan_tools import CityModel
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

Более подробные примеры использования приведены ниже.


## Данные
Перед запуском примеров необходимо скачать [входные данные](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP) и поместить их в директорию `examples/data`. Вы можете использовать свои собственные данные, но они должны соответствовать структуре, описанной в [спецификации](https://blocknet.readthedocs.io/en/latest/index.html). 

## Примеры

Следующие примеры помогут ознакомиться с библиотекой:

1. [Генерация слоя городских кварталов](examples/1%20blocks_cutter.ipynb) в соответствии с кластеризацией землепользования и зданий.
2. [Агрегирование информации о городских кварталах](examples/2%20data_getter.ipynb) – как заполнить кварталы агрегированной информацией, а также сформировать матрицу доступности между кварталами.
3. [Создание модели города](examples/3%20city_model.ipynb) – как создать **модель города** и визуализировать ее (чтобы убедиться в ее реальности и правильности).
4. [Оценка обеспеченности с помощью линейной оптимизацией](examples/3a%20city_model%20lp_provision.ipynb) – как определить обеспеченность городскими сервисами.
5. [Оценка обеспеченности по итерационному алгоритму](examples/3b%20city_model%20iterative_provision.ipynb) – еще один пример оценки обеспеченности, но с использованием другого итерационного метода.
6. [Генетический алгоритм оптимизации требований к мастер-плану](examples/3d%20city_model%20genetic.ipynb) – как оптимизировать требования к мастер-планированию для определенной территории или всего города по определенному сценарию.
7. [Балансировка параметров территории](examples/3c%20city_model%20balancer.ipynb) при увеличении жилой площади, не снижая качество жизни в городе.

Мы советуем начать с п. 3 [создание модели города](examples/3%20city_model.ipynb), если вы скачали подготовленные нами [исходные данные](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP).

## Документация

У нас есть [документация](https://iduprojects.github.io/masterplanning/), но наши [примеры](#примеры) лучше объяснят варианты использования.

## Project structure

Последняя версия библиотеки доступна в master branch.

It includes the follwoing modules and directories:

* [**masterplan_tools**](https://github.com/iduprojects/masterplanning/tree/main/masterplan_tools) - directory with the framework code:
    * preprocessing - module for data preprocessing
    * models - models classes
    * method - methods provided by masterplan_tool
    * utils - module for measurements units static values
* [data](https://github.com/iduprojects/masterplanning/tree/main/data) - directory with data for experiments and tests
* [tests](https://github.com/iduprojects/masterplanning/tree/main/tests) - directory with unit and integration tests
* [eaxmples](https://github.com/iduprojects/masterplanning/tree/main/examples) - directory with tutorials
* [docs](https://github.com/iduprojects/masterplanning/tree/main/docs) - directory with RTD documentation



## Разработка

Для начала разработки библиотеки необходимо выполнить следующие действия:

1. Склонируйте репозиторий: `git clone https://github.com/iduprojects/masterplanning`.
2. Создайте виртуальную среду (опционально), поскольку библиотека требует точных версий пакетов: `python -m venv venv` и активируйте ее.
3. Установите библиотеку в редактируемом режиме: `python -m pip install -e '.[dev]' --config-settings editable_mode=strict`.
4. Устаовите pre-commit hooks: `pre-commit install`.
5. Создайте новую ветку на основе **develop**: `git checkout -b develop <new_branch_name>`.
6. Добавьте изменения в код.
7. Сделайте коммит, переместите новую ветку и выполните pull-request в **develop**.

Редактируемая установка позволяет свести к минимуму количество повторных инсталляций. При добавлении новых файлов в библиотеку разработчику необходимо повторить шаг 3.

## Лицензия

Проект имеет [лицензию BSD-3-Clause](./LICENSE.md).

## Благодарности

Библиотека была разработана как основная часть проекта Университета ИТМО № 622280 **"Библиотека алгоритмов машинного обучения для задач генерации ценностно-ориентированных требований к мастер-планированию урбанизированных территорий"**.

## Контакты

Вы можете связаться с нами:

- [НЦКР](https://actcognitive.org/o-tsentre/kontakty) – Национальный Центр Когнитивных Разработок
- [ИДУ](https://idu.itmo.ru/en/contacts/contacts.htm) – Институт Дизайна и Урбанистики
- [Татьяна Чурякова](https://t.me/tanya_chk) – руководитель проекта

## Ссылки

Опубликованные работы:

1. [Morozov A. S. et al. Assessing the transport connectivity of urban territories, based on intermodal transport accessibility // Frontiers in Built Environment. – 2023. – Т. 9. – С. 1148708.](https://www.frontiersin.org/articles/10.3389/fbuil.2023.1148708/full)
2. [Kontsevik G. et al. Assessment of Spatial Inequality in Agglomeration Planning // International Conference on Computational Science and Its Applications. – Cham : Springer Nature Switzerland, 2023. – С. 256-269.](https://link.springer.com/chapter/10.1007/978-3-031-36808-0_17)
3. [Morozov A. et al. Assessment of Spatial Inequality Through the Accessibility of Urban Services // International Conference on Computational Science and Its Applications. – Cham : Springer Nature Switzerland, 2023. – С. 270-286.](https://link.springer.com/chapter/10.1007/978-3-031-36808-0_18)
4. [Судакова В. В. Символический капитал территории как ресурс ревитализации: методики выявления // Вестник Омского государственного педагогического университета. Гуманитарные исследования. – 2023. – №. 2 (39). – С. 45-49](<https://vestnik-omgpu.ru/volume/2023-2-39/vestnik_2(39)2023_45-49.pdf>)

Принятые к публикации работы:

1. Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S. Digital Master Plan as a tool for generating territory development requirements // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
2. Natykin M.V., Budenny S., Zakharenko N. and Mityagin S.A. Comparison of solution methods the maximal covering location problem of public spaces for teenagers in the urban environment // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
3. Natykin M.V., Morozov A., Starikov V. and Mityagin S.A. A method for automatically identifying vacant area in the current urban environment based on open source data // 12th International Young Scientists Conference in Computational Science – YSC 2023
4. Kontsevik G., Churiakova T., Markovskiy V., Antonov A. and Mityagin S. Urban blocks modelling method // 12th International Young Scientists Conference in Computational Science – YSC 2023
