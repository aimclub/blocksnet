# UrbanBlockNet

![Your logo](https://i.ibb.co/jTVHdkp/background-without-some.png)

[![PythonVersion](https://img.shields.io/badge/python-3.10-blue)](https://pypi.org/project/masterplan_tools/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Readme_en](https://img.shields.io/badge/lang-en-red.svg)](README-EN.md)

**UrbanBlockNet** – это библиотека с открытым исходным кодом, включающая методы моделирования урбанизированных территорий для задач генерации ценностно-ориентированных требований к мастер-планированию. Библиотека предназначена для формирования универсальной информационной модели города, основанной на доступности городских кварталов. В библиотеке также представлены инструменты для работы с информационной моделью города, которые позволяют: генерировать слой городских кварталов, рассчитывать обеспеченность на основе нормативных требований, а также получить оптимальные требования к мастер-планированию территорий.

## Преимущества UrbanBlockNet

UrbanBlockNet — это библиотека для моделирования сценариев развития города, например при создании мастер-плана, поддерживающая следующие инструменты:

1. Метод генерации слоя городских кварталов - разделение территории на наименьшие элементы для анализа городской территории - кварталы. Метод генерации слоя городских кварталов основан на алгоритмах кластеризации с учетом дополнительных данных о землпепользовании.
2. Универсальная информационная модель города используется для дальнейшей аналитики городских территорий и получения информации о доступности городских кварталов. Универсальная модель города включает агрегированную информацию о сервисах, интермодальной доступности и городских кварталах.
3. Методы оценки городской обеспеченности различными типами сервисов с учетом нормативных требований и ценностных установок населения. Оценка обеспеченности производится при помощи итеративного алгоритма на графах, а также при помощи решения задачи линейной оптимизации.
4. Метод вычисления функции оценки оптимальности проектов мастер-планирования, основанный на ценностных установках населения и систем внешних ограничений. Метод основан на решении задачи оптимизации: необходимо найти оптимальную застройку, чтобы повысить обеспеченность. Задача решается при помощи генетического алгоритма, добавлена поддержка пользовательских сценариев.

Основные отличия от существующих решений:

- Метод генерации слоя городских кварталов учитывет тип землепользования, что позволяет определить ограничения для развития территории в контексте мастер-планирования.
- Универсальная информационная модель города может быть построена на открытых данных; наименьшая пространственная единица для анализа - квартал, что делает возможным аналитику в масштабах города.
- При оценки обеспеченности учитываются не только нормативные документы, но и ценностные утсановки населения.
- Генетический алгоритм для оптимизации застройки поддерживает пользовательские сценарии.
- Поддержка различных нормативных требований.

## Установка

Чтобы установить библиотеку **UrbanBlockNet** необходимо использовать `pip`:

```
pip install git+https://github.com/iduprojects/masterplanning
```

## Использование

Затем, необходимые классы импортируются из библиотеки:

```
from masterplan_tools import CityModel
```

Используются необходимые функции и модули:

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
2. [Агрегирование информации о городских кварталах](examples/2%20data_getter.ipynb) – как заполнить кварталы информацией о сервисах, содержащихся в них, а также сформировать матрицу интермодальной доступности между кварталами.
3. [Создание информационной модели города](examples/3%20city_model.ipynb) – как создать **информационную модель города** и визуализировать ее (чтобы убедиться в ее реальности и правильности).
4. [Оценка обеспеченности с помощью линейной оптимизации](examples/3a%20city_model%20lp_provision.ipynb) – как определить обеспеченность городскими сервисами.
5. [Оценка обеспеченности по итерационному алгоритму](examples/3b%20city_model%20iterative_provision.ipynb) – еще один пример оценки обеспеченности, но с использованием другого итерационного метода.
6. [Генетический алгоритм оптимизации требований к мастер-плану](examples/3d%20city_model%20genetic.ipynb) – как оптимизировать поиск требований к мастер-планированию для определенной территории или всего города по определенному сценарию.
7. [Балансировка параметров территории](examples/3c%20city_model%20balancer.ipynb) при увеличении жилой площади, не снижая качество жизни в городе.

Мы советуем начать с п. 3 [создание информационной модели города](examples/3%20city_model.ipynb), если вы скачали подготовленные нами [исходные данные](https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP).

## Документация

Более детальная информация и описание работы с UrbanBlockNet доступны в [документации](https://blocknet.readthedocs.io/en/latest/).

## Структура проекта

Последняя версия библиотеки доступна в main branch.

Она включает следующие модули и директории:

- [**masterplan_tools**](https://github.com/iduprojects/masterplanning/tree/main/masterplan_tools) - каталог с кодом фреймворка:
  - preprocessing - модуль для предпроцессинга
  - models - классы моделей
  - method - методы работы инструментов библиотеки
  - utils - модуль для статических единиц изменения
- [data](https://github.com/iduprojects/masterplanning/tree/main/data) - каталог с данными для экспериментов и тестов
- [tests](https://github.com/iduprojects/masterplanning/tree/main/tests) - каталог с единицами измерения и интеграционными тестами
- [examples](https://github.com/iduprojects/masterplanning/tree/main/examples) - каталог с примерами работы методов
- [docs](https://github.com/iduprojects/masterplanning/tree/main/docs) - каталог с RTD документацией

## Разработка

Для начала разработки библиотеки необходимо выполнить следующие действия:

1. Склонируйте репозиторий: `git clone https://github.com/iduprojects/masterplanning`.
2. Создайте виртуальную среду (опционально), поскольку библиотека требует точных версий пакетов: `python -m venv venv` и активируйте ее.
3. Установите библиотеку в редактируемом режиме: `python -m pip install -e '.[dev]' --config-settings editable_mode=strict`.
4. Установите pre-commit: `pre-commit install`.
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

## Публикации по инструментам библиотеки

Опубликованные работы:

1. [Morozov A. S. et al. Assessing the transport connectivity of urban territories, based on intermodal transport accessibility // Frontiers in Built Environment. – 2023. – Т. 9. – С. 1148708.](https://www.frontiersin.org/articles/10.3389/fbuil.2023.1148708/full)
2. [Morozov A. et al. Assessment of Spatial Inequality Through the Accessibility of Urban Services // International Conference on Computational Science and Its Applications. – Cham : Springer Nature Switzerland, 2023. – С. 270-286.](https://link.springer.com/chapter/10.1007/978-3-031-36808-0_18)

Принятые к публикации работы:

1. Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S. Digital Master Plan as a tool for generating territory development requirements // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
2. Natykin M.V., Budenny S., Zakharenko N. and Mityagin S.A. Comparison of solution methods the maximal covering location problem of public spaces for teenagers in the urban environment // International Conference on Advanced Research in Technologies, Information, Innovation and Sustainability 2023 – ARTIIS 2023
3. Natykin M.V., Morozov A., Starikov V. and Mityagin S.A. A method for automatically identifying vacant area in the current urban environment based on open source data // 12th International Young Scientists Conference in Computational Science – YSC 2023
4. Kontsevik G., Churiakova T., Markovskiy V., Antonov A. and Mityagin S. Urban blocks modelling method // 12th International Young Scientists Conference in Computational Science – YSC 2023
