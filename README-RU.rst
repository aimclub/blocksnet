BlocksNet
=========

.. logo-start

.. figure:: https://i.ibb.co/QC9XD07/blocksnet.png
   :alt: BlocksNet logo

.. logo-end

|Documentation Status| |PythonVersion| |Black| |Readme_en|

.. description-start

**BlocksNet** это библиотека с открытым исходным кодом, включающая методы моделирования урбанизированных территорий для
задач генерации ценностно-ориентированных требований к мастер-планированию. Библиотека предназначена для формирования
универсальной информационной модели города, основанной на доступности городских кварталов. В библиотеке также представлены
инструменты для работы с информационной моделью города, которые позволяют: генерировать слой городских кварталов, рассчитывать
обеспеченность на основе нормативных требований, а также получить оптимальные требования к мастер-планированию территорий.

.. description-end

Преимущества
------------------

.. features-start

BlocksNet — это библиотека для моделирования сценариев развития города, например при создании мастер-плана,
поддерживающая следующие инструменты:

1. Метод генерации слоя городских кварталов предполагает разделение территории города на наименьшие элементы для анализа - кварталы.
   Метод генерации слоя городских кварталов учитывает дополнительные данные о землепользовании.
2. Метод генерации интермодального графа города на основе открытых данных. Интермодальный
   граф содержит информацию об общественном транспорте и пешеходных маршрутах
   для более детального анализа городской мобильности.
3. Универсальная информационная модель города используется для дополнительного
   анализа городских территорий и получения информации о доступности городских кварталов.
   В модели города содержится агрегированная информация о сервисах, зданиях, интермодальной доступности,
   типах сервисов и городских кварталах.
4. Метод оценки связности кварталов на основе интермодальной доступности.
5. Методы оценки городской обеспеченности различными типами сервисов с учетом нормативных требований и ценностных
   установок населения. Оценка обеспеченности производится при помощи итеративного алгоритма на графах, а также при помощи
   решения задачи линейной оптимизации.
6. Метод вычисления функции оценки оптимальности проектов мастер-планирования, основанный на ценностных установках
   населения и систем внешних ограничений. Метод основан на решении задачи оптимизации: необходимо найти оптимальную
   застройку, чтобы повысить обеспеченность. Задача решается при помощи генетического алгоритма, добавлена поддержка
   пользовательских сценариев.
7. Метод выявления свободной площади на основе открытых данных.

Основные отличия от существующих решений:

-  Метод генерации слоя **городских кварталов** учитывает тип землепользования,
   что позволяет определить ограничения для развития территории в
   контексте мастер-планирования.
-  Универсальная информационная **модель города** может быть построена на
   открытых данных; наименьшая пространственная единица для анализа - квартал,
   что делает возможным аналитику в масштабах города.
-  При оценке **обеспеченности** учитываются не только нормативные документы, но и
   ценностные установки населения.
-  Генетический алгоритм для оптимизации застройки поддерживает пользовательские **сценарии**.
-  Поддержка различных нормативных требований.

.. features-end

Установка
------------

.. installation-start

**BlocksNet** может быть установлен с помощью ``pip``:

::

   pip install git+https://github.com/iduprojects/blocksnet

.. installation-end

Использование
----------

.. use-start

Импортируйте необходимые классы из библиотеки ``blocksnet``:

::

   from blocksnet import City

Затем используйте необходимые классы и модули:

::

   city = City(
      blocks_gdf=blocks,
      adjacency_matrix=adj_mx
   )
   city.plot()

.. use-end

Более подробные примеры использования приведены в `примерах <#examples>`__.

Данные
----

Прежде чем использовать библиотеку, вы можете скачать `исходные данные <https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP>`__
и расположить их в директории ``examples/data``. Вы можете использовать собственные данные,
но они должны соответствовать спецификации согласно
`API documentation <https://blocknet.readthedocs.io/en/latest/index.html>`__.

Примеры
--------

Следующие примеры помогут ознакомиться с библиотекой::

1. Главный `пайплайн <examples/pipeline>`__ библиотеки. Включает полную инициализацию модели города ``City`` и генетическую оптимизацию ``Genetic``.
2. `Генерация слоя городских кварталов <examples/1%20blocks_generator.ipynb>`__ с помощью класса ``BlocksGenerator`` на основе данных о геометриях объектов города.
3. `Генерация интермодального графа <examples/2%20graph_generator.ipynb>`__ - с помощью класса ``GraphGenerator``.
   Включает построение матрицы смежности с помощью класса ``AdjacencyCalculator`` для данного слоя кварталов.
4. `Инициализация модели города <examples/city.ipynb>`__ и использование методов модели.
   Данный пример демонстрирует, как работать с моделью города ``City``, получать доступ к информации о типах сервисов ``ServiceType`` или кварталов
   ``Block``. Особенно полезно, если вы хотите принимать участие в разработке.
5. `Оценка обеспеченности <examples/3%20provision.ipynb>`__ - как оценить обеспеченность города выбранным типом сервиса ``ServiceType``,
6. `Метод оптимизации застройки <examples/4%20genetic.ipynb>`__ основанный на генетическом алгоритме.
   Целью метода является поиск отимальных требований к мастер-планированию территории квартала ``Block`` или всего ``City`` для выбранного сценария развития.
7. `Определение свободных площадей <examples/5%20vacant_area.ipynb>`__ выбранного городского квартала ``Block``.

Документация
-------------

Подробная информация и описание библиотеки BlocksNet представлены в
`документации <https://blocknet.readthedocs.io/en/latest/>`__.

Структура проекта
-----------------

Последняя версия библиотеки предсталена в ветке ``main``.

Репозиторий включает следующие директории и модули:

-  `blocksnet <https://github.com/iduprojects/blocksnet/tree/main/blocksnet>`__
   - директория с кодом библиотеки:

   -  preprocessing - модуль предобработки данных
   -  models - основные классы сущностей, используемые в библиотеке
   -  method - методы библиотеки для работы с моделью ``City``
   -  utils - модуль вспомогательных функций и констант

-  `tests <https://github.com/iduprojects/blocksnet/tree/main/tests>`__
   ``pytest`` тесты
-  `examples <https://github.com/iduprojects/blocksnet/tree/main/examples>`__
   примеры работы методов
-  `docs <https://github.com/iduprojects/blocksnet/tree/main/docs>`__ -
   ReadTheDocs документация

Разработка
----------

.. developing-start

Для начала разработки библиотеки необходимо выполнить следующие действия:

1. Клонировать репозиторий:
   ::

       $ git clone https://github.com/aimclub/blocksnet

2. (По желанию) Создать виртуальное окружение, так как библиотека требует точных версий пакетов:
   ::

       $ python -m venv venv

   Активировать виртуальное окружение, если оно было создано.

3. Установить библиотеку в режиме редактирования с dev-зависимостями:
   ::

       $ make install-dev

4. Установить pre-commit хуки:
   ::

       $ pre-commit install

5. Создать новую ветку на основе ``develop``:
   ::

       $ git checkout -b develop <new_branch_name>

6. Начать внесение изменений в своей новосозданной ветке, помня о том,
   чтобы не работать в ветке ``master``! Работайте с этой копией на вашем
   компьютере, используя Git для управления версиями.

7. Обновить
   `тесты <https://github.com/aimclub/blocksnet/tree/main/tests>`__
   в соответствии с вашими изменениями и запустить следующую команду:

   ::

         $ make test

   Убедитесь, что все тесты проходят успешно.

8. Обновить
   `документацию  <https://github.com/aimclub/blocksnet/tree/main/docs>`__
   и файл README в соответствии с вашими изменениями.

11. Когда вы закончите редактирование и локальное тестирование, выполните:

   ::

         $ git add modified_files
         $ git commit

чтобы записать ваши изменения в Git, затем отправьте их на GitHub с помощью:

::

      $ git push -u origin my-contribution

И, наконец, перейдите на веб-страницу вашего форка репозитория BlocksNet и нажмите 'Pull Request' (PR), чтобы отправить свои изменения на ревью разработчикам.

.. developing-end

Ознакомьтесь с разделом Contributing на ReadTheDocs для получения дополнительной информации.

Лицензия
-------

Проект имеет `лицензию BSD-3-Clause <./LICENSE>`__

Acknowledgments
---------------

.. acknowledgments-start

Библиотека была разработана как основная часть проекта Университета ИТМО № 622280
**"Библиотека алгоритмов машинного обучения для задач генерации ценностно-ориентированных
требований к мастер-планированию урбанизированных территорий"**.

Реализовано при финансовой поддержке Фонда поддержки проектов Национальной
технологической инициативы в рамках реализации "дорожной карты" развития
высокотехнологичного направления "Искусственный интеллект" на период до 2030
года (Договор № 70-2021-00187)

.. acknowledgments-end

Контакты
--------

.. contacts-start

Вы можете связаться с нами:

-  `НЦКР <https://actcognitive.org/o-tsentre/kontakty>`__ - Национальный Центр Когнитивных Разработок
-  `ИДУ <https://idu.itmo.ru/en/contacts/contacts.htm>`__ - Институт Дизайна и Урбанистики
-  `Tatiana Churiakova <https://t.me/tanya_chk>`__ - руководитель проекта
-  `Василий Стариков <https://t.me/vasilstar>`__ - ведущий разработчик

.. contacts-end

Публикации
-----------------------------

.. publications-start

Опубликованные работы:

-  `Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S.
   Digital Master Plan as a tool for generating territory development
   requirements // International Conference on Advanced Research in
   Technologies, Information, Innovation and Sustainability 2023 –
   ARTIIS 2023 <https://link.springer.com/chapter/10.1007/978-3-031-48855-9_4>`__
-  `Morozov A. S. et al. Assessing the transport connectivity of urban
   territories, based on intermodal transport accessibility // Frontiers
   in Built Environment. – 2023. – Т. 9. – С.
   1148708. <https://www.frontiersin.org/articles/10.3389/fbuil.2023.1148708/full>`__
-  `Morozov A. et al. Assessment of Spatial Inequality Through the
   Accessibility of Urban Services // International Conference on
   Computational Science and Its Applications. – Cham : Springer Nature
   Switzerland, 2023. – С.
   270-286. <https://link.springer.com/chapter/10.1007/978-3-031-36808-0_18>`__
-  `Natykin M.V., Morozov A., Starikov V. and Mityagin S.A. A method for
   automatically identifying vacant area in the current urban
   environment based on open source data // 12th International Young
   Scientists Conference in Computational Science – YSC 2023. <https://www.sciencedirect.com/science/article/pii/S1877050923020306>`__
-  `Natykin M.V., Budenny S., Zakharenko N. and Mityagin S.A. Comparison
   of solution methods the maximal covering location problem of public
   spaces for teenagers in the urban environment // International
   Conference on Advanced Research in Technologies, Information,
   Innovation and Sustainability 2023 – ARTIIS 2023. <https://link.springer.com/chapter/10.1007/978-3-031-48858-0_35>`__
-  `Kontsevik G., Churiakova T., Markovskiy V., Antonov A. and Mityagin
   S. Urban blocks modelling method // 12th International Young
   Scientists Conference in Computational Science – YSC 2023. <https://www.sciencedirect.com/science/article/pii/S1877050923020033>`__

.. publications-end

.. |Documentation Status| image:: https://readthedocs.org/projects/blocknet/badge/?version=latest
   :target: https://blocknet.readthedocs.io/en/latest/?badge=latest
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue
   :target: https://pypi.org/project/blocksnet/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Readme_en| image:: https://img.shields.io/badge/lang-en-red.svg
   :target: README.md
