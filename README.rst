BlocksNet
=========

.. logo-start

.. figure:: https://i.ibb.co/QC9XD07/blocksnet.png
   :alt: BlocksNet logo

.. logo-end

|Documentation Status| |PythonVersion| |Black| |Readme_ru|

.. description-start

**BlocksNet** is an open-source library that includes methods of
modeling urbanized areas for the generation of value-oriented master
planning requirements. The library is provided for generating a
universal information city model based on the accessibility of urban
blocks. The library also provides tools for working with the information
city model, which allow: to generate a layer of urban blocks, to
calculate provisioning based on regulatory requirements, to obtain
optimal requirements for master planning of territories.

.. description-end

Features
------------------

.. features-start

BlocksNet — a library for modeling urban development scenarios
(e.g. creating a master plan), supporting the following tools:

1. The method of generating a layer of urban blocks is the division of
   the territory into the smallest elements for the analysis of the
   urban area - blocks. The method of generating a layer of urban blocks
   is based on clustering algorithms taking into account additional data
   on land use.
2. The Universal Information City Model is used to further analyze urban
   areas and to obtain information on the accessibility of urban blocks.
   The City Model includes aggregated information on services,
   intermodal accessibility and urban blocks.
3. Methods for assessing urban provision of different types of services
   with regard to normative requirements and value attitudes of the
   population. The estimation of provisioning is performed by iterative
   algorithm on graphs, as well as by solving linear optimization
   problem.
4. A method for computing the function for evaluating the optimality of
   master planning projects based on the value attitudes of the
   population and systems of external limitations. The method is based
   on solving an optimization problem: it is necessary to find an
   optimal development to increase the provision. The problem is solved
   with the help of genetic algorithm, user scenarios support is added.

Main differences from existing solutions:

-  The method of generating a layer of **urban blocks** considers the
   type of land use, which makes it possible to define limitations for
   the development of the territory in the context of master planning.
-  The universal information **city model** can be built on open data;
   the smallest spatial unit for analysis is a block, which makes it
   possible to analyze on a city scale.
-  Not only normative documents are taken into account when assessing
   **provision**, but also the value attitudes of the population.
-  Genetic algorithm for optimization of development supports
   user-defined **scenarios**.
-  Support for different regulatory requirements.

.. features-end

Installation
------------

.. installation-start

**BlocksNet** can be installed with ``pip``:

::

   pip install git+https://github.com/iduprojects/blocksnet

.. installation-end

How to use
----------

.. use-start

Then use the library by importing classes from ``blocksnet``:

::

   from blocksnet import City

Next, use the necessary functions and modules:

::

   city = City(
      blocks_gdf=blocks,
      adjacency_matrix=adj_mx
   )
   city.plot()

.. use-end

For more detailed use case see our `examples <#examples>`__.

Data
----

Before running the examples, you must download the `input
data <https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP>`__
and place it in the ``examples/data`` directory. You can use your own
data, but it must follow the structure described in the
`API documentation <https://blocknet.readthedocs.io/en/latest/index.html>`__.

Examples
--------

Next examples will help to get used to the library:

1. `City blocks generating <examples/1%20blocks_cutter.ipynb>`__ - city
   blocks generating according to landuse and buildings clustering.
2. `Aggregating city blocks
   information <examples/2%20data_getter.ipynb>`__ - how to fill blocks
   with aggregated information and also generate the accessibility
   matrix between blocks.
3. `City model creation <examples/3%20city_model.ipynb>`__ - how to
   create the **city model** and visualize it (to make sure it is real
   and correct).
4. `Linear optimization provision
   assessment <examples/3a%20city_model%20lp_provision.ipynb>`__ - how
   to assess provision of certain city service type.
5. `Iterative algorithm provision
   assessment <examples/3b%20city_model%20iterative_provision.ipynb>`__
   - another example of how to assess provision, but using different
   iterative method.
6. `Genetic algorithm master plan
   optimization <examples/3d%20city_model%20genetic.ipynb>`__ - how to
   optimize the search for master planning requirements for a specific
   area or the entire city in a specific scenario.
7. `Balancing territory
   parameters <examples/3c%20city_model%20balancer.ipynb>`__ - how to
   increase certain territory population without decreasing the quality
   of life of the city.

We advice to start with `city model
creation <examples/3%20city_model.ipynb>`__ example, if you downloaded
the `input
data <https://drive.google.com/drive/folders/1xrLzJ2mcA0Qn7FG0ul8mTkfzKolvUoiP>`__
we prepared.

Documentation
-------------

Detailed information and description of BlocksNet is available in
`documentation <https://blocknet.readthedocs.io/en/latest/>`__.

Project Structure
-----------------

The latest version of the library is available in the main branch.

The repository includes the following directories and modules:

-  `blocksnet <https://github.com/iduprojects/blocksnet/tree/main/blocksnet>`__
   - directory with the framework code:

   -  preprocessing - data preprocessing module
   -  models - entities' classes used in library
   -  method - library tool methods on ``City`` model
   -  utils - module for helping functions and consts

-  `tests <https://github.com/iduprojects/blocksnet/tree/main/tests>`__
   ``pytest`` testing
-  `examples <https://github.com/iduprojects/blocksnet/tree/main/examples>`__
   examples of how methods work
-  `docs <https://github.com/iduprojects/blocksnet/tree/main/docs>`__ -
   ReadTheDocs documentation

Developing
----------

.. developing-start

To start developing the library, one must perform following actions:

1. Clone the repository:
   ::

       $ git clone https://github.com/aimclub/blocksnet

2. (Optional) Create a virtual environment as the library demands exact package versions:
   ::

       $ python -m venv venv

   Activate the virtual environment if you created one.

3. Install the library in editable mode with development dependencies:
   ::

       $ make install-dev

4. Install pre-commit hooks:
   ::

       $ pre-commit install

5. Create a new branch based on ``develop``:
   ::

       $ git checkout -b develop <new_branch_name>

6. Start making changes on your newly created branch, remembering to
   never work on the ``master`` branch! Work on this copy on your
   computer using Git to do the version control.

7. Update
   `tests <https://github.com/aimclub/blocksnet/tree/main/tests>`__
   according to your changes and run the following command:

   ::

         $ make test

   Make sure that all tests pass.

8. Update the
   `documentation <https://github.com/aimclub/blocksnet/tree/main/docs>`__
   and README files according to your changes.

11. When you're done editing and local testing, run:

   ::

         $ git add modified_files
         $ git commit

to record your changes in Git, then push them to GitHub with:

::

          $ git push -u origin my-contribution

Finally, go to the web page of your fork of the BlocksNet repo, and click
'Pull Request' (PR) to send your changes to the maintainers for review.

.. developing-end

Check out the Contributing on ReadTheDocs for more information.

License
-------

The project has `BSD-3-Clause license <./LICENSE>`__

Acknowledgments
---------------

.. acknowledgments-start

The library was developed as the main part of the ITMO University
project #622280 **“Machine learning algorithms library for the tasks of
generating value-oriented requirements for urban areas master
planning”**

.. acknowledgments-end

Contacts
--------

.. contacts-start

You can contact us:

-  `NCCR <https://actcognitive.org/o-tsentre/kontakty>`__ - National
   Center for Cognitive Research
-  `IDU <https://idu.itmo.ru/en/contacts/contacts.htm>`__ - Institute of
   Design and Urban Studies
-  `Tatiana Churiakova <https://t.me/tanya_chk>`__ - project manager
-  `Vasilii Starikov <https://t.me/vasilstar>`__ - lead software engineer

.. contacts-end

Publications
-----------------------------

.. publications-start

Published:

1. `Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S.
   Digital Master Plan as a tool for generating territory development
   requirements // International Conference on Advanced Research in
   Technologies, Information, Innovation and Sustainability 2023 –
   ARTIIS 2023 <https://link.springer.com/chapter/10.1007/978-3-031-48855-9_4>`__
2. `Morozov A. S. et al. Assessing the transport connectivity of urban
   territories, based on intermodal transport accessibility // Frontiers
   in Built Environment. – 2023. – Т. 9. – С.
   1148708. <https://www.frontiersin.org/articles/10.3389/fbuil.2023.1148708/full>`__
3. `Morozov A. et al. Assessment of Spatial Inequality Through the
   Accessibility of Urban Services // International Conference on
   Computational Science and Its Applications. – Cham : Springer Nature
   Switzerland, 2023. – С.
   270-286. <https://link.springer.com/chapter/10.1007/978-3-031-36808-0_18>`__
4. `Natykin M.V., Morozov A., Starikov V. and Mityagin S.A. A method for
   automatically identifying vacant area in the current urban
   environment based on open source data // 12th International Young
   Scientists Conference in Computational Science – YSC 2023. <https://www.sciencedirect.com/science/article/pii/S1877050923020306>`__
5. `Natykin M.V., Budenny S., Zakharenko N. and Mityagin S.A. Comparison
   of solution methods the maximal covering location problem of public
   spaces for teenagers in the urban environment // International
   Conference on Advanced Research in Technologies, Information,
   Innovation and Sustainability 2023 – ARTIIS 2023. <https://link.springer.com/chapter/10.1007/978-3-031-48858-0_35>`__

Accepted:

1. Kontsevik G., Churiakova T., Markovskiy V., Antonov A. and Mityagin
   S. Urban blocks modelling method // 12th International Young
   Scientists Conference in Computational Science – YSC 2023

.. publications-end

.. |Documentation Status| image:: https://readthedocs.org/projects/blocknet/badge/?version=latest
   :target: https://blocknet.readthedocs.io/en/latest/?badge=latest
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue
   :target: https://pypi.org/project/blocksnet/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Readme_ru| image:: https://img.shields.io/badge/lang-ru-yellow.svg
   :target: README-RU.md
