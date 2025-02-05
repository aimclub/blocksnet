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
planning requirements. The library provides tools for generating an information
city model based on the accessibility of urban blocks.
The library also provides tools for working with the information
city model, which allows one: to assess urban network metrics such as connectivity
and centrality, to calculate service type provision based on regulatory
requirements and to obtain optimal requirements for master planning of territories.

.. description-end

Features
------------------

.. features-start

BlocksNet — a library for modeling urban development scenarios
(e.g. creating a master plan), supporting the following tools:

-  Method for generating a layer of urban blocks is the division of
   the territory into the smallest elements for the analysis of the
   urban area - blocks. The method of generating a layer of urban blocks
   is based on clustering algorithms taking into account additional data
   on land use.
-  Intermodal graph generator and accessibility matrix calculator based
   on `IduEdu <https://github.com/DDonnyy/IduEdu>`__ library.
-  The Universal Information City Model is used to further analyze urban
   areas and to obtain information on the accessibility of urban blocks.
   The City Model includes aggregated information on services and buildings,
   intermodal accessibility, service types hierarchy, and urban blocks.
-  Method for accessing the connectivity of the blocks based on intermodal
   accessibility.
-  Methods for assessing urban provision of different types of services
   with regard to normative requirements and value attitudes of the
   population. The estimation of provisioning is performed by iterative
   algorithm on graphs, as well as by solving linear optimization
   problem.
-  Method for computing the function for evaluating the optimality of
   master planning projects based on the value attitudes of the
   population and systems of external limitations. The method is based
   on solving an optimization problem: it is necessary to find an
   optimal development to increase the provision. The problem is solved
   with the help of simulated annealing algorithm, user scenarios
   support is added.
-  Method for identifying vacant areas based on open-data.
-  Land use prediction based on services within blocks.
-  Centrality and diversity assessments, spacematrix morphotypes identification
   method, integration metric assessment etc.

Main differences from existing solutions:

-  The method of generating a layer of **urban blocks** considers the
   type of land use, which makes it possible to define limitations for
   the development of the territory in the context of master planning.
-  The universal information **city model** can be built on open data;
   the smallest spatial unit for analysis is a block, which makes it
   possible to analyze on a city scale.
-  **Provision assessment** takes into account the competition element created
   between residents and services.
-  **Services optimization** algorithm based on simulated annealing supports
   user-defined **scenarios**.
-  Support for different regulatory requirements.
-  Pretty easy to use out of the box. The library is aimed to help students,
   so it balances between being friendly to non-programmers as well as useful
   and respective for advanced possible users and contributors.

.. features-end

Installation
------------

.. installation-start

**BlocksNet** can be installed with ``pip``:

::

   pip install blocksnet

.. installation-end

How to use
----------

.. use-start

Use the library by importing classes from ``blocksnet``:

::

   from blocksnet import City

Next, use the necessary classes and modules:

::

   city = City(
      blocks=blocks_gdf,
      acc_mx=acc_mx,
   )
   city.plot()

.. use-end

For more detailed use case see our `examples <#examples>`__.

Data
----

Before running the examples, one can use the data from `tests
section <#tests/data>`__
and place it in the ``examples/data`` directory. You can use your own
data, but it must follow the structure described in the
`API documentation <https://aimclub.github.io/blocksnet/>`__.

Examples
--------

Next examples will help to get used to the library:

1. Main `pipeline <examples/0%20pipeline>`__ of the library. Includes full ``City`` model initialization
   and ``Provision`` assessment.
2. `City blocks generating <examples/1%20blocks_generator.ipynb>`__ using ``BlocksGenerator`` class
   based on city geometries data.
3. `Accessibility matrix calculation <examples/2%20accessibility_processor.ipynb>`__ -
   using the ``AccessibilityProcessor`` class. Includes intermodal graph
   generating for given city blocks.
4. `City model initialization <examples/3%20city.ipynb>`__ and its methods usage.
   The example explains, how to work with ``City`` model, access ``ServiceType`` or
   ``Block`` information etc. Extremely helpful if you want to participate in the developing of **BlocksNet**.
5. `Provision assessment <examples/methods/provision.ipynb>`__ - how
   to assess provision of certain city ``ServiceType``,
6. `Development optimization method <examples/methods/annealing_optimizer.ipynb>`__ based on
   simulated annealing algorithm. The goal of the method is to optimize the search for master planning
   requirements for specific ``Block`` or the entire ``City`` in a specific scenario.
7. `Vacant area identifying <examples/vacant_area.ipynb>`__ for a certain city ``Block``.

Documentation
-------------

Detailed information and description of BlocksNet is available in
`documentation <https://aimclub.github.io/blocksnet/>`__.

Project Structure
-----------------

The latest version of the library is available in the ``main`` branch.

The repository includes the following directories and modules:

-  `blocksnet <https://github.com/aimclub/blocksnet/tree/main/blocksnet>`__
   - directory with the library code:

   -  preprocessing - data preprocessing module
   -  models - entities' classes used in library
   -  method - library tool methods based on ``City`` model
   -  utils - module containing utulity functions and consts

-  `tests <https://github.com/aimclub/blocksnet/tree/main/tests>`__
   ``pytest`` testing
-  `examples <https://github.com/aimclub/blocksnet/tree/main/examples>`__
   examples of how methods work
-  `docs <https://github.com/aimclub/blocksnet/tree/main/docs>`__ -
   documentation sources

Developing
----------

.. developing-start

To start developing the library, one must perform following actions:

1. Clone the repository:
   ::

       $ git clone https://github.com/aimclub/blocksnet

2. (Optional) Create a virtual environment as the library demands exact package versions:
   ::

       $ make venv

   Activate the virtual environment if you created one:
   ::

       $ source .venv/bin/activate

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
   and **README** according to your changes.

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

Check out the `Contributing <https://aimclub.github.io/blocksnet/blocksnet/contributing.html>`__ for more information.

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

This research is financially supported by the Foundation for National
Technology Initiative's Projects Support as a part of the roadmap
implementation for the development of the high-tech field of Artificial
Intelligence for the period up to 2030 (agreement 70-2021-00187)

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

Also, you are welcomed to our `issues <https://github.com/aimclub/blocksnet/issues>`__ section!

.. contacts-end

Publications
-----------------------------

.. publications-start

-  `Churiakova T., Starikov V., Sudakova V., Morozov A. and Mityagin S.
   Digital Master Plan as a tool for generating territory development
   requirements // International Conference on Advanced Research in
   Technologies, Information, Innovation and Sustainability 2023 –
   ARTIIS 2023 <https://link.springer.com/chapter/10.1007/978-3-031-48855-9_4>`__

.. publications-end

.. |Documentation Status| image:: https://github.com/aimclub/blocksnet/actions/workflows/documentation.yml/badge.svg?branch=main
   :target: https://aimclub.github.io/blocksnet/
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue
   :target: https://pypi.org/project/blocksnet/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Readme_ru| image:: https://img.shields.io/badge/lang-ru-yellow.svg
   :target: README-RU.rst
