Contributing
============

We welcome you to `check the existing
issues <https://github.com/aimclub/blocksnet/issues>`__ for bugs or
enhancements to work on. If you have an idea for an extension to BlocksNet,
please `file a new
issue <https://github.com/aimclub/blocksnet/issues/new>`__ so we can
discuss it.

Make sure to familiarize yourself with the project layout before making
any major contributions.

How to contribute
-----------------

.. include:: ../../../README.rst
   :start-after: .. developing-start
   :end-before: .. developing-end

(If it looks confusing to you, then look up the `Git
documentation <http://git-scm.com/documentation>`__ on the web.)

Before submitting your pull request
-----------------------------------

Before you submit a pull request for your contribution, please work
through this checklist to make sure that you have done everything
necessary so we can efficiently review and accept your changes.

If your contribution changes BlocksNet in any way:

-  Update the
   `documentation <https://github.com/aimclub/blocksnet/tree/main/docs>`__
   so all of your changes are reflected there.

-  Update the
   `README <https://github.com/aimclub/blocksnet/blob/main/README.md>`__
   if anything there has changed.

If your contribution involves any code changes:

-  Update the `project unit` tests to test your code changes.

-  Make sure that your code is properly commented with
   `docstrings <https://www.python.org/dev/peps/pep-0257/>`__ and
   comments explaining your rationale behind non-obvious coding
   practices.

If your contribution requires a new library dependency:

-  Double-check that the new dependency is easy to install via ``pip``
   or Anaconda and supports Python 3. If the dependency requires a
   complicated installation, then we most likely won't merge your
   changes because we want to keep BlocksNet easy to install.

Contribute to the documentation
-------------------------------
Take care of the documentation.

All the documentation is created with the Sphinx autodoc feature. Use ..
automodule:: <module_name> section which describes all the code in the module.

-  If a new package with several scripts:

   1. Go to `docs/source/BlocksNet <https://github.com/aimclub/blocksnet/tree/master/docs>`__ and create new  ``your_name_for_file.rst`` file.

   2. Add a Header underlined with “=” sign. It’s crucial.

   .. 3. Add automodule description for each of your scripts. ::

   ..     .. automodule:: blocksnet.your.first.script.path

   ..     .. automodule:: blocksnet.your.second.script.path
   ..     ...

   4. Add your_name_for_file to the toctree at ``docs/source/blocksnet/api/index.rst``

-  If a new module to the existed package:

    Most of the sections are already described in `docs/source/BlocksNet <https://github.com/aimclub/blocksnet/tree/master/docs>`__ , so you can:

   -  choose the most appropriate and repeat 3-d step from the previous section.
   -  or create a new one and repeat 2-3 steps from the previous section.

-  If a new function or a class to the existing module:

    Be happy. Everything is already done for you.

After submitting your pull request
----------------------------------

After submitting your pull request,
`Travis-CI <https://travis-ci.com/>`__ will automatically run unit tests
on your changes and make sure that your updated code builds and runs on
Python 3. We also use services that automatically check code quality and
test coverage.

Check back shortly after submitting your pull request to make sure that
your code passes these checks. If any of the checks come back with a red
X, then do your best to address the errors.

Acknowledgements
----------------

This document guide is based at well-written `TPOT Framework
contribution
guide <https://github.com/EpistasisLab/tpot/blob/master/docs_sources/contributing.md>`__.
