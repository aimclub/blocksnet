Installation from GitHub
========================

1. Clone the repository:
   ::

       git clone https://github.com/iduprojects/blocksnet

2. (Optional) Create a virtual environment as the library demands exact package versions:
   ::

       python -m venv venv

   Activate the virtual environment if you created one.

3. Install the library in editable mode with development dependencies:
   ::

       python -m pip install -e '.[dev]' --config-settings editable_mode=strict

4. Install pre-commit hooks:
   ::

       pre-commit install

5. Create a new branch based on **develop**:
   ::

       git checkout -b develop <new_branch_name>

6. Make your code changes.

7. Commit your changes, push the new branch, and create a pull request into **develop**.
