# Source Code for Bachelor Thesis Dumitrana Mihnea

## Setup Instructions
    1. Install Python 3.9+
    2. install poetry package manager from https://python-poetry.org/docs/#installing-with-the-official-installer 
    3. Clone the repository
    4. Run `poetry install` in the root directory of the repository

If using PyCharm, the IDE should prompt you to set the interpreter to the one created by poetry.  
If not, you can manually set the interpreter to the one created by poetry.

**NOTE:** In order to find the virtual environment location created by poetry, run  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`poetry env info` or `poetry config virtualenvs.path` in the root directory of the repository.

## Running the code
To run the code, you can use the following command:

    `poetry run python main.py`

or manually run the main.py file using your IDE.