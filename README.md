# Source Code for Bachelor Thesis Dumitrana Mihnea

## Setup Instructions
1. Install Python 3.10+
2. Install [CUDA Toolkit 12.6+](https://developer.nvidia.com/cuda-downloads)
3. Install [Poetry package manager](https://python-poetry.org/docs/#installing-with-the-official-installer)
4. Clone the repository
5. Run `poetry install` in the root directory of the repository

If using PyCharm, the IDE should prompt you to set the interpreter to the one created by poetry.  
If not, you can manually set the interpreter to the one created by poetry.

**NOTE:** In order to find the virtual environment location created by poetry, run  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`poetry env info` or `poetry config virtualenvs.path` in the root directory of the repository.

## Running the code

**NOTE:** Before running the program make sure you have at least 2GB of free space on your disk.  

To run the code, you can use the following command:

    `poetry run python main.py`

or manually run the main.py file using your IDE.
