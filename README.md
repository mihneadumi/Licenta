# Source Code for Bachelor Thesis Dumitrana Mihnea

## Setup Instructions
1. Install Python 3.10+
2. Install <a href="https://developer.nvidia.com/cuda-downloads" target="_blank">CUDA Toolkit 12.6+</a> for your operating system version.  
   **NOTE:** Make sure to select the correct version for your operating system and architecture (x86_64 or ARM64).  
   **NOTE:** If you are using a virtual machine, make sure to enable GPU passthrough in your VM settings.
3. Install <a href="https://python-poetry.org/docs/#installing-with-the-official-installer" target="_blank">Poetry package manager</a>
4. Clone the repository
5. Run `poetry install` in the root directory of the repository

If using PyCharm, the IDE should prompt you to set the interpreter to the one created by poetry.  
If not, you can manually set the interpreter to the one created by poetry.

**NOTE:** In order to find the virtual environment location created by poetry to set as interpreter, run  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`poetry env info` in the root directory of the repository.  
**NOTE:** The Virtualenv executable should be copied and introduced at interpreter settings in PyCharm.  
Example executable path: `C:\Users\"Your username"\AppData\Local\pypoetry\Cache\virtualenvs\licenta-LWv2Yd-r-py3.13\Scripts\python.exe`

## Running the code

**NOTE:** PLEASE RESTART YOUR IDE AFTER INSTALLING CUDA AND POETRY AND CHANGING PATH VARIABLES  
**NOTE:** Before running the program make sure you have at least 2GB of free space on your disk.  

To run the code, you can use the following command:

    poetry run python main.py

or manually run the main.py file using your IDE.
