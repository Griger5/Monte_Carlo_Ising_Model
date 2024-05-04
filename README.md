# Monte Carlo Ising Model
This program simulates a simplified 2D Ising model with the Monte Carlo method. It uses NumPy to load a 2D spin lattice and Matplotlib to visualize the result. The program is also able to use PyCUDA for (optional) GPU acceleration. <br>

In every step of the simulation, one spin is selected at random. Its energy is calculated with the given formula:

$$E_{i,j} = -s_{i,j}*(s_{i+1,j}+s_{i-1,j}+s_{i,j+1}+s_{i,j-1})$$

Where:

-  ___i, j___ are the indexes of the spin's row and column, respectively

-  ___s___ is the value of the given spin (either 1 or -1)

-  ___E___ is the energy of the square

If the calculated energy is greater than 0, the spin will switch. Otherwise, it will switch with the probability given by the following formula:

$$P(switch_{i,j})=e^{(2*E_{i,j})/T}$$

Where ___T___ is the systems temperature.
## Installation:
First, clone the repository.
```sh
git  clone  https://github.com/Griger5/Monte_Carlo_Ising_Model.git
```
Then create and activate the virtual environment for the program.
```sh
python  -m  venv  .venv
```
#### On Windows:
PowerShell:
```sh
Set-ExecutionPolicy  -ExecutionPolicy  RemoteSigned  -Scope  CurrentUser
.venv\Scripts\Activate.ps1
```
cmd:
```sh
.venv\Scripts\activate.bat
```
#### On Linux:
bash/zsh:
```sh
source  .venv/bin/activate
```
csh/tcsh:
```sh
source  .venv/bin/activate.csh
```
---
After that is done, install all the required libraries using:
```sh
pip  install  -r  requirements.txt
```
## Usage
This program supports user command line arguments. When none are present, it will run with the default parameters listed below. If a GPU and CUDA drivers are detected, the simulation will run on the GPU. Otherwise it simply runs on the CPU.
#### Default parameters:
- Number of rows: 200
- Number of columns: 200
- Number of steps in the simulation: 10^6 (10^8 with GPU present)
- Temperature: 2.0
- Spin values: Initialized randomly

When in the virtual environment, you can run the program with:
```sh
python  ising_model.py [--args n] [--flags]
```
### Args:
#### \-\-file [filepath]
Loads the lattice from a file. The file should contain a flattened lattice (as in: '1 -1 1 -1 1 1 -1 1...'). If rows and columns are not specified by the user, the program will attempt to turn the loaded lattice into a square.
#### \-\-rows [int]
Specify the number of rows for the spin lattice.
#### \-\-cols [int]
Specify the number of columns for the spin lattice.
#### \-\-steps [int]
Specify the number of Monte Carlo steps in the simulation.
#### \-\-temp [double]
Specify the temperature for the simulation.
### Flags:
#### \-\-anim
When present, produces an animation of the simulation instead of a "before and after" image.
#### \-\-no_gpu
When present, forces the program to compute on CPU.