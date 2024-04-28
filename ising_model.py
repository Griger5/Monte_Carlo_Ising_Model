import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess

if subprocess.check_output("nvidia-smi"):
    from pathlib import Path
    import pycuda.autoinit
    import pycuda.driver as drv
    gpu_present = True
else:
    gpu_present = False

def initGrid(rows, cols):
    grid = 2*np.random.randint(2, size=rows*cols)-1
    grid = grid.reshape(rows, cols)

    return grid

def addBorderWithZeros(array):
    rows, cols = array.shape
    b_array = np.zeros(((rows+2),(cols+2)))
    for i in range(1, rows+1):
        b_array[i][1:cols+1] = array[i-1]

    return b_array

def switchSpin(grid, i, j):
    grid[i][j] = -1 * grid[i][j]

def calculateEnergy(grid, i, j):
    energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i-1][j]+grid[i][j+1]+grid[i][j-1])
    
    return energy_ij

def runIsingModel(grid, temperature, steps):
    rows, cols = grid.shape
    for _ in range(steps):
        i = np.random.randint(1, rows-1)
        j = np.random.randint(1, cols-1)
        energy = calculateEnergy(grid, i, j)
        if energy > 0:
            switchSpin(grid, i, j)
        elif energy < 0:
            r = np.random.uniform()
            if r < np.exp(2*energy/temperature):
                switchSpin(grid, i, j)

def runIsingOnGPU(grid, temperature, steps):
    rows, cols = grid.shape

    grid = grid.astype(np.int32)
    rows = np.array(rows).astype(np.int32)
    cols = np.array(cols).astype(np.int32)
    steps = np.array(steps).astype(np.int32)
    temp = np.array(temperature).astype(np.float64)

    root_dir = Path(__file__).resolve().parent
    ptx_path = str(root_dir)+"\\gpu_assets\\compiled_kernel\\ising_model.ptx"

    mod = drv.module_from_file(ptx_path)

    ising = mod.get_function("runIsingModel")
    ising(drv.InOut(grid), drv.In(rows), drv.In(cols), drv.In(temp), drv.In(steps), block=(1024,1,1), grid=(20,1))

    return grid

def animIsingModel(grid, temperature, steps):
    rows, cols = grid.shape
    saved_grids = [(np.copy(grid), 0)]
    for k in range(steps):
        i = np.random.randint(1, rows-1)
        j = np.random.randint(1, cols-1)
        energy = calculateEnergy(grid, i, j)
        if energy > 0:
            switchSpin(grid, i, j)
        elif energy < 0:
            r = np.random.uniform()
            if r < np.exp(2*energy/temperature):
                switchSpin(grid, i, j)
        
        if k/100000 < 1:
            if k%10000 == 0:
                saved_grids.append((np.copy(grid), k))
        else:
            if k%100000 == 0:
                saved_grids.append((np.copy(grid), k))

    saved_grids.append((np.copy(grid), steps))
    
    return saved_grids

def showGrid(start_grid, grid, steps, temp):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    cmap = plt.get_cmap("viridis")

    ax1.imshow(start_grid[1:-1,1:-1], interpolation="none", cmap=cmap)
    ax2.imshow(grid[1:-1,1:-1], interpolation="none", cmap=cmap)
    
    fig.suptitle(f"Spin lattice vs Spin lattice after {steps} steps\nTemperature={temp}", fontsize=16, fontweight="bold")

    plt.show()

def updateGrid(i, fig, ax, frames, temp):
    ax.set_array(frames[i][0][1:-1,1:-1])
    fig.suptitle(f"Spin lattice after {frames[i][1]} steps\nTemperature={temp}", fontsize=16, fontweight="bold")
    
    return ax,

def animateGrid(frames, temp):
    fig = plt.figure()
    cmap = plt.get_cmap("viridis")
    ax = plt.imshow(frames[0][0][1:-1,1:-1], interpolation="none", cmap=cmap)
    fig.suptitle(f"Spin lattice after 0 steps\nTemperature={temp}", fontsize=16, fontweight="bold")
    anim = animation.FuncAnimation(fig, updateGrid, frames=len(frames), interval=200, fargs=(fig, ax, frames, temp), repeat=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an Ising model using Monte Carlo method.")
    parser.add_argument("--file", help="File with the starting grid, in a \'1 1 -1 ... -1 1\' format. If rows and columns aren't specified, the program will attempt to make it a square matrix. If only one is specified, the program will attempt to calculate the other one.")
    parser.add_argument("--rows", type=int, help="Number of rows for the lattice. If not specified, it defaults to 200.")
    parser.add_argument("--cols", type=int, help="Number of columns for the lattice. If not specified, it defaults to 200.")
    parser.add_argument("--steps", type=int, help="Number of Monte Carlo steps. If not specified, it defaults to 10e6.")
    parser.add_argument("--temp", type=int, help="Sets the temperature. If not specified, it defaults to 2.")
    parser.add_argument("--anim", action=argparse.BooleanOptionalAction, help="Flag. Constructs an animation of the simulation.")
    parser.add_argument("--no_gpu", action=argparse.BooleanOptionalAction, help="Flag. Forces the program to calculate on CPU.")
    args = parser.parse_args()

    if args.no_gpu:
        gpu_present = False

    rows = 200
    cols = 200
    steps = 100_000_000 if gpu_present else 1_000_000
    temperature = 2
    
    if args.file:
        with open(args.file) as file:
            grid = np.array(file.read().split(" ")).astype(int)
            length = grid.shape[0]
            if args.rows and args.cols:
                if length == args.rows*args.cols:
                    rows, cols = args.rows, args.cols
                else:
                    print("Wrong dimensions, can't construct a matrix with the given file.")
                    exit(-1)
            elif args.rows:
                if length/args.rows%1 != 0:
                    print("Wrong dimensions, can't construct a matrix with the given file.")
                    exit(-1)
                else: 
                    rows, cols = args.rows, length/args.rows
            elif args.cols:
                if length/args.cols%1 != 0:
                    print("Wrong dimensions, can't construct a matrix with the given file.")
                    exit(-1)
                else: 
                    rows, cols = length/args.cols, args.cols
            elif np.sqrt(length)%1 != 0:
                print("Unable to construct a square matrix, specify the size.")
                exit(-1)
            else:
                rows, cols = np.sqrt(length), np.sqrt(length)

        grid = grid.reshape(int(rows), int(cols))             
    else:
        if args.rows:
            rows = args.rows
        if args.cols:
            cols = args.cols

        grid = initGrid(rows, cols)

    grid = addBorderWithZeros(grid)

    if args.steps:
        steps = args.steps
    if args.temp:
        temperature = args.temp

    if args.anim:
        anim_frames = animIsingModel(grid, temperature, steps)
        animateGrid(anim_frames, temperature)
    else:
        start_grid = np.copy(grid)
        if gpu_present:
            grid = runIsingOnGPU(grid, temperature, steps)
        else:
            runIsingModel(grid, temperature, steps)
        showGrid(start_grid, grid, steps, temperature)