import argparse
import numpy as np
import matplotlib.pyplot as plt

def initGrid(rows, cols):
    grid = 2*np.random.randint(2, size=rows*cols)-1
    grid = grid.reshape(rows, cols)

    return grid

def switchSpin(grid, i, j):
    grid[i][j] = -1 * grid[i][j]

def calculateEnergy(grid, i, j):
    rows, cols = grid.shape
    if i == 0 and j == 0:
        #top-left corner
        energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i][j+1])
    elif i == 0 and j == cols-1:
        #top-right corner
        energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i][j-1])
    elif i == rows-1 and j == 0:
        #bottom-left corner
        energy_ij = -grid[i][j] * (grid[i-1][j]+grid[i][j+1])
    elif i == rows-1 and j == cols-1:
        #bottom-right corner
        energy_ij = -grid[i][j] * (grid[i-1][j]+grid[i][j-1])
    elif i == 0:
        #upper row
        energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i][j+1]+grid[i][j-1])
    elif i == rows-1:
        #bottom row
        energy_ij = -grid[i][j] * (grid[i-1][j]+grid[i][j+1]+grid[i][j-1])
    elif j == 0:
        #left-most column
        energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i-1][j]+grid[i][j+1])
    elif j == cols-1:
        #right-most column
        energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i-1][j]+grid[i][j-1])
    else:
        energy_ij = -grid[i][j] * (grid[i+1][j]+grid[i-1][j]+grid[i][j+1]+grid[i][j-1])
    
    return energy_ij

def runIsingModel(grid, temperature, steps):
    rows, cols = grid.shape
    for _ in range(steps):
        i = np.random.randint(rows)
        j = np.random.randint(cols)
        energy = calculateEnergy(grid, i, j)
        if energy > 0:
            switchSpin(grid, i, j)
        elif energy < 0:
            r = np.random.uniform()
            if r < np.exp(2*energy/temperature):
                switchSpin(grid, i, j)

def showGrid(start_grid, grid, steps, temp):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(start_grid, interpolation="none")
    ax2.imshow(grid, interpolation="none")
    
    fig.suptitle(f"Spin lattice vs Spin lattice after {steps} steps\nTemperature={temp}", fontsize=16, fontweight="bold")

    plt.show()

if __name__ == "__main__":
    rows = 200
    cols = 200
    steps = 1000000
    temperature = 2

    parser = argparse.ArgumentParser(description="Simulate an Ising model using Monte Carlo method.")
    parser.add_argument("--file", help="File with the starting grid, in a \'1 1 -1 ... -1 1\' format. If rows and columns aren't specified, the program will attempt to make it a square matrix. If only one is specified, the program will attempt to calculate the other one.")
    parser.add_argument("--rows", type=int, help="Number of rows for the lattice. If not specified, it defaults to 200.")
    parser.add_argument("--cols", type=int, help="Number of columns for the lattice. If not specified, it defaults to 200.")
    parser.add_argument("--steps", type=int, help="Number of Monte Carlo steps. If not specified, it defaults to 10e6.")
    parser.add_argument("--temp", type=int, help="Sets the temperature. If not specified, it defaults to 2.")
    args = parser.parse_args()
    
    if args.file:
        with open(args.file) as file:
            grid = np.array(file.read().split(" ")).astype(int)
            length = grid.shape[0]
            if length < 4:
                print("The matrix is too small.")
                exit(-1)
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

    if args.steps:
        steps = args.steps
    if args.temp:
        temperature = args.temp

    start_grid = np.copy(grid)
    runIsingModel(grid, temperature, steps)
    showGrid(start_grid, grid, steps, temperature)