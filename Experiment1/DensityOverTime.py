import numpy as np 
import matplotlib.pyplot as plt 
import glob

pos_files = glob.glob("ExpPosNPY/*.npy")
sorter = np.argsort([int(namestring.split("/")[-1].split("_")[0]) for namestring in pos_files])
pos_files = [pos_files[num] for num in sorter]

square_size = 50  # added to the centroid positions to get more accurate box size

densities = np.zeros(len(pos_files))
for i in range(len(pos_files)):
    positions = np.load(pos_files[i])
    pmax = np.max(positions)
    pmin = np.min(positions)

    box_area = (pmax-pmin)**2
    num_parts = len(positions)
    densities[i] = num_parts / box_area

plt.plot(densities)
plt.title("Number Density (Underestimation!) over Time")
plt.savefig("DensityOverTime.png")