# Jonas Hallstrom, 10/16/2023, based on Freud documentation example
# Updated 06/18/2024

import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import hoomd
import glob

pos_files = glob.glob("../SimPosNPY/*.npy")
sorter = np.argsort([int(namestring.split("/")[-1].split("_")[0]) for namestring in pos_files])
pos_files = [pos_files[num] for num in sorter]
center_positions = np.load(pos_files[-1])
center_positions = np.column_stack([center_positions, np.zeros(len(center_positions))])
print(center_positions)
print(np.shape(center_positions))

file = "../trajectory.gsd"
with gsd.hoomd.open(file) as trajectory:
    snap=trajectory[-1]
box = snap.configuration.box
system = (box, center_positions)

r_max, r_min = (160, 50)
rdf = freud.density.RDF(bins=(r_max-r_min)//2, r_max=r_max, r_min=r_min)
system = (box, center_positions)
aq = freud.locality.AABBQuery.from_system(system)
nlist = aq.query(center_positions, {"r_max":r_max, "exclude_ii":True}).toNeighborList()
rdf.compute(aq, neighbors=nlist, reset=False)

max_idx = np.argmax(rdf.rdf)
print(rdf.bin_centers)

print(rdf)
plt.plot(rdf.bin_centers, rdf.rdf)
plt.axvline(rdf.bin_centers[max_idx], linestyle=":")
plt.axvline(1.15*rdf.bin_centers[max_idx], linestyle=":", color="tab:green")
plt.axvline(1.2*rdf.bin_centers[max_idx], linestyle=":", color="tab:green")
plt.axvline(2*np.sin(np.pi*79/2/180)*rdf.bin_centers[max_idx], linestyle=":", color="tab:red")
plt.axvline(2*np.cos(np.pi*79/2/180)*rdf.bin_centers[max_idx], linestyle=":", color="tab:red")
plt.axvline(2*rdf.bin_centers[max_idx], linestyle=":", color="tab:red")
plt.title("Peak: {}. Blue peak, Red predicted next peaks,\nand green cutoffs at 1.15 and 1.2 peak".format(rdf.bin_centers[max_idx]) )
plt.savefig("FreudRDF_Peak{}.png".format(int(rdf.bin_centers[max_idx])))

