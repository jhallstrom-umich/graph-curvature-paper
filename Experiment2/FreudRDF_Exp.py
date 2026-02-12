# Jonas Hallstrom, 10/16/2023, based on Freud documentation example
# Updated 06/18/2024

import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import hoomd
import glob

pos_files = glob.glob("ExpPosNPY/*.npy")
sorter = np.argsort([int(namestring.split("/")[-1].split("_")[0]) for namestring in pos_files])
pos_files = [pos_files[num] for num in sorter]
center_positions = np.load(pos_files[-1])
print(center_positions)
print(np.shape(center_positions))

bmax = np.max(center_positions)
bmin = np.min(center_positions)
center_positions = center_positions - (bmax+bmin)/2
center_positions = np.column_stack([center_positions, np.zeros(len(center_positions))])
L = bmax - bmin
box = freud.box.Box(1.5*L, 1.5*L, 0)  # 1.2 because we DONT want PBC bonds for the experiment

r_max, r_min = (160, 50)
rdf = freud.density.RDF(bins=100, r_max=r_max, r_min=r_min)
system = (box, center_positions)
aq = freud.locality.AABBQuery.from_system(system)
nlist = aq.query(center_positions, {"r_max":r_max, "exclude_ii":True}).toNeighborList()
rdf.compute(aq, neighbors=nlist, reset=False)

peak_indices = [i for i, height in enumerate(rdf.rdf) if height>10]
#peak_indices = [idx for idx in peak_indices if (idx-1 not in peak_indices)]
top_peaks = rdf.bin_centers[peak_indices]
top_peaks = np.sort(top_peaks)
print(top_peaks)
max_idx = np.argmax(rdf.rdf)
bond_length = top_peaks[0]
angle_guess = []
for peak in top_peaks[1:]:
    angle1 = 2*np.arcsin( (peak/(2*bond_length)) ) * (180/np.pi)
    angle2 = 2*np.arccos( (peak/(2*bond_length)) ) * (180/np.pi)
    angle_guess.append([angle1, angle2])
print(angle_guess)
angle_guess = ["({:.1f}, {:.1f})".format(*pair) for pair in angle_guess]

print(rdf)
plt.plot(rdf.bin_centers, rdf.rdf)
plt.axvline(rdf.bin_centers[max_idx], linestyle=":")
plt.title("Peak (dotted line): {}".format(rdf.bin_centers[max_idx]) )
plt.savefig("FreudRDF.png")
plt.show()
