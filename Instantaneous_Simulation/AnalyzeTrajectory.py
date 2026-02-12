# Jonas Hallstrom, 10/21/2023

import numpy as np
import matplotlib.pyplot as plt
import gsd.hoomd

filename = "trajectory.gsd"
traj = gsd.hoomd.open(filename)
print(len(traj))

print(traj[0].log.keys())
for key in traj[0].log.keys():
    print(key)
    print(traj[0].log[key])
    print(traj[-1].log[key])
    print()

num_particles = int(traj[0].log["md/compute/ThermodynamicQuantities/num_particles"][0])
particle_potential = np.array([traj[i].log["particles/md/pair/LJ/energies"] for i in range(len(traj))])
print(num_particles, np.shape(particle_potential), sum([1 if energy==0 else 0 for energy in particle_potential[-1]]))
print(sum(particle_potential[-1, 0:num_particles]))

timesteps = np.array([traj[i].log["Simulation/timestep"] for i in range(len(traj))])
kinetic_temperature = np.array([traj[i].log["md/compute/ThermodynamicQuantities/kinetic_temperature"] for i in range(len(traj))])
potential_energy = np.array([traj[i].log["md/compute/ThermodynamicQuantities/potential_energy"] for i in range(len(traj))])

plt.plot(timesteps, kinetic_temperature)
plt.savefig("KineticTemperature.png")
plt.clf()

plt.plot(timesteps, potential_energy)
plt.savefig("PotentialEnergy.png")
plt.clf()

plt.plot(kinetic_temperature, potential_energy)
plt.savefig("PotentialE_vs_KineticT.png")
plt.clf()
