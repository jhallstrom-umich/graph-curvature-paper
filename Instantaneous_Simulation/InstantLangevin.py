# Jonas Hallstrom, started 10/20/2023
# Modified from files from Chris Qian and the HOOMD-Blue Examples Repository
# Modified 05/28/2024 to record potential energy
# Cleaned up 10/17/24
# Commented and updated 09/09/26

# Imports
import itertools
import gsd.hoomd
import hoomd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def angle_to_quat(axis, angle):
    # Axis should be vector, angle should be in radians
    # Turns axis-angle representation of orientation into a quaternion
    if np.abs(axis[2]) != 1 or axis[1] != 0 or axis[0] != 0:
        # For this 2D simulation we only want z-axis rotations for orientation
        print("Recieved an axis that wasn't the z-axis for the NP orientations!")
        print(axis)
        quit()
    axis = np.array(axis) / np.sum( [axis[i]**2 for i in range(3)] )**(1/2)
    return [np.cos(angle/2)]+list(axis*np.sin(angle/2))

""" Initializing the system """
shared_seed = 0  # this seed is for numpy and hoomd randomization 

edge_N = 5  # number of LJ particles on square nanoparticle edge
edge_a = 50  # length of square nanoparticle
(epsilon, sigma, rcut) = (0.13, 26.5, 100)   # LJ parameters
LJ_mass = 1  # Mass of a single LJ particle
square_mass = LJ_mass*(edge_N**2)  # Mass of composite square nanoparticle

num_squares = 4000
density = 8e-5  # Number density

temp = 0.5
steps = 20e4
dt = 0.1

lang_gamma = 0.1  # langevin parameters
lang_gammar = 0.1

LJ_position = np.array( list(itertools.product(np.linspace(-edge_a/2, edge_a/2, edge_N), repeat=2)) )
LJ_position = np.concatenate((LJ_position, np.zeros([LJ_position.shape[0],1])),axis=1)  # position of LJ particles in a square NP


# set up initial positions of nanoparticles 
area = num_squares / density
L = np.sqrt(area)
k = int(np.ceil(np.sqrt(num_squares)))
if k%2 != 0:
    k += 1  # make it even so that row shifting thing in a few lines works
x = np.linspace(-L/2, L/2, 4+k)[2:-2]  # give them a little space on either side, actual spacing is L/(k+4)
# for big enough N, the above spacing should not make the initial condition significantly more dense/compressed
position = list(itertools.product(x, repeat=2))
position = np.array(position)
position = np.concatenate((position, np.zeros([position.shape[0],1])), axis=1)
position[::2, 0] += L / (2*(k+4))  # offset every other row so it becomes more of a centered square initial condition
print("{} square NPs, actual density of {:.2e}, goal density of {:.2e}".format(num_squares, num_squares / (L**2), density))
position_ids = np.arange(len(position))
np.random.seed(shared_seed)
np.random.shuffle(position_ids)  # so that later num_squares random positions are taken from the k**2 ones generated


# set up HOOMD initial frame
frame = gsd.hoomd.Frame()
frame.particles.types = ['Square', 'A']
frame.particles.N = num_squares
frame.particles.position = position[position_ids[:num_squares]]
frame.particles.orientation = [angle_to_quat([0, 0, 1], np.pi*np.random.rand()) for i in range(num_squares)]
frame.particles.typeid = [0] * num_squares
frame.particles.mass = [square_mass] * num_squares
frame.configuration.box = [L, L, 0, 0, 0, 0]
I = np.zeros(shape=(3, 3))
for r in LJ_position:
    I += LJ_mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
frame.particles.moment_inertia = [0, 0, I[2, 2]] * num_squares
with gsd.hoomd.open(name='initial.gsd', mode='w') as f:
    f.append(frame)


""" Randomize the system """
# The purpose of this initial short simulation is just to allow the NPs to move around somewhat
#  with long-range interactions turned off
rigid = hoomd.md.constrain.Rigid()
rigid.body['Square'] = {
    "constituent_types": ['A'] * (edge_N**2),
    "positions": LJ_position,
    "orientations": [(1.0, 0.0, 0.0, 0.0)]*(edge_N**2),
    }

gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=shared_seed)
sim.create_state_from_gsd(filename='initial.gsd')

rigid.create_bodies(sim.state)

s = sim.state.get_snapshot()

integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
integrator.rigid = rigid
cell = hoomd.md.nlist.Cell(buffer=2, exclusions=['body'])
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=sigma)
lj.r_cut[('A', 'A')] = sigma  # hard body LJ interactions for the randomization run, no long-range attraction
lj.params[('Square', ('A', 'Square'))] = dict(epsilon=0, sigma=0)
lj.r_cut[('Square', ('A', 'Square'))] = 0
integrator.forces.append(lj)
rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
langevin = hoomd.md.methods.Langevin(
    filter=rigid_centers_and_free,
    kT=temp)
randomize_gammas = 0.1  # not too much friction for this quick randomization simulation
langevin.gamma["Square"] = randomize_gammas
langevin.gamma_r["Square"] = [randomize_gammas]*3
integrator.methods.append(langevin)

sim.operations.integrator = integrator
sim.state.thermalize_particle_momenta(filter=rigid_centers_and_free, kT=temp)
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=rigid_centers_and_free)
sim.operations.computes.append(thermodynamic_properties)

sim.run(0)
kepp = thermodynamic_properties.kinetic_energy / num_squares
pepp = thermodynamic_properties.potential_energy / num_squares
print("Initialized per-particle kinetic energy {:.2e} and potential energy {:.2e}".format(kepp, pepp))

sim.operations.computes.append(thermodynamic_properties)
sim.run(1_000)
kepp = thermodynamic_properties.kinetic_energy / num_squares
pepp = thermodynamic_properties.potential_energy / num_squares
print("Randomized hard-LJ per-particle kinetic energy {:.2e} and potential energy {:.2e}".format(kepp, pepp))
hoomd.write.GSD.write(state=sim.state, filename='randomized.gsd', mode='wb')


""" Fully simulate the randomized system """
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=shared_seed)
sim.create_state_from_gsd(filename='randomized.gsd')

integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
integrator.rigid = rigid
cell = hoomd.md.nlist.Cell(buffer=2, exclusions=['body'])
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=sigma)
lj.r_cut[('A', 'A')] = rcut
lj.params[('Square', ('A', 'Square'))] = dict(epsilon=0, sigma=0)
lj.r_cut[('Square', ('A', 'Square'))] = 0
integrator.forces.append(lj)
rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
langevin = hoomd.md.methods.Langevin(
    filter=rigid_centers_and_free,
    kT=temp)
langevin.gamma["Square"] = lang_gamma
langevin.gamma_r["Square"] = [lang_gammar]*3
integrator.methods.append(langevin)

sim.operations.integrator = integrator

# Add writer
gsd_writer = hoomd.write.GSD(filename='trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(1_000),
                             mode='xb', filter=hoomd.filter.All()) #Type(["Square"]))
sim.operations.writers.append(gsd_writer)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=rigid_centers_and_free)

sim.operations.computes.append(thermodynamic_properties)
sim.run(0)
kepp = thermodynamic_properties.kinetic_energy / num_squares
pepp = thermodynamic_properties.potential_energy / num_squares
print("Randomized full-LJ simulation per-particle kinetic energy {:.2e} and potential energy {:.2e}".format(kepp, pepp))

# Add logger, which will put logged quantities in the gsd along with trajectory
logger = hoomd.logging.Logger()
logger.add(sim, quantities=["timestep", "tps"])
logger.add(thermodynamic_properties, quantities=["kinetic_temperature", "potential_energy", "num_particles"])
logger.add(lj, quantities=["energies"])
gsd_writer.logger = logger

# Now that everything is put together, run the simulation!
print(datetime.datetime.now())
#print(sim.timestep, langevin.kT.value, thermodynamic_properties.kinetic_temperature)
#print(sim.tps, thermodynamic_properties.potential_energy)
sim.run(steps)

hoomd.write.GSD.write(state=sim.state, mode='wb', filename='cooled.gsd')
