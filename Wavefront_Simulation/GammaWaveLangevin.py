# Jonas Hallstrom, started 10/20/2023
# Modified from files from Chris Qian and the HOOMD-Blue Examples Repository
# Modified 01/10/2024 to write images and velocities
# Modified 02/29/2024 to replace some noninteracting NPs with interacting NPs over time
# Modified 07/06/2024 to make "noninteracting NPs" have much higher friction (gamma values)
#  rather than not interact with other NPs
# Cleaned up 10/16/24
# Commented and updated 09/09/26

# Imports
import itertools
import gsd.hoomd
import hoomd
import numpy as np
import datetime

def angle_to_quat(axis, angle):
    # Axis should be vector, angle should be in radians
    # Turns axis-angle representation of orientation into a quaternion    
    if np.abs(axis[2]) != 1 or axis[1] != 0 or axis[0] != 0:
        print("Recieved an axis that wasn't the z-axis for the NP orientations!")
        print(axis)
        quit()
    axis = np.array(axis) / np.sum( [axis[i]**2 for i in range(3)] )**(1/2)
    return [np.cos(angle/2)]+list(axis*np.sin(angle/2))

""" Initializing the system """
# nanoparticle and LJ parameters
edge_N = 5
edge_a = 50
(epsilon, sigma, rcut) = (0.13, 26.5, 100)
LJ_mass = 1  # Mass of a single LJ particle
square_mass = LJ_mass*(edge_N**2)  # Mass of composite square nanoparticle

num_squares = 4000
density = 8e-5  # Number density

num_waves = 7
if num_waves>9:
    print("num_waves should be 9 at most")
    quit(0)
temps = [0.5]*(num_waves+1)
steps = [20e3]*(num_waves+1)
end_steps = 40e3
steps[-1] += end_steps
dt = 0.1

lang_gamma = 0.1
lang_gammar = 0.1
NI_lang_gamma = 100
NI_lang_gammar = 1000

LJ_position = np.array( list(itertools.product(np.linspace(-edge_a/2, edge_a/2, edge_N), repeat=2)) )
LJ_position = np.concatenate((LJ_position, np.zeros([LJ_position.shape[0],1])),axis=1)  # position of LJ particles in a square NP


area = num_squares / density
L = np.sqrt(area)
wave_m = [1]*num_waves  # should be of length 1 less than temps and steps
wave_b = np.linspace(1, -1, num_waves+1)[1:]
wave_b = [L*num for num in wave_b]
print(wave_b)

# The whole simulation will be done in num_waves+1 parts
for idx in range(num_waves+1):
    print(wave_b[idx-1])
    if idx == 0:
        # They all start as noninteracting squares
        k = int(np.ceil(np.sqrt(num_squares))+0.01)
        x = np.linspace(-L/2, L/2, 2+k)[1:-1]  # give them a little space on either side
        position = list(itertools.product(x, repeat=2))
        position = np.array(position)
        position = np.concatenate((position, np.zeros([position.shape[0],1])), axis=1)
        print(num_squares, num_squares / (L**2), density, "\n")
        position_ids = np.arange(len(position))
        np.random.shuffle(position_ids)
        position = position[position_ids]

        orientation = [angle_to_quat([0, 0, 1], np.pi*np.random.rand()) for i in range(num_squares)]

        typeid = [0] * num_squares
        mass = [square_mass] * num_squares

    else:
        # use the final positions of the previous stage to initialize this stage
        prev_frame = gsd.hoomd.open('cooled{}.gsd'.format(idx-1))
        snap = prev_frame[0]
        types = snap.particles.types
        prev_typeid = snap.particles.typeid
        prev_pos = snap.particles.position
        prev_ori = snap.particles.orientation

        np_mask = [i for i in range(len(prev_typeid)) if types[prev_typeid[i]] in ['Square', 'NI_Square']]

        position = prev_pos[np_mask]
        orientation = prev_ori[np_mask]

        typeid = []
        mass = []
        for i in range(len(np_mask)):
            if types[prev_typeid[np_mask[i]]]=="Square":
                # If you're an interacting square, stay that way.
                typeid.append(2)
                mass.append(square_mass)
            else:
                (x, y, z) = position[i]
                if y > wave_m[idx-1] * x + wave_b[idx-1]:
                    # If you're NI and in the new zone that turns on interactions,
                    #  become an interacting square
                    typeid.append(2)
                    mass.append(square_mass)
                else:
                    typeid.append(0)
                    mass.append(square_mass)

    frame = gsd.hoomd.Frame()
    frame.particles.types = ['NI_Square', 'NI_A', 'Square', 'A']  # the order of these is hardcoded into logic above
    frame.particles.N = num_squares
    frame.particles.position = position[0:num_squares]
    frame.particles.orientation = orientation
    frame.particles.typeid = typeid
    frame.particles.mass = mass
    frame.configuration.box = [L, L, 0, 0, 0, 0]

    I = np.zeros(shape=(3, 3))
    for r in LJ_position:
        I += LJ_mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
    frame.particles.moment_inertia = [0, 0, I[2, 2]] * num_squares

    with gsd.hoomd.open(name='initial{}.gsd'.format(idx), mode='w') as f:
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
    rigid.body['NI_Square'] = {
        "constituent_types": ['NI_A'] * (edge_N**2),
        "positions": LJ_position,
        "orientations": [(1.0, 0.0, 0.0, 0.0)]*(edge_N**2),
        }

    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu, seed=1)
    sim.create_state_from_gsd(filename='initial{}.gsd'.format(idx))

    rigid.create_bodies(sim.state)

    s = sim.state.get_snapshot()

    if idx==0:
        random_steps = 1e4
    else:
        # For all stages besides the first, we don't want to actually randomize but just instantiate the rigid bodies 
        random_steps = 0

    random_rcut = sigma  # "hard-body" lennard jones interaction, only short-ranged repulsion
    integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
    integrator.rigid = rigid
    cell = hoomd.md.nlist.Cell(buffer=0.1, exclusions=['body'])
    lj = hoomd.md.pair.LJ(nlist=cell)
    # A and NI_A have the same interactions with each other
    lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[('A', 'A')] = random_rcut
    lj.params[('NI_A', ('NI_A', 'A'))] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[('NI_A', ('NI_A', 'A'))] = random_rcut
    # Both Square and NI_Square do not interact with anything
    lj.params[('Square', ('A', 'Square', 'NI_A', 'NI_Square'))] = dict(epsilon=0, sigma=0)
    lj.r_cut[('Square', ('A', 'Square', 'NI_A', 'NI_Square'))] = 0
    lj.params[('NI_Square', ('A', 'NI_A', 'NI_Square'))] = dict(epsilon=0, sigma=0)
    lj.r_cut[('NI_Square', ('A', 'NI_A', 'NI_Square'))] = 0
    integrator.forces.append(lj)
    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    langevin = hoomd.md.methods.Langevin(
        filter=rigid_centers_and_free,
        kT=temps[0])
    if idx==0:
        langevin.gamma["Square"] = 0.1
        langevin.gamma_r["Square"] = [0.1]*3
        langevin.gamma["NI_Square"] = 0.1
        langevin.gamma_r["NI_Square"] = [0.1]*3
    else:
        langevin.gamma["Square"] = lang_gamma
        langevin.gamma_r["Square"] = [lang_gammar]*3
        langevin.gamma["NI_Square"] = NI_lang_gamma
        langevin.gamma_r["NI_Square"] = [NI_lang_gammar]*3
    integrator.methods.append(langevin)

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(filter=rigid_centers_and_free, kT=temps[0])
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=rigid_centers_and_free)

    sim.operations.computes.append(thermodynamic_properties)

    sim.run(random_steps)
    if idx==0:
        hoomd.write.GSD.write(state=sim.state, filename='randomized.gsd', mode='wb')
    else:
        hoomd.write.GSD.write(state=sim.state, filename='initial_rigid{}.gsd'.format(idx), mode='wb')



    """ Fully simulate the next stage of the system """
    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu, seed=1)
    if idx==0: 
        sim.create_state_from_gsd(filename='randomized.gsd')
    else:
        sim.create_state_from_gsd(filename='initial_rigid{}.gsd'.format(idx))

    integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
    integrator.rigid = rigid
    cell = hoomd.md.nlist.Cell(buffer=2, exclusions=['body'])
    lj = hoomd.md.pair.LJ(nlist=cell)
    # A and NI_A have the same interactions with each other
    lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[('A', 'A')] = rcut
    lj.params[('NI_A', ('NI_A', 'A'))] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[('NI_A', ('NI_A', 'A'))] = rcut
    # Both Square and NI_Square do not interact with anything
    lj.params[('Square', ('A', 'Square', 'NI_A', 'NI_Square'))] = dict(epsilon=0, sigma=0)
    lj.r_cut[('Square', ('A', 'Square', 'NI_A', 'NI_Square'))] = 0
    lj.params[('NI_Square', ('A', 'NI_A', 'NI_Square'))] = dict(epsilon=0, sigma=0)
    lj.r_cut[('NI_Square', ('A', 'NI_A', 'NI_Square'))] = 0
    integrator.forces.append(lj)
    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    langevin = hoomd.md.methods.Langevin(
        filter=rigid_centers_and_free,
        kT=temps[0])
    langevin.gamma["Square"] = lang_gamma
    langevin.gamma_r["Square"] = [lang_gammar]*3
    langevin.gamma["NI_Square"] = NI_lang_gamma
    langevin.gamma_r["NI_Square"] = [NI_lang_gammar]*3
    integrator.methods.append(langevin)

    sim.operations.integrator = integrator

    # Add writer
    gsd_writer = hoomd.write.GSD(filename='trajectory{}.gsd'.format(idx),
                                 trigger=hoomd.trigger.Periodic(1_000),
                                 mode='xb', filter=hoomd.filter.All())
    sim.operations.writers.append(gsd_writer)

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=rigid_centers_and_free)

    sim.operations.computes.append(thermodynamic_properties)

    # Add logger, which will put logged quantities in the gsd along with trajectory
    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=["timestep", "tps"])
    logger.add(thermodynamic_properties, quantities=["kinetic_temperature", "pressure", "potential_energy", "num_particles"])
    gsd_writer.logger = logger

    sim.run(0)
    print(datetime.datetime.now())
    print(sim.timestep, langevin.kT.value)
    print(sim.tps, thermodynamic_properties.potential_energy)
    print()
    langevin.kT = temps[idx]
    sim.run(steps[idx])

    hoomd.write.GSD.write(state=sim.state, mode='wb', filename='cooled{}.gsd'.format(idx))

    gsd_writer.flush()
