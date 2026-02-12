# Jonas Hallstrom, 03/24/2024

import glob
import numpy as np
import matplotlib.pyplot as plt
import gsd.hoomd
import hoomd
import itertools

def angle_to_quat(axis, angle):
    # Axis should be vector, angle should be in radians
    if np.abs(axis[2]) != 1 or axis[1] != 0 or axis[0] != 0:
        print("Recieved an axis that wasn't the z-axis for the NP orientations!")
        print(axis)
        quit()
    axis = np.array(axis) / np.sum( [axis[j]**2 for j in range(3)] )**(1/2)
    return [np.cos(angle/2)]+list(axis*np.sin(angle/2))

def quat_to_zaxis_angle(quat):
    # turns a quaternion into an angle, assuming that the quaternion is for an
    #  orientation about the zaxis
    partial_norm = np.sqrt(quat[1]**2 + quat[2]**2 + quat[3]**2)
    (ax, ay, az) = quat[1:] / partial_norm
    theta = 2*np.arctan2(partial_norm, quat[0])

    if ax!=0 or ay!=0 or az*az != 1.0:
        print("quat_to_angle error, not a 2d orientation!")
        print(ax, ay, az, theta)
        return None
    else:
        return az*theta

""" Get data and organize it the same way as experiment """

positions = []
angles = []
for j in range(len(glob.glob("WaveData/trajectory?.gsd"))):  # assumes not more than 10
    print(j)
    filename = "WaveData/trajectory{}.gsd".format(j)
    traj = gsd.hoomd.open(filename)
    num_frames = len(traj)

    center_ids = [i for i, x in enumerate(traj[0].particles.types) if x in ["Square", "NI_Square"]]
    for i in range(num_frames):
        snap = traj[i]
        center_positions = [pos for idx, pos in enumerate(snap.particles.position) if snap.particles.typeid[idx] in center_ids]
        orientations = [180*quat_to_zaxis_angle(quat)/np.pi  for idx, quat in enumerate(snap.particles.orientation) if snap.particles.typeid[idx] in center_ids]

        positions.append(np.array(center_positions))
        angles.append(orientations)

num_frames = len(positions)
print(num_frames)
for f in range(num_frames):
    num_parts = len(positions[f])
    np.save("SimPosNPY/"+str(f)+"_pos.npy", positions[f][:, :2])
    np.save("SimAngNPY/"+str(f)+"_ang.npy", angles[f])
    np.save("SimIDNPY/"+str(f)+"_id.npy", np.arange(num_parts))
