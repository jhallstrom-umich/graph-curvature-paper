# Jonas Hallstrom, 03/24/2024s

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

""" Get data and organize it a little"""
file = 'frame_x_y_angle(orientation)_label.txt'

# frame, xpos, ypos, angle in degrees, ID
data = np.genfromtxt(file)

# NOTE: simulation and experiment have different length units, need to convert based on RDF peaks
length_convert = 1.7

frame_data = data[:, 0].astype("int")
num_frames = frame_data[-1]+1
x_data = length_convert*data[:, 1].astype("float")  # greater than 0
y_data = length_convert*data[:, 2].astype("float")  # greater than 0
box_size = max([max(x_data), max(y_data)])
x_data = x_data - box_size/2
y_data = y_data - box_size/2
y_data = -y_data  # invert to match IMSHOW images
box_size = 1.1*box_size
ang_data = data[:, 3].astype("float")
id_data = data[:, 4].astype("int")
num_ids = np.max(id_data)+1


by_frame = []  # data sorted by frame
current = 0
for f in range(num_frames):
    f_rows = np.where(frame_data == f)[0]  # all the data rows in frame f
    good_rows = [row for row in f_rows if not np.isnan(ang_data[row])]
    if f==0:
        print(np.shape(np.stack([ x_data[good_rows], y_data[good_rows] ], axis=1)))
    np.save("ExpPosNPY/"+str(f)+"_pos.npy", np.stack([ x_data[good_rows], y_data[good_rows] ], axis=1))
    np.save("ExpAngNPY/"+str(f)+"_ang.npy", ang_data[good_rows])
    np.save("ExpIDNPY/"+str(f)+"_id.npy", id_data[good_rows])
    #by_frame.append([])
    #for data_arr in [frame_data, x_data, y_data, ang_data, id_data]:
    #    by_frame[-1].append(data_arr[good_rows])
