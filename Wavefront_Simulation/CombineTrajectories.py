
import gsd.hoomd
import glob

f = gsd.hoomd.open(name='CombinedTrajectory.gsd', mode='w')
for idx in range(len(glob.glob("WaveData/trajectory*.gsd"))):
    traj = gsd.hoomd.open('WaveData/trajectory{}.gsd'.format(idx))
    f.extend(traj)
