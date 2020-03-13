# This is a testing set to make sure the intermediates are working
# 2020 Mar 9


import numpy as np
import pandas as pd
from MDAnalysis.coordinates.DCD import DCDReader
from numpy.linalg import norm
import os
import sys
import re


Nframes = 651
pdbdir="/data/nseelam04/pros_nowat_3ltp/v155d/template/equitrim.pdb"
cols = ["Atom Index", "Atom Type", "Res Name", "Chain", "Res ID", "X", "Y", "Z", "B", "Charge"]
tpsdir = "/data/nseelam04/pros_nowat_3ltp/v155d/tpsseed1_c2.3_c-1.4_fr51/tps_getp/"
tpswin = [3.1, 3.2]
sdir = "/usr/people/nseelam/labnotes/scripts/paperrepos/"
chosen_atoms = [("OMP", 1, "C6"), 
                ("OMP", 1, "CX"), 
                ("LYS", 72, "NZ"), 
                ("LYS", 72, "HZ1"), 
                ("LYS", 72, "HZ2"), 
                ("LYS", 72, "HZ3")]

model = EnzymeTrajectories(pdbdir, tpsdir, tpswin, sdir, chosen_atoms, Nframes)
model.get_tpstrajs()
model.get_accepted_trajs()
atom_pairs = [(135, 142), (55, 58), (55, 57), (55, 56), (135, 58), (135, 57), (135, 56)]
t = model.recompose_traj(model.trajs[1][-1], atom_pairs)
t2 = model.recompose_traj(model.trajs[1][-5], atom_pairs)

import matplotlib.pyplot as plt
plt.plot(t[:, 0]);plt.plot(t2[:, 0]); plt.show()



isrev = {'-1': True, '0': False}

pdb = pd.read_csv(pdbdir, 
                  skipfooter=1, 
                  skiprows=1, 
                  header=None,
                  delim_whitespace=True,
                  usecols=range(1,11), 
                  names=cols)


atom_list = {}
for atom in chosen_atoms:
    key = "-".join([str(i) for i in atom])
    value = pdb[(pdb["Res Name"] == atom[0]) & 
        (pdb["Res ID"] == atom[1]) &
        (pdb["Atom Type"] == atom[2])].index[0]
    atom_list.update({key: value})



trajs = {}
with open("trajsum.txt", "r") as f:
    x = f.readlines()

key = 1
value = list(filter(lambda y: "accepted 1" in y[0], [x[i:(i+2)] for i in range(0, len(x), 3)]))

# This is a bit crazy, but split on string to find the move that was accepted
accepts = [int(v[0].split(';')[0].split()[-1]) for v in value]
traj_recompose = [re.split(',', v[1][2:].split("#")[0].replace(" ", "").replace(';',',')) for v in value]
traj_recompose = list(map(lambda x: x[:-1], traj_recompose))
trajs.update({key: (accepts, traj_recompose)})


def get_trace(dcdfile, frame_start, frame_end, isrev):
    """
    Given a DCD, read all atoms and
    slice out the trace of interest.

    Requires the stitch forward + backward.
    """
    dcd = np.array(DCDReader(dcdfile))
    if isrev:
        dcd = dcd[::-1]
    return dcd #dcd[start_pos:end_pos, :, :]


def get_dist(atom1,atom2,axis=1):
    """Get the distance between 2 atoms. Expects 3 x Nframes"""
    return norm(atom1 - atom2, axis=axis)



# Recompose trajectories if the entire sequence is present


#OMP C6 - 135
#OMP CX - 142
#LYS NZ - 55
#LYS HZ1 - 56
#LYS HZ2 - 57
#LYS HZ3 - 58


