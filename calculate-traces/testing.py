# This is a testing set to make sure the intermediates are working
# 2020 Mar 9


import numpy as np
import pandas as pd
from MDAnalysis.coordinates.DCD import DCDReader
from numpy.linalg import norm
import os
import sys
import re

pdbdir="/data/nseelam04/pros_nowat_3ltp/v155d/template/equitrim.pdb"
cols = ["Atom Index", "Atom Type", "Res Name", "Chain", "Res ID", "X", "Y", "Z", "B", "Charge"]
chosen_atoms = [("OMP", 1, "C6"), 
                ("OMP", 1, "CX"), 
                ("LYS", 72, "NZ"), 
                ("LYS", 72, "HZ1"), 
                ("LYS", 72, "HZ2"), 
                ("LYS", 72, "HZ3")]

isrev = {'-1': True, '0': False}

pdb = pd.read_csv(fname, 
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
    #if isrev:
    #    dcd = dcd[::-1, :, :]

    start_pos = len(dcd) - frame_end + frame_start
    end_pos = start_pos + frame_end
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


