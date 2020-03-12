# 2020 March 12
# Natasha Seelam

# Collect the TPS trajectories

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
atom_pairs = [(135, 142), (55, 58), (55, 57), (55, 56), (135, 58), (135, 57), (135, 56)]


model = EnzymeTrajectories(pdbdir, tpsdir, tpswin, sdir, chosen_atoms, Nframes)

key = model.trajkeys[0]

traces = model.get_traces_replicate(key, atom_pairs)


import matplotlib.pyplot as plt

f = plt.figure(figsize=(10,12))

for i in range(545):
    x = traces[i, :, 0]
    y = traces[i, :, 1] - traces[i, :, 4]
    plt.plot(x,y, alpha = 0.05)


plt.xticks(np.arange(1.5,3.6,0.2))
plt.yticks(np.arange(-3.2,1.4,0.2))
plt.xlim([1.5, 3.5])
plt.ylim([-3.2, 1.2])
plt.show()