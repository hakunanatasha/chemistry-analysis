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
import pickle as pkl
import gzip
sys.path.insert(0, "/usr/people/nseelam/labnotes/scripts/paperrepos/chemistry-analysis/calculate-traces")
from calculate_traces import *

prot = "v155d"
print("Protein=", prot)



pdbdir="/data/nseelam04/pros_nowat_3ltp/" + prot + "/template/equitrim.pdb"
sdir = "/data/nseelam04/pros_nowat_3ltp/trace_data/" + prot + "/"

if prot == "v155d":
    tpsdir = "/data/nseelam04/pros_nowat_3ltp/v155d/tpsseed1_c2.3_c-1.4_fr51/tps_getp/"
    atom_pairs = [(135, 142), (55, 58), (55, 57), (55, 56), (135, 58), (135, 57), (135, 56)]
elif prot == "s127a":
    tpsdir = "/data/nseelam04/pros_nowat_3ltp/s127a/tpsseed1_c2.1_c-1_fr14/tps_getp/"
    atom_pairs = [(124, 131), (55, 56), (55, 57), (55, 58), (124, 56), (124, 57), (124, 58)]
else:
    tpsdir = "/data/nseelam04/pros_nowat_3ltp/wt/tpsseed1_set_c2.1_c-1.2_fr53/tpsgetp_0.1win/"
    atom_pairs = [(45, 52), (34, 37), (34, 36), (34, 35), (45, 37), (45, 36), (45, 35)]


Nframes = 651
cols = ["Atom Index", "Atom Type", "Res Name", "Chain", "Res ID", "X", "Y", "Z", "B", "Charge"]
tpswin = [3.45, 5] #[3.1, 3.2]

chosen_atoms = [("OMP", 1, "C6"), 
                ("OMP", 1, "CX"), 
                ("LYS", 72, "NZ"), 
                ("LYS", 72, "HZ1"), 
                ("LYS", 72, "HZ2"), 
                ("LYS", 72, "HZ3")]


model = EnzymeTrajectories(pdbdir, tpsdir, tpswin, sdir, chosen_atoms, cols, Nframes)

for key in model.trajkeys:
    print("Ensemble = r", key)
    traces = model.get_traces_replicate(key, atom_pairs)
    with gzip.open(sdir + prot + "_r" + str(key) + model.tpswin + ".pkl.gzip", "wb") as f:
        pkl.dump(traces, f)


import gzip
import pickle as pkl
import numpy
import matplotlib.pyplot as plt

with gzip.open('wt_r1ws3.45_we5.pkl.gzip', 'rb') as f:
    traces = pkl.load(f)


for i in range(traces.shape[0]):
    x = traces[i,:,0]
    y = traces[i,:,1] - traces[i,:,4]
    plt.plot(x,y, alpha=0.05)


plt.show()