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


##OMP C6 - 135
#OMP CX - 142
#LYS NZ - 55
#LYS HZ1 - 56
#LYS HZ2 - 57
#LYS HZ3 - 58

# Dists
#C6-CX
#NZ-HZ1
#NZ-HZ2
#NZ-HZ3
#C6-HZ1
#C6-HZ2
#C6-HZ3


import gzip
import pickle as pkl
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import numpy as np, h5py 

prot = "wt"
ensmax = 5
ldir = "/data/nseelam04/pros_nowat_3ltp/trace_data/" + prot 
sdir1= "/usr/people/nseelam/labnotes/scripts/paperrepos/OMPDC_work/" + prot + "/"

for r in range(1,ensmax+1):
    print(prot, "Ensemble", r,)
    with gzip.open(ldir + "/" + prot + '_r' + str(r) + 'ws3.45_we5.pkl.gzip', 'rb') as f:
        traces = pkl.load(f)

    RC1 = np.zeros(shape = (traces.shape[0], 651)) * np.nan
    RC2 = np.zeros(shape = (traces.shape[0], 651)) * np.nan
    for i in range(traces.shape[0]):
        x = traces[i,:,0]
        y1 = traces[i,:,1] - traces[i,:,4]
        y2 = traces[i,:,2] - traces[i,:,5]
        y3 = traces[i,:,3] - traces[i,:,6]
        y = np.nanmax(np.vstack([y1, y2, y3]), axis=0)
        RC1[i, :] = x
        RC2[i, :] = y

    savemat(sdir1 + prot + "_C6CX_" + str(r) + "_ws3.45_we5.mat", mdict={'C6CX':RC1})
    savemat(sdir1 + prot + "_NZHC6_" + str(r) + "_ws3.45_we5.mat", mdict={'NZHC6':RC2})

    #savemat(sdir1 + prot + "_" + str(r) + "_ws3.45_we5.mat", mdict={'traces':traces})
    #plt.close('all')
    #for i in range(traces.shape[0]):
    #    x = traces[i,:,0]
    #    y1 = traces[i,:,1] - traces[i,:,4]
    #    y2 = traces[i,:,2] - traces[i,:,5]
    #    y3 = traces[i,:,3] - traces[i,:,6]
    #    y = np.nanmax(np.vstack([y1, y2, y3]), axis=0)
    #    plt.plot(x,y, alpha=0.05)
    #    plt.xticks(np.arange(1.5, 3.7, 0.2))
    #    plt.yticks(np.arange(-3.6, 1.4, 0.2))
    #    plt.xlim([1.49, 3.51])
    #    plt.ylim([-3.6, 1.2])
    #plt.savefig(sdir1 + "figures/" + prot + "_"+str(r) + ".png")


