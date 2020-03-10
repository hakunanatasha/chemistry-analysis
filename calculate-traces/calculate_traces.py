"""
2020 March 06

Calculate the traces of the TPS trajectories.
The following will identify, for a system + slice of trace
for the desired atoms.

NSNOTE: pydcd environment has MDAnalysis

TODO
2020.03.09: find_chosenatoms PDB columns can be a mapped iterable
"""

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

#OMP C6 - 135
#OMP CX - 142
#LYS NZ - 55
#LYS HZ1 - 56
#LYS HZ2 - 57
#LYS HZ3 - 58
isrev = {'-1': True, '0': False}

class EnzymeTrajectories:
    """
    The following class is intended to process the TPS trajectories
    of an enzyme, and draw out traces for storage
    """

    def __init__(self, 
                 pdbdir, 
                 tpsdir, 
                 tpswin, 
                 sdir, 
                 chosen_atoms,
                 Nframes=651):
        """
        pdbdir - location of the PDB
        tpsdir - location of the TPS trajectories
        tpswin - window length to extract
        sdir   - Location to save the trace files
        """

        self.pdbdir = pdbdir
        self.tpsdir = tpsdir
        self.tpswin = "ws" + str(tpswin[0]) + "_we" + str(tpswin[1])
        self.sdir = sdir
        self.chosen_atoms = chosen_atoms
        self.Nframes = Nframes

        self.pdb = self.read_pdb(pdbdir)

    def find_chosenatoms(self, cols=["Res Name", "Res ID", "Atom Type"]):
        """Given a list of chosen atoms, identify the location in PDB """
        self.atom_list = {}
        for atom in self.chosen_atoms:
            key = "-".join([str(i) for i in atom])
            value = self.pdb[(self.pdb[cols[0]] == atom[0]) & 
                     (self.pdb[cols[1]] == atom[1]) &
                     (self.pdb[cols[2]] == atom[2])].index[0]
            self.atom_list.update({key: value})

    def get_tpstrajs(self):
        """
        Slices trajectories based on TPS directories
        """
        tpsdirs = glob.glob(self.tpsdir + "*")
        self.tps_files = [t for t in tpsdirs if self.tpswin in t]

    def get_accepted_trajs(self):
        """
        Of the DCD files, Identify the accepted ensemble.
        Of the accepted ensemble, get the line after that tells me 
        how I constructed the new trajectory.
        """
        trajs = {}
        for replicate in self.tps_files:

            with open(self.tpsdir + replicate + "/trajsum.txt", "r") as f:
                x = f.readlines()
            
            key = int(replicate.split("r")[1][0])
            value = list(filter(lambda y: "accepted 1" in y[0], [x[i:(i+2)] for i in range(0, len(x), 3)]))
            
            # This is a bit crazy, but split on string to find the move that was accepted
            accepts = [int(v[0].split(';')[0].split()[-1]) for v in value]
            traj_recompose = [re.split(',', v[1][2:].split("#")[0].replace(" ", "").replace(';',',')) for v in value]
            traj_recompose = list(map(lambda x: x[:-1], traj_recompose))
            trajs.update({key: (accepts, traj_recompose)})

        self.trajs = trajs

    def recompose_traj(self, trajectory):
        """
        Given a trajectory, extract the frames of interest.
        """
        geometries = np.zeros(shape = (self.Nframes, len(self.distances)))
        for frame in range(0, len(trajectory), 4):
            tset = traj[(frame):(frame+4)]

            # Collect the start/stop of traj to build + traj existing
            formed_start, formed_end = [int(i) for i in tset[0].split(':')]
            calc_start, calc_end = [int(i) for i in tset[1].split(':')]
            irev = isrev[tset[2]]
            trajname = tset[3]

            if "prod" in trajname:
                trajname = "dcd/" + trajname + "_trim.dcd"

            if formed_start > formed_end:
                formed_start, formed_end = formed_end, formed_start

            if calc_start > calc_end:
                calc_start, calc_end = calc_end, calc_start

            if os.path.exists(trajname):
                
                dcd = get_trace(trajname, calc_start, calc_end, irev)
                
                fxn = lambda x: get_dist(dcd[:, x[0], :],dcd[:, x[1], :])
                r = np.array(list(map(fxn, rdists)))

                if irev:
                    cstart = r.shape[1] - (calc_end+1)
                    cend   = cstart + (calc_end+1-calc_start)
                    geometries[formed_start-1:formed_end, :] = r[:, cstart:cend].T
                else:
                    geometries[formed_start-1:formed_end, :] = r[:, calc_start:(calc_end+1)].T

        return frames

    @staticmethod
    def readpdb(fname):
        """
        Read canonical PDB and identify the atom ordering.
        Constructstruct.py converts to pdb
        """
        pdb = pd.read_csv(fname, 
                          skipfooter=1, 
                          skiprows=1, 
                          header=None,
                          delim_whitespace=True,
                          usecols=range(1,11), 
                          names=cols)
        return pdb

    @staticmethod
    def sliceatoms(pdb, chosenatoms):
        """
        Given certain canonical atoms you're looking into,
        return the position in the PDB that they come from.
        """
        aidx = []
        for atom in chosen_atoms:
            aidx.append(pdb[(pdb["Res Name"] == atom[0]) &
                       (pdb["Res ID"] == atom[1]) &
                       (pdb["Atom Type"] == atom[2])].index[0])
        return aidx

    @staticmethod
    def get_dist(atom1,atom2,axis=1):
        """Get the distance between 2 atoms. Expects 3 x Nframes"""
        return norm(atom1 - atom2, axis=axis)

    @staticmethod
    def get_trace(dcdfile, isrev):
        """
        Given a DCD, read all atoms and
        slice out the trace of interest.

        Requires the stitch forward + backward.
        """
        dcd = np.array(DCDReader(dcdfile))
        if isrev:
            dcd = dcd[::-1, :, :]

        return dcd



if __name__ == "__main__":
