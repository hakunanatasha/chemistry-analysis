"""
2020 March 06

Calculate the traces of the TPS trajectories.
The following will identify, for a system + slice of trace
for the desired atoms.

NSNOTE: pydcd environment has MDAnalysis
"""

import numpy as np
import pandas as pd
from MDAnalysis.coordinates.DCD import DCDReader
from numpy.linalg import norm
import os
import sys

pdbdir="/data/nseelam04/pros_nowat_3ltp/v155d/template/equitrim.pdb"
cols = ["Atom Index", "Atom Type", "Res Name", "Chain", "Res ID", "X", "Y", "Z", "B", "Charge"]
chosen_atoms = [("OMP", 1, "C6"), 
                ("OMP", 1, "CX"), 
                ("LYS", 72, "NZ"), 
                ("LYS", 72, "HZ1"), 
                ("LYS", 72, "HZ2"), 
                ("LYS", 72, "HZ3")]


class EnzymeTrajectories:
    """
    The following class is intended to process the TPS trajectories
    of an enzyme, and draw out traces for storage
    """

    def __init__(self, pdbdir, tpsdir, tpswin, sdir):
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

        self.pdb = self.read_pdb(pdbdir)

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
            value = list(filter(lambda y: "accepted 1" in y[0], [x[i:i+2] for i in range(0, len(x), 3)]))
            
            # This is a bit crazy, but split on string to find the move that was accepted
            value = [int(v.split(';')[0].split()[-1]) for v in value]
            trajs.update({key: value})
        
        self.trajs = trajs

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
    def get_trace(dcdfor, dcdrev, isrev=False):
        """
        Given a DCD, read all atoms and
        slice out the trace of interest.

        Requires the stitch forward + backward.
        """
        dcdfor = np.array(DCDReader(dcdfor))
        dcdrev = np.array(DCDReader(dcdrev))
        return dcdfor, dcdrev



if __name__ == "__main__":
