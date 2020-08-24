"""
2020 June 08
Modifying the windows to match ensemble length
 
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
import glob
import os
import sys
import re
    
#OMP C6 - 135
#OMP CX - 142
#LYS NZ - 55
#LYS HZ1 - 56
#LYS HZ2 - 57
#LYS HZ3 - 58
isrev = {'-1': True, '0': False}
#cols = ["Atom Index", "Atom Type", "Res Name", "Chain", "Res ID", "X", "Y", "Z", "B", "Charge"]

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
                 cols,
                 Nframes=651,
		         ensembles=None):
        """
        pdbdir - location of the PDB
        tpsdir - location of the TPS trajectories
        tpswin - window length to extract
        sdir   - Location to save the trace files
        ensembles - number of MCs to consider
        """

        self.pdbdir = pdbdir
        self.tpsdir = tpsdir
        self.tpswin = "ws" + str(tpswin[0]) + "_we" + str(tpswin[1])
        self.sdir = sdir
        self.chosen_atoms = chosen_atoms
        self.Nframes = Nframes

        # Assign the PDB
        print("\n01. Get PBB:")
        self.pdb = self.readpdb(pdbdir, cols)

        # Assign indices to necessary atoms
        print("\n02. Get Atom index:")
        self.atom_index = self.find_chosenatoms(self.pdb, self.chosen_atoms)

        # Assign the tps-files of interest.
        print("\n03. Get TPS directories:")
        self.get_tpstrajs()
	
        if ensembles is not None:
            substrings = ['_r' + str(j) + '_' for j in range(1, ensembles + 1)]
            self.tps_files = [t for t in self.tps_files if any(x in t for x in substrings)]
	
        print("\n04. Get accepted Trajectories:") 
        self.get_accepted_trajs()

    def get_tpstrajs(self):
        """
        Slices trajectories based on TPS directories
        """
        tpsdirs = glob.glob(self.tpsdir + "*")
        self.tps_files = [t for t in tpsdirs if self.tpswin in t]
        print("\tRetrieved TPS files for", self.tpswin)

    def get_accepted_trajs(self):
        """
        Of the DCD files, Identify the accepted ensemble.
        Of the accepted ensemble, get the line after that tells me 
        how I constructed the new trajectory.
        """
        accepted_id = {}
        trajs = {}
        direcs = {}
        for replicate in self.tps_files:

            with open(replicate + "/trajsum.txt", "r") as f:
                x = f.readlines()
            
            key = int(replicate.split("set_r")[-1].split("_")[0])
            value = list(filter(lambda y: "accepted 1" in y[0], [x[i:(i+2)] for i in range(0, len(x), 3)]))
            
            # This is a bit crazy, but split on string to find the move that was accepted
            accepts = [int(v[0].split(';')[0].split()[-1]) for v in value]
            traj_recompose = [re.split(',', v[1][2:].split("#")[0].replace(" ", "").replace(';',',')) for v in value]
            traj_recompose = list(map(lambda x: x[:-1], traj_recompose))
            accepted_id.update({key: accepts})
            trajs.update({key: traj_recompose})
            direcs.update({key: replicate})

        self.trajs = trajs
        self.trajkeys = list(trajs.keys())
        self.direcs = direcs
        print("\tExtracted accepted trajectories for", self.tpswin)

    def get_traces_replicate(self, trajkey, atom_pairs, report=5):
        """
        Given a particular trajectory key, extract all necessary features.
        """
        trajs = self.trajs[trajkey]
        traces = np.zeros(shape = (len(trajs), self.Nframes, len(atom_pairs))) * np.nan

        for trajidx, trajectory in enumerate(trajs): 
            if (trajidx+1)%report == 0:
                print("Completed", trajidx+1, "/", len(trajs))
            traces[trajidx, :, :] = self.recompose_traj(self.direcs[trajkey], self.Nframes, trajectory, atom_pairs)

        return traces

    @staticmethod
    def recompose_traj(direc, Nframes, trajectory, atom_pairs):
        """
        Given a trajectory, extract all the frames from the correct DCDs.

        2020.03.12: Currently compatible with ONLY extracting distances.
        """
        geometries = np.zeros(shape = (Nframes, len(atom_pairs)))*np.nan
        for frame in range(0, len(trajectory), 4):
            tset = trajectory[(frame):(frame+4)]

            # Collect the start/stop of traj to build + traj existing
            formed_start, formed_end = [int(i) for i in tset[0].split(':')]
            calc_start, calc_end = [int(i) for i in tset[1].split(':')]
            irev = isrev[tset[2]]
            trajname = tset[3]

            if "prod" in trajname:
                trajname = direc + "/dcd/" + trajname + "_trim.dcd"

            if formed_start > formed_end:
                formed_start, formed_end = formed_end, formed_start

            if calc_start > calc_end:
                calc_start, calc_end = calc_end, calc_start

            if os.path.exists(trajname) and "/dcd/prod" in trajname:
                
                dcd = EnzymeTrajectories.read_dcd(trajname, irev)
                
                fxn = lambda x: EnzymeTrajectories.get_dist(dcd[:, x[0], :],dcd[:, x[1], :])
                r = np.array(list(map(fxn, atom_pairs)))

                if irev:
                    cstart = r.shape[1] - (calc_end+1)
                    cend   = cstart + (calc_end+1-calc_start)
                    geometries[formed_start-1:formed_end, :] = r[:, cstart:cend].T
                else:
                    geometries[formed_start-1:formed_end, :] = r[:, calc_start:(calc_end+1)].T

        return geometries

    @staticmethod
    def readpdb(fname, cols):
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
        print("\tExtracted PDB")
        return pdb

    @staticmethod
    def find_chosenatoms(pdb, chosenatoms, cols=["Res Name", "Res ID", "Atom Type"]):
        """
        Given certain canonical atoms you're looking into,
        return the position in the PDB that they come from.
        """
        aidx = []
        for atom in chosenatoms:
            aidx.append(pdb[(pdb[cols[0]] == atom[0]) &
                            (pdb[cols[1]] == atom[1]) &
                            (pdb[cols[2]] == atom[2])].index[0])
        print("\tIdentified atom index in the PDB")
        return aidx

    @staticmethod
    def get_dist(atom1,atom2,axis=1):
        """Get the distance between 2 atoms. Expects 3 x Nframes"""
        return norm(atom1 - atom2, axis=axis)

    @staticmethod
    def read_dcd(dcdfile, isrev):
        """
        Given a DCD, read all atoms and
        slice out the trace of interest.

        Requires the stitch forward + backward.
        """
        dcd = np.array(DCDReader(dcdfile))
        if isrev:
            dcd = dcd[::-1, :, :]

        return dcd



#if __name__ == "__main__":
