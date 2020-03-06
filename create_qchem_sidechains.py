# Feb 28 2020 

import sys
import os
from copy import deepcopy
 
# Numerics
import numpy as np
import pandas as pd
import pickle as pkl

# regex
import re
 
# Static Methods
from LRfunctions import *
from numpy.linalg import norm
import hdf5storage


class EnzymeAnalysis(object):
    """
     
    The following creates an object "Enzyme Analysis" that will do the following steps:

    1) Extract the necessary quantum region from the trajectory files

    2) Create Q-CHEM NBO files with the appropriate chemical details

    3) Create a submission script that will deploy all these analyses. 
 
    """
    def __init__(self,
                 clustnames,
                 qmregion,
                 qchemdir,
                 pdbcolumns=None,
                 pdbdir=None,
                 atomdir=None,
                 ldir=None,
                 ):

        # Initialize terms
        self.clustnames = clustnames

        if pdbcolumns is None:
            print("Assuming extended pdb columns")
            self.pdbcolumns = [ "Atom", 
                                "Atom ID", 
                                "Atom Type", 
                                "Res", "Res ID", 
                                "X Coor", 
                                "Y Coor", 
                                "Z Coor", 
                                "B-factor", 
                                "Charge", 
                                "Chain ID"]

        # Initialize PDB
        if pdbdir is None:
            self.pdbdir = "trimtest.pdb" #Provide the PDB 
        else:
            self.pdbdir = pdbdir

        # Initialize Trajectory coordinate Dir
        if atomdir is None:
            self.atomdir = "dcddir/" #Provide the DCD Directory
        else:
            self.atomdir = atomdir

        # Initialize troughs directory
        if ldir is None:
            self.ldir = "troughs/" #Provide the directory with troughs
        else:
            self.ldir = ldir


        self.qmregion = qmregion
        self.qchemdir = qchemdir
        # Load the data
        print("Loading pdb")
        with open(self.pdbdir, 'r') as f:
            self.pdblines = f.readlines()


        self.pdblines = [line for line in self.pdblines if "REMARK" not in line] #Strip the header
        self.pdblines = [self.pdblines[idx] for idx in self.qmregion]
        
        # Initialize PDB coordinates
        self.initCoordinates(qmregion=self.qmregion, pdbcolumns=self.pdbcolumns)
        

    def initCoordinates(self, qmregion, pdbcolumns):
        """
        Inputs:
        -------
        Qmregion - specify the PYTHONIC INDEX OF THE ATOMS that you will make the QM-analysis on. Requires knowledge of the original PDB structure.
                ~NOTE, expert advice suggests using the linker-atoms for covalent bonds OR a region that is covalently saturated. 
        pdbcolumns - names of the pdb file columns (important if extended or not); Default - None


        Outputs:
        --------
        Initializes pdb coordinates in "pdbdir"

        """
        with open(self.pdbdir, 'r') as f:
            pdb = f.readlines()
            pdb = [i.split() for i in pdb]
            #Strip pdb remarks
            pdb = [line for line in pdb if "REMARK" not in line]
            #Last two lines are a always a terminate statement, strip
            pdb = [line for line in pdb if "TER" not in line]
            pdb = [line for line in pdb if "END" not in line]
            pdb = pd.DataFrame(pdb)

        pdb.loc[:, 5] = pdb.loc[:, 5].apply(lambda x: float(x))
        pdb.loc[:, 6] = pdb.loc[:, 6].apply(lambda x: float(x))
        pdb.loc[:, 7] = pdb.loc[:, 7].apply(lambda x: float(x))
        atomnames = {idx : [pdb.iloc[idx, 3], pdb.iloc[idx, 4], pdb.iloc[idx, 2]] for idx in range(len(pdb))}

        if pdbcolumns is not None:
            pdb.columns = self.pdbcolumns

        if qmregion is not None:
            pdbsmall = pdb.iloc[qmregion, :]
            pdbsmall.iloc[:, 5] = pdbsmall.iloc[:, 5].apply(lambda x: float(x))
            pdbsmall.iloc[:, 6] = pdbsmall.iloc[:, 6].apply(lambda x: float(x))
            pdbsmall.iloc[:, 7] = pdbsmall.iloc[:, 7].apply(lambda x: float(x))
        else:
            pdbsmall = None

        self.pdb = pdb
        self.atomnames = atomnames
        self.pdbqm = pdbsmall

    def load_atom_positions(self, clustnames=None, Nframes=150):
        """
        
        The following function explicitly downloads and processes the all-atom trajectory data.

        Input: Cluster Names (optional list of strings)
               Nframes (number of frames per trajectory)

        Output: Assigns "raw traj" to self, which is a dictionary for each ensemble type (reac/nonreac) and the cluster ID

        """
        if clustnames is None:
            clustnames = self.clustnames

        # Load the MATLAB files
        trajs = {}
        for enstype in ["reac", "nonreac"]:
            for clust_id in self.clustnames:
                print("Loading", enstype, str(clust_id), flush=True)
                key = enstype + " " + clust_id 
                value = hdf5storage.loadmat(self.atomdir + "clust_" + str(clust_id) + "_traj_" + enstype + ".mat")["trajs"][0,:]
                trajs.update({key: value}) #Get the Trajectories
        
        self.raw_trajs = trajs

    def select_QMregion(self, Natoms=341, Nframes=150):
        """
        From the QM region, downselect the desired QM region.
        Takes the raw trajectories and makes an array of 
        Ntrajectories x Natoms x xyz x Nframes

        Inputs:
        Natoms  - total atoms in the raw_frame dictionary (value.shape[0])
        Nframes - number of frames per simulation

        """
        qm_trajs = {}
        for enstype in ["reac", "nonreac"]:
            for clust_id in self.clustnames:
                print("Processing QM region of", enstype + " " + clust_id, flush=True)
                key = enstype + " " + clust_id
                value = self.raw_trajs[key]
                NT = len(value)
                traj = np.nan*np.zeros(shape = (NT, Natoms, 3, Nframes)) #NTrajectories x Natoms_per_dcd x 3_xyzcoorm x Nframes_per_simulation
                for tidx in range(NT):
                    traj[tidx, :, :, :] = value[tidx]
                qm_trajs.update({key: traj[:, self.qmregion, :, :]})

        self.qm_trajs = qm_trajs


    def translate_atom_positions(self, 
                                 fixed_atoms,
                                 trans_atoms,
                                 rdist,
                                 axis=0,
                                 clustnames=None, 
                                 Nframes=150,
                                 ):
        """
        For each of the trajectories, get the coordinate structure and translate it.
        """
        # Set the distance as a list; iterate through each case
        if isinstance(rdist, float):
            rdist = [rdist for i in range(len(fixed_atoms))]

        if clustnames is None:
            clustnames = self.clustnames

        traj_update = {}
        for enstype in ["reac", "nonreac"]:
            for clust_id in self.clustnames:
                key = enstype + " " + clust_id
                value = self.qm_trajs[key]
                for tidx in range(value.shape[0]):
                    xyzcoor = value[tidx, :, :, :]
                    for aidx in range(len(trans_atoms)):
                        atom1 = fixed_atoms[aidx]
                        atom2 = trans_atoms[aidx]
                        xyzcoor[atom2, :, :], _ = EnzymeAnalysis.translate_atoms(xyzcoor[atom1, :, :], xyzcoor[atom2, :, :], rdist[aidx], axis)
                    value[tidx, :, :, :] = xyzcoor
                traj_update.update({key: value})

        self.trans_traj = traj_update

    def create_trans_pdb(self,
                         xyzcoor,
                         line_id, 
                         atomname_old="CA",
                         atomname_new="H",
                         pdbfilename=None,
                         ):
        """
        Creates a PDB of translated atoms.

        Input:
        enstype - ensemble type
        clustid - cluster ID

        """
        plines = self.pdblines
        plines = EnzymeAnalysis.convert_atoms(plines, line_id, atomname_old, atomname_new)
        tpdb=EnzymeAnalysis.makePDB(self.pdb, plines, xyzcoor, pdbfilename)
        if pdbfilename is not None:
            return tpdb

    def get_atomtype(self, atommap, repl_atom):
        """
        Get the atom-name nomenclature

        Inputs:
        atommap - a dictionary of what each letter corresponds to in an atom map
        repl_atom - None or list of tuples, where each tuple is (index_to_replace, value_to_replace_with)
        """
        # Set the atomic map on organic atoms
        if atommap is None:
            print("Default organic atoms only.")
            atommap = {"C":"C", "O":"O", "M":"Mg", "H":"H", "N":"N"}

        # Make a dictionary of QM names
        self.qmatoms = {idx: self.atomnames[i] for idx, i in enumerate(self.qmregion)}

        # Identify the atom type and appropriate spacing. 
        self.qm_names = [atommap[j[-1][0]] if j[-1][0] in atommap.keys() else "X" for j in self.qmatoms.values()]
        self.qm_names = ["{:<5}".format(str_name) for str_name in self.qm_names]

        if repl_atom is not None:
            qm_names = self.qm_names
            for pair in repl_atom:
                qm_names[pair[0]] = pair[1]

        self.qm_names = qm_names

    def make_all_qchem(self,   
                       job_parameters, 
                       nbo_parameters, 
                       charge, 
                       multiplicity, 
                       comment='',
                       atommap=None,
                       repl_atom=None):
        """
        Make all the q-chem files for each ensemble/cluster
        """
        self.get_atomtype(atommap=atommap, repl_atom=repl_atom)
        for enstype in ["reac", "nonreac"]:
            
            ensdir = self.qchemdir + enstype + "/"
            if os.path.exists(ensdir) is False:
                os.makedirs(ensdir)

            for clustid in self.clustnames:
                clustdir = ensdir +"cluster" + str(clustid) + "/"

                print("Making trajectories for=", clustdir)
                
                if os.path.exists(clustdir) is False:
                    os.makedirs(clustdir)
                
                self.make_qchem_files(clustdir, 
                                      enstype, 
                                      clustid, 
                                      self.qm_names, 
                                      job_parameters, 
                                      nbo_parameters, 
                                      charge, 
                                      multiplicity, 
                                      comment)


    def make_qchem_files(self,
                         clusterdir,
                         enstype,
                         clustid,
                         atomlist,  
                         job_parameters, 
                         nbo_parameters, 
                         charge, 
                         multiplicity, 
                         comment):
        """
        """
        key = enstype + " " + clustid
        filename_prefix = enstype + "_" + clustid
        value = self.trans_traj[key]

        for trajidx in range(value.shape[0]): #Over all trajectories
            trajdir = clusterdir + "traj" + str(trajidx+1) + "/"
            
            if os.path.exists(trajdir) is False:
                os.makedirs(trajdir)

            xyzcoor = value[trajidx, :, :, :]
            for fridx in range(xyzcoor.shape[-1]):
                xyz = xyzcoor[:,:,fridx]
                qchemfile = trajdir + filename_prefix + "_traj" + str(trajidx + 1) + "_fr" + str(fridx + 1) + ".qcin"
                EnzymeAnalysis.make_qchem_files_frame(xyz, 
                                                      qchemfile, 
                                                      atomlist, 
                                                      job_parameters, 
                                                      nbo_parameters, 
                                                      charge, 
                                                      multiplicity, 
                                                      comment)
                
    @staticmethod
    def make_qchem_files_frame(xyzcoor,
                               qchemfile,
                               atomlist,  
                               job_parameters, 
                               nbo_parameters, 
                               charge, 
                               multiplicity, 
                               comment):
        """
        Make a q-chem file per frame, per trajectory
        """
        qlines = EnzymeAnalysis.qchemlines(xyzcoor, atomlist, job_parameters, nbo_parameters, comment, charge, multiplicity)
        with open(qchemfile, "w") as f:
            f.writelines(qlines)

    @staticmethod
    def convert_atoms(pdblines, line_id, atomname_old, atomname_new):
        """
        Convert atomtype to a new one.
        
        Inputs:
        line_id = lines in which you wish to swap atomname
        atomname_orig = atomname in PDB to change from
        atomname_new = atomname you wish to call PDB
        """
        # PDB is space sensitive; impose the same length
        if len(atomname_old) > len(atomname_new):
            ljust_space  = len(atomname_old) - len(atomname_new)
            atomname_new = atomname_new.ljust(ljust_space+1)
        else:
            ljust_space  = len(atomname_new) - len(atomname_old)
            atomname_old = atomname_old.ljust(ljust_space+1)

        replace_fxn = lambda x: x[1].replace(atomname_old, atomname_new) if x[0] in line_id else x[1]

        return list(map(replace_fxn, enumerate(pdblines))) 
        

    @staticmethod
    def translate_atoms(atom1, atom2, r0, axis):
        """
        Given an atom position, translate across a vector.

        Input: 
        atom1 (fixed, np.array) 
        atom2 (moved atom, np.array)
        dist  (distance to translate atom)
        """
        r = EnzymeAnalysis.get_dist(atom1, atom2, axis)
        dist = np.ones(shape=r.shape)*r0
        epsn = (dist-r)/r
        dBA = atom2 - atom1
        vect = (dBA)*epsn
        atom_new = atom2 + vect
        return atom_new, vect

    @staticmethod
    def get_dist(atom1,atom2,axis):
        """Get the distance between 2 atoms. Expects 3 x Nframes"""
        return norm(atom1 - atom2, axis=axis)

    @staticmethod
    def replace_coor(pdblines, xyzcoor, nround=3):
        """
        Given a line in the pdb, replaces the xyz pos of the original template to a new one
        
        Inputs:
        ------
        pdb = original pdb lines
        xyz = (numpyarray) coordinates you wish to replace
        nround - int, number of decimal places to round to


        Outputs:
        --------
        Returns lines for a valid pdb file

        """
        newpdb = []
        for ctr, line in enumerate(pdblines):
            line = re.split(r'(\s+)', line)
            line_pre  = line[:-14] #all text before the xyzcoor
            line_post = line[-8:] #all text after xyzcoor
            coorlines  = [line[-14] + line[-13], line[-12] + line[-11], line[-10] + line[-9]]
            #Remove pre-existing negatives in the coordinate lines
            coorlines = [i.replace('-', ' ') for i in coorlines]
            #At most, every coordinate can have only a few standard characters.
            #Print 3 after the decimal, three before.
            xyzcoor_replacement = ["{:3.3f}".format(xyzcoor[ctr, idx]) for idx in range(3)]
            #The following line has these operations:
            #Take the last "N" positions of the coordinate line that matches the length of the replacement line.
            #Replace those N positions with the replacement line.
            coorlines = [coorlines[idx].replace(coorlines[idx][-len(xyzcoor_replacement[idx]):], xyzcoor_replacement[idx]) for idx in range(3)]
            newpdb.append(''.join(line_pre + coorlines + line_post))
        return newpdb

    @staticmethod
    def makePDB(pdb, pdblines, xyzcoor, pdbfilename):
        """
        The following returns a pdb with the new coordinates of interest

        Inputs:
        -------
        xyzcoor - a numpy array of Natoms x 3 coordinates for the molecule of interest
        pdbfilename - a string of the filename/location you want to save the pdb too


        Output:
        -------
        Saves a file indicated by 'filename' with the coordinates you input on the default pdb you've chosen. OR returns the lines of the pdb.

        """
        assert (xyzcoor.shape[0] - pdb.shape[0]) < 1e-9, "Your xyz-coordinates do not match the order of the pdb file "
        pdblines = EnzymeAnalysis.replace_coor(pdblines, xyzcoor, nround = 3)
        if pdbfilename is not None:
            if pdbfilename[-4:] != ".pdb":
                pdbfilename += ".pdb" #Add the extension that user forgot to add
            with open(pdbfilename, 'w') as f:
                f.writelines(pdblines)
        else:
            return pdblines

    @staticmethod
    def qchemlines(traj,
                   atomlist, 
                   job_parameters, 
                   nbo_parameters, 
                   comment, 
                   charge, 
                   multiplicity):
        """
        Creates the lines for a qchem file

        trajs - the xyz coordinates for 1 frame
        atomlist - the atomlist you wish to add
        job parameters - the jobtype parameters
        nbo_parameters - NBO-specific flags
        comment - additional comments
        charge - net charge
        multiplicity - net multiplicity

        """
        # Job Parameter initialization
        lines = ["$rem\n"]
        for key in job_parameters.keys():
            lines.append("   " + str(key) +" " + str(job_parameters[key]) + "\n")

        # NBO parameter initialization
        lines += ["$end\n\n$comment\n"]
        lines += [str(comment) + "\n$end\n\n$nbo\n"]
        for key in nbo_parameters.keys():
            lines.append("   " + str(key) +" " + str(nbo_parameters[key]) + "\n")
        
        # Describe the molecule
        lines += ['$end\n\n$molecule\n']
        lines += ["   " + str(charge) + " " + str(multiplicity)  + "\n"]
        
        #Write the atom coors
        for atom in range(traj.shape[0]):
            xyzcoor = '%12.3f  %8.3f  %8.3f\n' % (traj[atom, 0], traj[atom, 1], traj[atom, 2])
            lines += ['    ' + str(atomlist[atom]) + xyzcoor]
        
        lines += ["$end\n\n\n\n"]
        return lines


if __name__ == "__main__":

    # ---------------------- #
    #    USER PARAMETERS 
    # ---------------------- #
    # Define the q-chem directory
    qchemdir="KARI_sidechains/" #Directory you wish to save your trajectories in

    # Q-CHEM job Details
    job_parameters = { "JOBTYPE" : "sp", 
                       "EXCHANGE" : "b3lyp", 
                       "BASIS" : "6-31G(d)", 
                       "GUI" : "=2", 
                       "NBO" : "true"}

    # NBO job Details
    nbo_parameters = {"nbosum" : "", 
                      "cmo" : "", 
                      "BNDIDX" : "", 
                      "3CHB" : "",
                      "3CBOND" : ""}

    # Electronic charge/multiplicity
    charge = 0
    multiplicity = 1

    # Define the atoms of the side-chains
    as4 = [79] + list(np.arange(81, 87))
    gl6 = [91] + list(np.arange(93,102))
    gl5 = [144] + list(np.arange(146, 156))
    qm  = list(np.arange(309,341))
    
    # Identify the CA and CB of interest; we will later want to convert CA into "H"
    CA = [0, 7, 18]
    CB = [1, 8, 19]
    rdist = 1.07 # Distance H-C bond

    # Atomic mapping
    atommap = None
    repl_atom = [(i, "{:<5}".format("H")) for i in CA]

    # ---------------------- #
    #    RUN OBJECT 
    # ---------------------- #

    # Define the enzyme analysis object
    x = EnzymeAnalysis(clustnames = [str(i) for i in range(1,6)],
                       qmregion = as4 + gl5 + gl6 + qm,
                       qchemdir = qchemdir)

    # Load and select atoms
    x.load_atom_positions()
    x.select_QMregion()

    # Translate the positions of the CAs 
    x.translate_atom_positions(fixed_atoms=CB, trans_atoms=CA, rdist=rdist)

    #Testing 
    x.make_all_qchem(job_parameters, 
                     nbo_parameters, 
                     charge, 
                     multiplicity, 
                     comment='',
                     atommap=atommap,
                     repl_atom=repl_atom)