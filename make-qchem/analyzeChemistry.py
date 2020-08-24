#Natasha Seelam (nseelam@mit.edu)
#The following should be run in the same folder as analysisFunctions OR, line 34 should be commented out.
#This also requires access to several files which are not provided due to being proprietary MIT data.

#October 12, 2019 THIS IS THE FILE THAT GENERATES THE NEW CLUSTER DATA IN /data/nseelam03/fullqmset/

#October 11, 2019
#Brian's files have a bunch of duplicates because of MC weighting. This can affect the LR classification task
#In order to perform better, you will need to filter out these cases.

#October 8, 2019
#The following testing cases have been moved to create_trajectory_files.py

#October 4, 2019
#The following script creates an object that uses CHARMM dcd binary files, and produces Q-CHEM analysis files 

#Necessary packages

#Numerics
import numpy as np
import pandas as pd
from numpy.linalg import norm


#Data Storage
from sklearn.externals import joblib as jb
#import hdf5storage #This is helpful in reading MATLAB files; you can omit

#Read/Write DCD
import MDAnalysis

#External functions
import sys
#sys.path.insert(0, "/path/to/analysisFunctions.py")
from analysisFunctions import *


class EnzymeAnalysis():
    """
    The following creates an object "Enzyme Analysis" that will do the following steps:

    1) Extract the necessary quantum region from the trajectory files

    2) Create Q-CHEM NBO files with the appropriate chemical details

    3) Create a submission script that will deploy all these analyses. 

    """
    def __init__(self, pdbdir = None, Ntrajs = 10000, Nframes = 150, qmregion = None, pdbcolumns = None, clusterloc = None, rxncoor = [], rseed = 1234, fxnkey = "dist", R_upper = 1.9, features = None):
        """
        Inputs:
        -------
        pdbdir = Specify the location of the pdb that provides the atom-template for the binary files.
        Ntrajs = number of trajectories you would like to analyze per cluster
        Nframes = the expected number of frames for a single DCD
        qmregion = the region you'd like to perform NBO calculations on in the original PDB index (pythonically ordered so atom 1 = 0)
        pdbcolumns = pdb column names; if it's extended or not will change which columns exist.
        clusterloc = location of where the cluster txt file is
        rxncoor = atoms in the QMregion index that define the reaction coordinate. Should be a list. 
        rseed = random seed initializer
        fxntype = rxncoordinate function (see analysis_functions). Most often "dist"
        R_upper = upper bound of the reactant basin coordinate
        features - a dictionary of ML features you want to preserve
        DEPRECATED P_lower = lower bound of the product basin coordinate 


        Outputs:
        --------
        Returns an initialized version of the object that includes the pdbdir, and the pdb of interest

        """
        if pdbdir is None:
            print("Using Default pdb directory for N.Seelam")
            pdbdir = "trimtest.pdb"
        if pdbcolumns is None:
            print("Assuming extended pdb columns")
            pdbcolumns = [ "Atom", 
                           "Atom ID", 
                           "Atom Type", 
                           "Res", "Res ID", 
                           "X Coor", 
                           "Y Coor", 
                           "Z Coor", 
                           "B-factor", 
                           "Charge", 
                           "Chain ID"]
        assert len(rxncoor) > 0, "You need to specify some reaction coordinate for further analysis"
        #assert pdbdir is not None, "You must specify the pdb directory of interest for further steps."
        self.Ntrajs   = Ntrajs
        self.Nframes  = Nframes
        self.pdbdir   = pdbdir
        self.qmregion = qmregion
        self.features = features
        with open(pdbdir, 'r') as f:
            self.pdblines = f.readlines()
        self.pdblines = [line for line in self.pdblines if "REMARK" not in line] #Strip the header
        self.pdblines = [self.pdblines[idx] for idx in qmregion]
        self.pdb, self.atomnames, self.pdbqm = self.initCoordinates(qmregion = self.qmregion, pdbcolumns = pdbcolumns)
        self.rseed   = rseed
        self.fxnkey  = fxnkey
        self.R_upper = R_upper
        self.rxncoor = rxncoor
        self.trajs   = {} # Empty dictionary to hold trajectories in
        if qmregion is None:
            print("No QM-Region was specified.")
        if clusterloc is None:
            print("You must specify the cluster location otherwise you cannot load DCD")
        else:
            self.clusterloc = clusterloc
        if features is not None:
            ml_names = []
            for key in features["geo_keys"]:
                for ft in features[key]:
                    ml_names.append("-".join([self.pdb["Atom Type"][idx] for idx in ft]))
        self.geo_names = ml_names


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
            pdb.columns = pdbcolumns
        if qmregion is not None:
            pdbsmall = pdb.iloc[qmregion, :]
            pdbsmall.iloc[:, 5] = pdbsmall.iloc[:, 5].apply(lambda x: float(x))
            pdbsmall.iloc[:, 6] = pdbsmall.iloc[:, 6].apply(lambda x: float(x))
            pdbsmall.iloc[:, 7] = pdbsmall.iloc[:, 7].apply(lambda x: float(x))
        else:
            pdbsmall = None
        return pdb, atomnames, pdbsmall


    def initDCDfiles(self, ensemble = "reac"):
        """

        Provided an MD dcd trajectory file, MDAnalysis package will read the timesteps and coordinates of the simulation of interest.

        Input:
        -----
        Trajectory location for reactive/nonreactive ensembles

        Output:
        -------
        Returns the coordinates of the dcdfile that is Ntime x Natoms x 3 [xyz] coordinates

        """
        self.ensemble = ensemble
        if ensemble.lower() == "reac":
            print("Loading the Reactive DCD Map File")
            mapname = self.clusterloc + 'all_reactive_clusters.txt'
        else:
            print("Finished loading the NonReactive DCD Map File")
            mapname = self.clusterloc + 'all_nonreactive_clusters.txt'
        self.tloc    = map_clusters(mapname)
        self.Nclust   = list(self.tloc.keys())
        print("Completed loading map in self.tloc, self.Nclust")

    def initUniqueDCDfiles(self, ensemble = "reac"):
        """

        When generating simulations in a MC fashion, you end up with duplicate trajectories. You can opt to peform ML with non-duplicate simulations only by expressly
        filtering trajectories of interest. This function is identical to initDCDfiles, but will then preprocess and filter the files of interest.

        Input:
        -----
        Trajectory location for reactive/nonreactive ensembles

        Output:
        -------
        Returns the coordinates of the UNIQUE dcdfile that is Ntime x Natoms x 3 [xyz] coordinates

        """
        self.initDCDfiles(ensemble = ensemble)
        print("For " + ensemble + " finding unique DCD files across the clusters.")
        tloctmp = {}
        for clustkey in self.Nclust:
            tmp = self.tloc[clustkey]
            tloctmp.update({clustkey : list(set(tmp))})
        self.tloc = tloctmp
        print("Completed loading map of UNIQUE DCDs in self.tloc, self.Nclust")


    def loadTrajs(self, clustid, nmod):
        """
        After assigning the cluster map, load the trajectories

        Input:
        -----
        clustid - the cluster you wish to load features into. Should be either a key, or a list


        Output:
        ------
        Assigns trajectories to self.rdata, self.nrdata

        Nclust, trajloc, qmregion, Ntrajs, Nframes, rseed, rxncoor_atoms, fxnkey, R_upper, P_lower

        """
        if isinstance(clustid, list) or isinstance(clustid, np.ndarray):
            assert all(x in self.Nclust for x in clustid), "ClustID set provided contains an incorrect key. Check self.Nclust."
            for cd in clustid:
                self.trajs.update(load_trajectories(clustid       = cd, 
                                                    trajloc       = self.tloc, 
                                                    qmregion      = self.qmregion, 
                                                    Ntrajs        = self.Ntrajs, 
                                                    Nframes       = self.Nframes * 2, 
                                                    rseed         = self.rseed, 
                                                    rxncoor_atoms = self.rxncoor, 
                                                    fxnkey        = self.fxnkey,
                                                    R_upper       = self.R_upper,
                                                    nmod          = nmod,
                                                    features      = self.features,
                                                    ))
        else:
            assert clustid in self.Nclust, "ClustID is an incorrect key. Check self.Nclust."
            print("Loading trajs for " + str(self.ensemble) + " ensemble, cluster " + str(clustid))
            self.trajs.update(load_trajectories(clustid       = clustid, 
                                                trajloc       = self.tloc, 
                                                qmregion      = self.qmregion, 
                                                Ntrajs        = self.Ntrajs, 
                                                Nframes       = self.Nframes * 2, 
                                                rseed         = self.rseed, 
                                                rxncoor_atoms = self.rxncoor, 
                                                fxnkey        = self.fxnkey,
                                                R_upper       = self.R_upper,
                                                nmod          = nmod,
                                                features      = self.features,
                                                ))


    def filterTrajs(self, trev = 50, tfor = 50, Ntraj_short = True, keep_trajs = False, Nmin = None):
        """

        The following will slice out each trajectory so that you have an equal forward/backward number of points per calculation.
        Every point will define 0 at the trev + 1 (pythonically trev)
        
        You must specify what slice of the trajectory to look at. 50fs before and 50fs after t0 is good value

        Inputs:
        -------

        trev = fs back in time
        tfor = fs forward in time
        Ntraj_short = if you want a shorter number of trajectories. Default is implemented (True). True will re-adjust the number of Trajectories
        keep_trajs = option to keep the DCD; this might make an enormous object in memory. Default is to override
        Nmin = keep a subset of trajectories regardless if you want a few or not
        features - time aligned form of ML features

        This function will automatically also find the new minimum number of trajectories between clusters.
        
        Default is 50 fs forward or backward in time for full 100 fs. timestep

        """
        if keep_trajs is True:
            print("Saving the original DCD files as self.old_traj")
            self.old_traj = self.trajs
        trajtmp = {}
        dNtraj = self.Ntrajs
        for clustid in self.trajs.keys():
            trajtmp.update({clustid : filtertraj(self.trajs[clustid], self.Nframes * 2, trev, tfor, Nmin, self.features)})
            dNtraj = min([dNtraj, trajtmp[clustid]["Trajs"].shape[0]])
        self.trajs = trajtmp
        self.Nframes_old = self.Nframes
        self.Nframes = trev + tfor + 1
        if Ntraj_short is True:
            self.minimizeTrajs()
        print("Updated " + self.ensemble + " trajectory to have to new Nframes, " + str(trev+tfor+1))


    def minimizeTrajs(self):
        """
        Finds the minimum number of trajectories for all files

        Inputs:
        ------
        None


        Outputs:
        --------
        Re-assigns Ntrajs
        """
        NT = []
        for clustid in self.trajs.keys():
            NT.append(self.trajs[clustid]["Trajs"].shape[0])
        
        self.Ntrajs = min(NT)


    def makeQchem(self, clustid, qchemdir, atomlist, job_parameters, nbo_parameters, charge = 0, multiplicity = 1, lambda_value = False):
        """
        
        Given a set of trajectories, and a location to place them, the following will make q-chem files.

        Inputs:
        -------
        clustid - the cluster you want to make q-chem files for
        qchemdir - directory location to input the q-chem files
        atomlist - atomtypes 'C,H,O,N etc.' for each atom of interest. It should match the order of the QM region
        job_parameters - job options for q-chem
        nbo_parameters - nbo options for q-chem
        charge - charge of qm region (should be 0 default)
        multiplicity - number of unpaired electrons, 1 by default (spin states are complicated for metals...)
        lambda_value - optional, if you want to print the value of the rxncoordinate. Default False, flag as True for rxn coordinate. Supply your own for other stuff

        """
        lambda_value = determine_lambda_comment(clustid, lambda_value)
        if qchemdir[-1] is not "/":
            qchemdir += "/"
        if isinstance(clustid, list) or isinstance(clustid, np.ndarray):
            for cd in clustid:
                print("Making qchem files for " + str(self.ensemble) + ", cluster " + str(cd))
                prefix = self.ensemble + "_clust" + str(cd)
                Ntrajs = self.trajs[cd]["Trajs"].shape[0]
                make_qchemfiles(self.trajs[cd]["Trajs"], qchemdir + self.ensemble + "/cluster" + str(clustid) + "/", atomlist, Ntrajs, self.Nframes, prefix, job_parameters, nbo_parameters, charge, multiplicity, self.trajs[cd]["Trajnames"], lambda_value[cd])
        else:
            assert clustid in self.Nclust, "ClustID is an incorrect key. Check self.Nclust."

            print("Making qchem files for " + str(self.ensemble) + ", cluster " + str(clustid))
            prefix = self.ensemble + "_clust" + str(clustid)
            Ntrajs = self.trajs[clustid]["Trajs"].shape[0]
            make_qchemfiles(self.trajs[clustid]["Trajs"], qchemdir + self.ensemble + "/cluster" + str(clustid) + "/", atomlist, Ntrajs, self.Nframes, prefix, job_parameters, nbo_parameters, charge, multiplicity, self.trajs[clustid]["Trajnames"], lambda_value)


    def makePDB(self, xyzcoor, pdbfilename = None):
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
        assert (xyzcoor.shape[0] - self.pdb.shape[0]) < 1e-9, "Your xyz-coordinates do not match the order of the pdb file "
        pdblines = replaceCoor(self.pdblines, xyzcoor, nround = 3)
        if pdbfilename is not None:
            if pdbfilename[-4:] != ".pdb":
                pdbfilename += ".pdb" #Add the extension that user forgot to add
            with open(pdbfilename, 'w') as f:
                f.writelines(pdblines)
        else:
            return pdblines






# ---------------------------------------------- #
# Testing Cases/Deprecated Functions
# ---------------------------------------------- #
#    def loadTrajs(self, clustid):
#        """
#        After assigning the cluster map, load the trajectories
#
#        Input:
#        -----
#        clustid - the cluster you wish to load features into. Should be either a key, or a list
#
#
#        Output:
#        ------
#        Assigns trajectories to self.rdata, self.nrdata
#
#        Nclust, trajloc, qmregion, Ntrajs, Nframes, rseed, rxncoor_atoms, fxnkey, R_upper, P_lower
#
#        """
#        if isinstance(clustid, list) or isinstance(clustid, np.ndarray):
#            assert all(x in self.Nclust for x in clustid), "ClustID set provided contains an incorrect key. Check self.Nclust."
#            for cd in clustid:
#                self.trajs.update(load_trajectories(clustid       = cd, 
#                                                    trajloc       = self.tloc, 
#                                                    qmregion      = self.qmregion, 
#                                                    Ntrajs        = self.Ntrajs, 
#                                                    Nframes       = self.Nframes * 2, 
#                                                    rseed         = self.rseed, 
#                                                    rxncoor_atoms = self.rxncoor, 
#                                                    fxnkey        = self.fxnkey,
#                                                    R_upper       = self.R_upper,
#                                                    ))
#        else:
#            assert clustid in self.Nclust, "ClustID is an incorrect key. Check self.Nclust."
#            print("Loading trajs for " + str(self.ensemble) + " ensemble, cluster " + str(clustid))
#            self.trajs.update(load_trajectories(clustid       = clustid, 
#                                                trajloc       = self.tloc, 
#                                                qmregion      = self.qmregion, 
#                                                Ntrajs        = self.Ntrajs, 
#                                                Nframes       = self.Nframes * 2, 
#                                                rseed         = self.rseed, 
#                                                rxncoor_atoms = self.rxncoor, 
#                                                fxnkey        = self.fxnkey,
#                                                R_upper       = self.R_upper,
#                                                ))
#
#traj = trajs[0,0,:,:]
#comment = a.trajs[1]["Trajnames"][0]
#lines = qchemlines(traj, atomlist, job_parameters, nbo_parameters, comment, charge, multiplicity)
#for line in lines:
#    print(line)#
#
#
#
