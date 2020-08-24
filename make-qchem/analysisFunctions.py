#Natasha Seelam (nseelam@mit.edu)
#October 12, 2019 THIS IS THE FILE THAT GENERATES THE NEW CLUSTER DATA IN /data/nseelam03/fullqmset/

#October 6, 2019
#The following has assorted necessary functions

#Necessary packages

#Numerics
import numpy as np
import pandas as pd
from numpy.linalg import norm


#System commands
import os
import sys

#Data Storage
from sklearn.externals import joblib as jb
#import hdf5storage #This is helpful in reading MATLAB files; you can omit

#Read/Write DCD
import MDAnalysis

#String Processing
import re


# ------------------------ #
# GEOMETRIC FUNCTIONS
# ----------------------- #
def unitvec(atom1, atom2):
    """Get the unit vector that defines 2 atoms. The first atom is the final position and the second atom is the initial position"""
    #Final - initial; Atom 1 is the final
    vec = (atom1-atom2) / ((((atom1-atom2)**2).sum(axis = 0)**0.5))
    return vec


def get_psi(atom1, atom2, atom3, atom4):
    """Getting the torsion, defined in get_fts.m in prockari_long/ """
    #%Using a Stack Exchange Version
    #%https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    #%"""formula from Wikipedia article on "Dihedral angle"; formula was removed
    #%from the most recent version of article (no idea why, the article is a
    #%mess at the moment) but the formula can be found in at this permalink to
    #%an old version of the article:
    #%https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    #%uses 1 sqrt, 3 cross products"""
    b0 = -1.0 * (atom2 - atom1) #[-1.0 * atom2[i] - atom1[i] for i in range(3)]
    b1 = atom3 - atom2          #[atom3[i] - atom2[i] for i in range(3)]
    b2 = atom4 - atom3          #[atom4[i] - atom3[i] for i in range(3)]
    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)
    y = np.dot(b0xb1_x_b1xb2, b1)/norm(b1)
    x = np.dot(b0xb1, b1xb2)
    psi = 180.0/np.pi * np.arctan2(y, x)
    return(psi)

fxntype = {'dist' : lambda x : (((x[:, 0, :]-x[:, 1, :])**2).sum(axis=1))**0.5, 
           'comb' : lambda x : (((x[:, 0, :]-x[:, 1, :])**2).sum(axis=1))**0.5 - (((x[:, 2, :]-x[:, 1, :])**2).sum(axis=1))**0.5,
           'angs' : lambda x : [np.arccos(theta) * 180./np.pi for theta in [np.dot(unitvec(x[frame_idx, 0, :], x[frame_idx, 1, :]), unitvec(x[frame_idx, 2, :], x[frame_idx, 1, :])) for frame_idx in range(x.shape[0])]], 
           'tors' : lambda x : [get_psi(x[frame_idx, 0, :], x[frame_idx, 1, :], x[frame_idx, 2, :], x[frame_idx, 3, :]) for frame_idx in range(x.shape[0])]}

#fxntype = {'dist' : lambda x : (((x[0]-x[1])**2).sum(axis=1))**0.5, 
#           'comb' : lambda x : (((x[0]-x[1])**2).sum(axis=1))**0.5 - (((x[2]-x[1])**2).sum(axis=1))**0.5,
#           'angs' : lambda x : [np.arccos(theta) * 180./np.pi for theta in [np.dot(unitvec(x[frame_idx, 0, :], x[frame_idx, 1, :]), unitvec(x[frame_idx, 2, :], x[frame_idx, 1, :])) for frame_idx in range(x.shape[0])]], 
#           'tors' : lambda x : [get_psi(x[frame_idx, 0, :], x[frame_idx, 1, :], x[frame_idx, 2, :], x[frame_idx, 3, :]) for frame_idx in range(x.shape[0])]}


# ------------------------ #
# PROCESSING FUNCTIONS
# ------------------------ #

def map_clusters(filename):
    """
    Given a file name, will open the txt file and process into clusters

    Input:
    ------
    filename - Text file in the form of "cluster ID", "dcdlocation and file name"

    Output:
    ------
    returns a dictionary where the index is the cluster ID and the value is a list of trajectories that were in that cluster
    
    """
    with open(filename, 'r') as f:
        txt = f.readlines()
    txt = [line.split(',') for line in txt]
    txt = [[int(line[0]), line[1].strip(' ').split('\n')[0]] for line in txt]
    map_dict = {}
    #Across all unique cluster keys from line[0], add the dcdfiles via line[1]
    for idx in np.unique([line[0] for line in txt]):
        map_dict.update({idx : [line[1] for line in txt if line[0] == idx]})
    return map_dict


def readdcd_full(filename, qmregion, rxncoor_atoms, fxnkey, features):
    """

    Provided an MD dcd trajectory file, will construct the full trajectory forward/backwards

    Input:
    -----
    filename - filename of the dcd file
    qmregion - qmregion you'd like to extract from the DCD file. Will use the PYTHONIC INDEX. Default is None
    rxncoor atoms - the reaction coordinate atoms in the QM-region number scheme (for KARI its C4C5 so 3,4)
    fxnkey - Type of function to evaluate the reaction coordinate (ex dist, angle etc.)
    R_upper - upper bound of the reactant basin
    DEPRECATED - P_lower - lower bound of the product basin 

    Output:
    -------
    Returns 2 objects:
    fulltraj - numpy array the coordinates of the both forward and backward that is Ntime x Natoms x 3 [xyz] coordinates
    trajid_isrev = 1 if the forward trajectory is run in reverse, 0 if the reverse trajectory is run backwards.


    """
    traj_pre = filename.replace('_for', ' ').replace('_rev', ' ').split() #Preamble of the file
    name_fordcd = traj_pre[0] + '_for' + traj_pre[1]
    name_revdcd = traj_pre[0] + '_rev' + traj_pre[1]
    fordcd = np.array(MDAnalysis.coordinates.DCD.DCDReader(name_fordcd))
    revdcd = np.array(MDAnalysis.coordinates.DCD.DCDReader(name_revdcd))
    fulltraj = np.zeros(shape = (fordcd.shape[0] + revdcd.shape[0], fordcd.shape[1], 3)) #Combined Trajectory
    #The reverse direction is written in backwards time, you must reverse the ordering of the data.
    for_r = get_rxncoor(fordcd[:, qmregion, :], rxncoor_atoms, fxnkey)
    rev_r = get_rxncoor(revdcd[:, qmregion, :], rxncoor_atoms, fxnkey)
    #If the maximum of the trajectory exists in the reverse file, then use that.
    if max(rev_r) > max(for_r):
        fulltraj[:fordcd.shape[0], :, :] = fordcd[::-1, :, :]
        fulltraj[fordcd.shape[0]:, :, :] = revdcd
        trajid_isrev = 1 #[1 for idx in range(fordcd.shape[0])] + [0 for idx in range(revdcd.shape[0])]
    else:
        fulltraj[:revdcd.shape[0], :, :] = revdcd[::-1, :, :]
        fulltraj[revdcd.shape[0]:, :, :] = fordcd
        trajid_isrev = 0
    if features is not None:
        fts = get_MLfeatures(features, fulltraj)
        return fulltraj[:, qmregion, :], trajid_isrev, fts
    else:
        return fulltraj[:, qmregion, :], trajid_isrev, None

def readdcd(filename, qmregion):
    """

    Provided an MD dcd trajectory file, MDAnalysis package will read the timesteps and coordinates of the simulation of interest.

    Input:
    -----
    filename - filename of the dcd file
    qmregion - qmregion you'd like to extract from the DCD file. Will use the PYTHONIC INDEX. Default is None

    Output:
    -------
    Returns the coordinates of the dcdfile that is Ntime x Natoms x 3 [xyz] coordinates
    
    """
    dcdfile = np.array(MDAnalysis.coordinates.DCD.DCDReader(filename))[:, qmregion, :]
    #The reverse direction is written in backwards time, you must reverse the ordering of the data.
    return dcdfile

def get_MLfeatures(features, trajdcd):
    """
    This function will calculate the number of ML features you would like, for each timepoint, within a trajectory

    Inputs:
    --------

    features - The features you would like to compute. This is provided as a dictionary, where the keys are the operation you want to perform (dist/ang/tors) and the atoms involved.
               This should be performed BEFORE you shave off the quantum region. One of the elements of features is the key "Nfts" which is the total number of features to compute. One more that must be present
               is 'geo_keys' which are the geometric features you must compute'. The keys of Nfts must be in fxntype.

    Outputs:
    ---------

    Returns a matrix of Nfeatures x Ntimepoints per trajectory.


    """
    #Initialize matrix of Nfeatures x Ntimepoints
    all_features = np.zeros(shape = (trajdcd.shape[0], features["Nfts"]))
    countr = 0
    for key in features['geo_keys']:
        ftset = features[key]
        for ft in ftset:
            all_features[:, countr] = fxntype[key](trajdcd[:, ft, :])
            countr += 1
    return all_features



def get_rxncoor(trajdcd, rxncoor_atoms, fxnkey):
    """
    Given a trajectory, it will compute the reaction coordinate

    Inputs:
    -------
    trajdcd - DCD file of the trajectory
    rxncoor_atoms - list with the indices of the atoms that define the reaction coordinate. 
    fxnkey - type of reaction coordinate
    R_upper - the upper bound of the reactant basin

    NOT IMPLEMENTED - angle, or torsion coordinate. 

    Returns:
    -------
    trough value of the feature of interest

    """
    r = fxntype[fxnkey](trajdcd[:, rxncoor_atoms, :])
    return r

def get_troughs_oscillatory(trajdcd, rxncoor_atoms, fxnkey, R_upper):
    """
    Given a trajectory, it will compute the last trough between the two atoms of the reaction coordinate.

    The t0 is defined by oscillatory bonds. You can also ask the last time r < R_upper (or last time in Reactant Basin)

    Inputs:
    -------
    trajdcd - DCD file of the trajectory
    rxncoor_atoms - list with the indices of the atoms that define the reaction coordinate. 
    fxnkey - type of reaction coordinate
    R_upper - the upper bound of the reactant basin

    NOT IMPLEMENTED - angle, or torsion coordinate. 

    Returns:
    -------
    trough value of the feature of interest

    """
    r = get_rxncoor(trajdcd, rxncoor_atoms, fxnkey)
    tp = [t for t in range(np.argmax(r))]
    rtp = [r[idx] for idx in tp]
    t0 = [t for t in range(1, len(rtp) - 1) if rtp[t] <  rtp[t+1] and rtp[t] < rtp[t-1] and rtp[t] < R_upper] #explicitly the distance is less than the step before and after it.
    if len(t0):
        t0 = t0[-1] #Find the last time you were in the reactant basin trough
    else:
        t0 = 0
    return t0


def choose_trajectories(qmregion, Ntrajs, trajloc, Nframes, rseed, fxnkey, R_upper, rxncoor_atoms, nmod, features):
    """
    The following script will, given Ntrajs, pick (from each cluster) this amount of trajectories randomly
    to be able to calculate the reactive/nonreactive performance.

    Inputs:
    ------
    qmregion - index of the atoms you wish to analyze chemistry of (must be from original PDB)
    Ntrajs   - number of trajectories you wish to store
    trajloc  - mapping of the trajectories you wish to process
    Nframes  - number of frames in the trajectory you wish to sample
    rseed    - Random seed used to generate the random numbers to sample from
    R_upper  - upper bound of the reactant basin; used for t0
    rxncoor_atoms - reaction coordinate atoms in pythonic qmregion indexing
    nmod - if you wish to print out progress, will print
    features - a dictionary of features if you choose to employ them

    Outputs:
    -------
    Returns a dictionary such that each key is the cluster ID, and each value is a matrix of Ntrajs x Natoms x 3 

    """
    #initialize trajectory array
    if Ntrajs > len(trajloc):
        Ntrajs = len(trajloc)
        print("You specified too many trajectories than assigned to this cluster. Reducing to = " + str(Ntrajs))
    trajcoor = np.zeros(shape = (Ntrajs, Nframes, len(qmregion), 3))
    rxncoor = np.zeros(shape = (Ntrajs, Nframes))
    isrev = [] #Confirms whether a trajectory is reversed or not
    t0 = [] #Confirms the t0 point of the trajectory
    trajs = [idx for idx in range(Ntrajs)]
    np.random.seed(rseed)
    np.random.shuffle(trajs)
    #idx - counter over trajectories, traj_idx - the actual number you wish to cull from
    if features is not None:
        feature_set = np.zeros(shape = (Ntrajs, Nframes, features["Nfts"]))
        for idx, traj_idx in enumerate(trajs):
            if idx % nmod == 0:
                print("Iteration = " + str(idx+1) + "/" + str(Ntrajs))
            fulltraj, irev, feature_set[idx, :, :] = readdcd_full(trajloc[traj_idx], qmregion, rxncoor_atoms, fxnkey, features = features)
            trajcoor[idx, :, :, :] = fulltraj[:Nframes, :, :]
            isrev.append(irev)
            t0.append(get_troughs_oscillatory(trajcoor[idx, :, :, :], rxncoor_atoms, fxnkey, R_upper))
            rxncoor[idx, :] = get_rxncoor(trajcoor[idx, :, :, :], rxncoor_atoms = rxncoor_atoms, fxnkey = fxnkey)
    else:
        feature_set = None
        for idx, traj_idx in enumerate(trajs):
            if idx % nmod == 0:
                print("Iteration = " + str(idx+1) + "/" + str(Ntrajs))
            fulltraj, irev, fts = readdcd_full(trajloc[traj_idx], qmregion, rxncoor_atoms, fxnkey, features = features)
            trajcoor[idx, :, :, :] = fulltraj[:Nframes, :, :]
            isrev.append(irev)
            t0.append(get_troughs_oscillatory(trajcoor[idx, :, :, :], rxncoor_atoms, fxnkey, R_upper))
            rxncoor[idx, :] = get_rxncoor(trajcoor[idx, :, :, :], rxncoor_atoms = rxncoor_atoms, fxnkey = fxnkey)
    return trajcoor, [trajloc[idx] for idx in trajs], t0, isrev, rxncoor, feature_set



def load_trajectories(clustid, trajloc, qmregion, Ntrajs, Nframes, rseed, rxncoor_atoms, fxnkey, R_upper, nmod, features):
        """
        After assigning the cluster map, load the trajectories

        Input:
        -----
        clustid   - cluster ID you want to collect trajectories from
        trajloc   - mapping of the trajectory for each cluster
        qmregion  - quantum region; index of atoms
        Ntrajs    - Number of trajectories you want to pull from each cluster
        Nframes   - Number of frames in the dataset
        rseed     - randomseed to reproduce the shuffling
        rxncoor   - a list of the atomindex of the qm-region the reaction coordinate is defined in
        fxnkey    - operation to perform on rxncoordinate; normally 'dist' for distance
        R_upper   - upperbound of the reactant basin; used for t0
        nmod      - if you wish to print out number of trajs every nmod iterations, flag
        features  - the ML features in an Ntrajs  x Ntimes  x Nfeatures

        Output:
        ------
        Returns dictionary with keys trajs, Ntraj x Nframe x Natoms x 3, and trajnames - the DCD files you read from
        """
        trajs = {}
        trajset = trajloc[clustid]
        trajcoor, trajnames, t0, isrev, rxncoor, fts = choose_trajectories(qmregion, Ntrajs, trajset, Nframes = Nframes, rseed = rseed, fxnkey = fxnkey, R_upper = R_upper, rxncoor_atoms = rxncoor_atoms, nmod = nmod, features = features)
        if features is not None:
            trajs.update({clustid : {"Trajs" : trajcoor, "Trajnames" : trajnames, "t0" : t0, "isrev" : isrev, "lambda" : rxncoor, "geofts" : fts}})
        else:
            trajs.update({clustid : {"Trajs" : trajcoor, "Trajnames" : trajnames, "t0" : t0, "isrev" : isrev, "lambda" : rxncoor, "geofts" : fts}})
        print("Finished collecting trajectories for cluster = " + str(clustid) )
        return trajs


def filtertraj(tset, Nframes, trev, tfor, Ntraj_short, features):
    """
    
    Given trajectories, will slice them so that there is a fixed (trev/tfor) number of timesteps forward/backward in time and t0 is always at the same timepoint.

    """
    #Slice only trajectories whose t0 position provides enough frames before/after (trev/tfor) to allow us to compute the fixed-length q-chem computations we want
    tmpframe = {1: ['for ' + str(idx) for idx in range(int(Nframes/2), 0, -1)] + ['rev ' + str(idx+1) for idx in range(int(Nframes/2))],
                0: ['rev ' + str(idx) for idx in range(int(Nframes/2), 0, -1)] + ['for ' + str(idx+1) for idx in range(int(Nframes/2))],
                }
    valid_trajs = [tidx for tidx in range(len(tset["t0"])) if tset["t0"][tidx] >= trev and tset["t0"][tidx] + tfor + 1 < Nframes]
    assert len(valid_trajs), "There aren't enough trajectories with your trev/tfor criteria. Try a smaller value."
    lambda_i = np.zeros(shape = (len(valid_trajs), trev+tfor + 1))
    #Slice out the relevant trajectories in the new, time-aligned, dataset
    tnames = [tset["Trajnames"][tidx] for tidx in valid_trajs]
    frame_order = {}
    if Ntraj_short is not None:
        valid_trajs = valid_trajs[:Ntraj_short]
    if features is None:
        fts = None
        trajs = np.zeros(shape = (len(valid_trajs), trev + tfor + 1, tset["Trajs"].shape[2], 3))
        for idx, t0idx in enumerate(valid_trajs):
            trajs[idx, :, :, :] = tset["Trajs"][t0idx, tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1), :, :]
            frame_order.update({ idx : tmpframe[tset["isrev"][t0idx]][tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1)] })
            lambda_i[idx, :] = tset["lambda"][t0idx, tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1)]
    else:
        fts = np.zeros(shape = (len(valid_trajs), trev + tfor + 1, features["Nfts"]))
        trajs = np.zeros(shape = (len(valid_trajs), trev + tfor + 1, tset["Trajs"].shape[2], 3))
        for idx, t0idx in enumerate(valid_trajs):
            trajs[idx, :, :, :] = tset["Trajs"][t0idx, tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1), :, :]
            frame_order.update({ idx : tmpframe[tset["isrev"][t0idx]][tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1)] })
            lambda_i[idx, :] = tset["lambda"][t0idx, tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1)]
            fts[idx, :, :] = tset["geofts"][t0idx, tset["t0"][t0idx]-trev:(tfor+tset["t0"][t0idx]+1), :]
    return {"Trajs" : trajs, "Trajnames" : tnames, "frame_order" : frame_order, "lambda" : lambda_i, "geofts" : fts}



def make_qchemfiles(trajs, qchemdir, atomlist, Ntrajs, Nframes, prefix, job_parameters, nbo_parameters, charge, multiplicity, comment, lambda_value = None):
    """
    
    Given a set of trajectories, and a location to place them, the following will make q-chem files.

    Inputs:
    -------
    trajs    - Ntrajectories x Nframes x Natoms x 3 coordinates
    qchemdir - directory location to input the q-chem files
    atomlist - atomtypes 'CHON etc.' for each atom of interest. It should match the order of the QM region
    Ntrajs   - number of trajectories to print
    Nframes  - number of frames to print
    prefix   - reactive/nonreactive + cluster
    job_parameters - jobtype for qchem
    nbo_parameters - nbo options
    charge -charge of system; default is 0
    multiplicity of system; default is 1
    comment - optional comment for the system
    lambda_value - optional, the value of the rxncoordinate

    """
    if lambda_value is None:
        for trajidx in range(Ntrajs):
            xyzcoor = trajs[trajidx, :, :, :]
            comm    = comment[trajidx]
            for fridx in range(xyzcoor.shape[0]):
                trajdir = qchemdir + "traj" + str(trajidx + 1) + "/"
                if os.path.exists(trajdir) is False:
                    os.makedirs(trajdir)
                qchemfile = trajdir + prefix + "_traj" + str(trajidx + 1) + "_fr" + str(fridx + 1) + ".qcin"
                with open(qchemfile, 'w') as f:
                    f.writelines(qchemlines(trajs[trajidx, fridx, :, :], atomlist, job_parameters, nbo_parameters, comm, charge, multiplicity))
    else:
        for trajidx in range(Ntrajs):
            xyzcoor = trajs[trajidx, :, :, :]
            comm    = comment[trajidx]
            for fridx in range(xyzcoor.shape[0]):
                trajdir = qchemdir + "traj" + str(trajidx + 1) + "/"
                if os.path.exists(trajdir) is False:
                    os.makedirs(trajdir)
                qchemfile = trajdir + prefix + "_traj" + str(trajidx + 1) + "_fr" + str(fridx + 1) + ".qcin"
                with open(qchemfile, 'w') as f:
                    f.writelines(qchemlines(trajs[trajidx, fridx, :, :], atomlist, job_parameters, nbo_parameters, lambda_value[trajidx][fridx] + comm, charge, multiplicity))


#This dictionary provides a set of functions to read in the lambda_flag 
#Trajectories x Order Params x Frames
#Trajectories x Order Params x Frames
convert_comment = { dict :       (lambda x: {key : [["".join(["lam " + str(jdx+1) + " : " + str(value[trajidx][jdx][idx])[:5]+ "\n"  for jdx in range(value[trajidx].shape[0])]) for idx in range(value[trajidx].shape[1])] for trajidx in range(value.shape[0])] for key, value in x.items() }),
                    list :       (lambda x : [["".join(["lam " + str(jdx+1) + " : " + str(x[trajidx][jdx][idx])[:5]+ "\n"  for jdx in range(x[0].shape[0])])  for idx in range(x[0].shape[1])] for trajidx in range(len(x))]), 
                    np.ndarray : (lambda x: [["".join(["lam " + str(jdx+1) + " : " + str(x[jdx][idx])[:5]+ "\n"  for jdx in range(x.shape[1])]) for idx in range(x.shape[2])] for trajidx in range(x.shape[0])]),
                    bool : None }



def determine_lambda_comment(clustid, lambda_flag = False):
    """
    Reduces overhead on code; determine the lambda-value if you want it in your q-chem files

    Input:
    clustid - cluster ID
    lambda_flag -What you want your lambda value to be. True, False, or a dictionary/list/etc of what you want

    IT ASSUMES the lambda value (x) is definitely a np.ndarray if you provide it.

    Output:
    Returns a dictionary where each key's value is a list with the lambda values for each trajectory/frame ran

    """
    if isinstance(clustid, list) or isinstance(clustid, np.ndarray):
        #If you want to print the rxncoordinate, add the following:
        if lambda_flag is False:
            lambda_value = {cd : None for cd in clustid}
        else:
            lfxn = convert_comment[type(lambda_flag)]
            lambda_value = lfxn(lambda_flag)
    else:
        if lambda_flag is False:
            lambda_value = None
        else:
            lfxn = convert_comment[type(lambda_flag)]
            lambda_value = lfxn(lambda_flag)
    return lambda_value


def qchemlines(traj, atomlist, job_parameters, nbo_parameters, comment, charge, multiplicity):
    """

    Creates the lines for a qchem file

    trajs - the xyz coordinates for 1 frame
    atomlist - the atomlist you wish to add
    parameters - the jobtype parameters

    """
    lines = ["$rem\n"]
    for key in job_parameters.keys():
        lines.append("   " + str(key) +" " + str(job_parameters[key]) + "\n")
    lines += ["$end\n\n$comment\n"]
    lines += [comment + "\n$end\n\n$nbo\n"]
    for key in nbo_parameters.keys():
        lines.append("   " + str(key) +" " + str(nbo_parameters[key]) + "\n")
    lines += ['$end\n\n$molecule\n']
    lines += ["   " + str(charge) + " " + str(multiplicity)  + "\n"]
    for atom in range(traj.shape[0]):
        xyzcoor = '%12.3f  %8.3f  %8.3f\n' % (traj[atom, 0], traj[atom, 1], traj[atom, 2])
        lines += ['    ' + str(atomlist[atom]) + xyzcoor]
    lines += ["$end\n\n\n\n"]
    return lines


#String Replacement
def replaceCoor(pdblines, xyzcoor, nround = 3):
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



