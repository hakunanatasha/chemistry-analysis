# chemistry-analysis

The following contains scripts I developed to process my QM/MM atomistic simulations. Leveraging these calculations allowed me to construct the datapool of electronic features to explore the KARI enzyme mechanism. Over 600K calculations were performed to identify the electonic-level mechanism of KARI.

## make-qchem
Makes Q-Chem files from CHARMM dcd trajectories

My doctoral research required me to develop electronic features from QM/MM atomistic based simulations. The atomistic simulations have binary files with xyz coordinates.
The following script is fairly specific to the workflow that I have in lab, but has default functions that are useful in doing the following:

1. Loading the PDB file of interest
2. Extracting the quantum (or isolated) region of atoms from a PDB
3. Reading in DCD (coordinate binary files) files for atomic xyz coordinates
4. Performing operations (such as calculating distances)

For the specific calculation I performed, I wanted to be able to include R-group side chains in my data. To do so, I had to proton-cap the R-groups. To do this, I took the CB (beta-carbon), and converted the CA (alpha-carbon) into an "H" type. Then, I translated the alpha-carbon closer to the beta carbon to account for the equilibrium bond length between CB-H versus CB-CA.

To get an example of how the scripts are used, see: https://hakunanatasha.github.io/chemistry-analysis/docs/
