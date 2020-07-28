#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import re
import sys
from collections import OrderedDict
from mpi4py import MPI
import argparse
from argparse import RawDescriptionHelpFormatter


class ResidueCounts(object):

    def __init__(self, pdb_fn, ligcode="LIG"):

        self.pdb = mt.load_pdb(pdb_fn)
        self.receptor_ids_ = None
        self.ligand_ids_ = None
        self.resid_pairs_ = None
        self.ligand_n_atoms_ = 0
        self.distance_calculated_ = False

        self.max_pairs_ = 500

        self.top = self.pdb.topology

    def get_receptor_seq(self):

        pattern = re.compile("[A-Za-z]*")

        res_seq = self.top.residues[:-1]
        self.seq = [pattern.match(x).group(0) for x in res_seq]
        self.ligand_n_atoms_ = self.top.select("resid %d" % len(self.seq)).shape[0]

        return self

    def get_resid_pairs(self):

        pairs_ = list(itertools.product(self.receptor_ids_, self.ligand_ids_))
        if len(pairs_) > self.max_pairs_:
            self.resid_pairs_ = pairs_[:self.max_pairs_]
        else:
            self.resid_pairs_ = pairs_

        return self

    def cal_distances(self, residue_pair, ignore_hydrogen=True):

        if ignore_hydrogen:
            indices_a = self.pdb.topology.select("resid %d and element H" % residue_pair[0])
            indices_b = self.pdb.topology.select("resid %d and element H" % residue_pair[1])
        else:
            indices_a = self.pdb.topology.select("resid %d" % residue_pair[0])
            indices_b = self.pdb.topology.select("resid %d" % residue_pair[1])

        pairs = itertools.product(indices_a, indices_b)

        return mt.compute_distances(self.pdb, pairs)[0]

    def contacts_nbyn(self, cutoff, resid_pair):

        #if not self.distance_calculated_:
        distances = np.sum(self.cal_distances(resid_pair) <= cutoff)

        return distances / (self.top.select("resid %d" % resid_pair[0]).shape[0] *
                            self.ligand_n_atoms_)

    def do_preparation(self):
        if self.receptor_ids_ is None:
            self.get_receptor_seq()
        if self.resid_pairs_ is None:
            self.get_resid_pairs()

        return self

    def distances_all_pairs(self, cutoff):
        # do preparation
        self.do_preparation()

        # looping over all pairs
        d = np.zeros(len(self.resid_pairs_))
        for i, p in enumerate(self.resid_pairs_):
            d[i] = self.contacts_nbyn(cutoff, p)

        return d


class AtomTypeCounts(object):
    """Featurization of Protein-Ligand Complex based on
    onion-shape distance counts of atom-types.

    Parameters
    ----------
    pdb_fn : str
        The input pdb file name.
    lig_code : str
        The ligand residue name in the input pdb file.

    Attributes
    ----------
    pdb : mdtraj.Trajectory
        The mdtraj.trajectory object containing the pdb.
    receptor_indices : np.ndarray
        The receptor (protein) atom indices in mdtraj.Trajectory
    ligand_indices : np.ndarray
        The ligand (protein) atom indices in mdtraj.Trajectory
    rec_ele : np.ndarray
        The element types of each of the atoms in the receptor
    lig_ele : np.ndarray
        The element types of each of the atoms in the ligand
    lig_code : str
        The ligand residue name in the input pdb file
    pdb_parsed_ : bool
        Whether the pdb file has been parsed.
    distance_computed : bool
        Whether the distances between atoms in receptor and ligand has been computed.
    distance_matrix_ : np.ndarray, shape = [ N1 * N2, ]
        The distances between all atom pairs
        N1 and N2 are the atom numbers in receptor and ligand respectively.
    counts_: np.ndarray, shape = [ N1 * N2, ]
        The contact numbers between all atom pairs
        N1 and N2 are the atom numbers in receptor and ligand respectively.

    """

    def __init__(self, pdb_fn):

        self.pdb = mt.load(pdb_fn)

        self.receptor_indices = np.array([])
        self.receptor_indices_2 = np.array([])
        self.receptor_indices_3 = np.array([])
        self.ligand_indices = np.array([])
        self.ligand_indices_2 = np.array([])
        self.ligand_indices_3 = np.array([])
 
        self.rec_ele = np.array([])
        self.lig_ele = np.array([])

       # self.lig_code = lig_code

        self.pdb_parsed_ = False
        self.distance_computed_ = False

        self.distance_matrix_ = np.array([])
        self.counts_ = np.array([])

    def parsePDB(self, rec_sele="protein", lig_sele="protein"): # modified by shikhar
        """
        Parse PDB file using mdtraj

        Parameters
        ----------
        rec_sele: str, default is protein.
            The topology selection for the receptor
        lig_sele: str, default is resname LIG
            The topology selection for the ligand

        Returns
        -------
        self: an insttable, bond = top.to_dataframe()ance of itself

        """
        top = self.pdb.topology

        self.receptor_indices = top.select('protein and chainid 0 1') # replace rec_sele with "protein and not resname HIS TYR PHE TYR" 
        self.receptor_indices_2 = top.select('resname HIS TYR TRP PHE and symbol C and chainid 0 1 and not name C CA CB') # added by shikhar
        self.receptor_indices_3 = top.select('resname TYR SER THR and symbol O and chainid 0 1 and not name O OXT')
        self.ligand_indices = top.select('protein and chainid 2')   # modified by shikhar
        self.ligand_indices_2 = top.select('resname HIS TYR TRP PHE and symbol C and chainid 2 and not name C CA CB') # added by shikhar
        self.ligand_indices_3 = top.select('resname TYR SER THR and symbol O and chainid 2 and not name O OXT')

        table, bond = top.to_dataframe()

        self.rec_ele = table['element'][self.receptor_indices] # replace element with "name"
        self.lig_ele = table['element'][self.ligand_indices]
	
        for i in self.receptor_indices_2:	# added by shikhar
        	self.rec_ele[i] = "CAM"
        for i in self.receptor_indices_3:
        	self.rec_ele[i] = "OH"

        for i in self.ligand_indices_2:	#added by shikhar
        	self.lig_ele[i] = "CAM"
        for i in self.ligand_indices_3:
        	self.lig_ele[i] = "OH"

        self.pdb_parsed_ = True

        return self

    def distance_pairs(self):
        """Calculate all distance pairs between atoms in the receptor and in the ligand

        Returns
        -------
        self: an instance of itself
        """

        if not self.pdb_parsed_:
            self.parsePDB()

        # all combinations of the atom indices from the receptor and the ligand
        all_pairs = itertools.product(self.receptor_indices, self.ligand_indices)

        # if distance matrix is not calculated
        if not self.distance_computed_:
            self.distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=all_pairs)[0]

        self.distance_computed_ = True

        return self

    def cutoff_count(self, cutoff=0.35):
        """
        Get the atom contact matrix

        Parameters
        ----------
        cutoff: float, default is 0.35 angstrom
            The distance cuntoff for the contacts

        Returns
        -------
        self: return an instance of itself
        """
        # get the inter-molecular atom contacts
        self.counts_ = (self.distance_matrix_ <= cutoff) * 1.0

        return self


def generate_features(complex_fn, ncutoffs, all_elements, keys):

    #all_elements = ["H", "C", "O", "N", "P", "S", "Br", "Du"]
    #keys = ["_".join(x) for x in list(itertools.product(all_elements, all_elements))]

    # parse the pdb file and get the atom element information
    cplx = AtomTypeCounts(complex_fn)  #, lig_code)
    cplx.parsePDB(rec_sele="protein and chainid 0 1", lig_sele="protein and chainid 2" )    #modified by shikhar

    # element types of all atoms in the proteins and ligands
    new_lig = [x if x in all_elements else "Du" for x in cplx.lig_ele]
    new_rec = [x if x in all_elements else "Du" for x in cplx.rec_ele]

    # the element-type combinations for all atom-atom pairs
    rec_lig_element_combines = ["_".join(x) for x in list(itertools.product(new_rec, new_lig))]
    cplx.distance_pairs()

    counts = []
    onion_counts = []

    # calculate all contacts for all shells
    for i, cutoff in enumerate(ncutoffs):
        cplx.cutoff_count(cutoff)
        if i == 0:
            onion_counts.append(cplx.counts_)
        else:
            onion_counts.append(cplx.counts_ - counts[-1])
        counts.append(cplx.counts_)

    results = []

    for n in range(len(ncutoffs)):
        #count_dict = dict.fromkeys(keys, 0.0)
        d = OrderedDict()
        d = d.fromkeys(keys, 0.0)
        # now sort the atom-pairs and accumulate the element-type to a dict
        for e_e, c in zip(rec_lig_element_combines, onion_counts[n]):
            d[e_e] += c

        results += d.values()

    return results, keys


if __name__ == "__main__":

    print("Start Now ... ")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    d = """
    Predicting protein-ligand binding affinities (pKa) with OnionNet model. 
    Citation: Coming soon ... ...
    Author: Liangzhen Zheng
    
    This script is used to generate inter-molecular element-type specific 
    contact features. Installation instructions should be refered to 
    https://github.com/zhenglz/onionnet

    Examples:
    Show help information
    python generate_features.py -h
    
    Run the script with one CPU core
    python generate_features.py -inp input_samples.dat -out features_samples.csv
    
    Run the script with MPI 
    mpirun -np 16 python generate_features.py -inp input_samples.dat -out features_samples.csv
    
    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input. The input file containg the file path of each \n"
                             "of the protein-ligand complexes files (in pdb format.)\n"
                             "There should be only 1 column, each row or line containing\n"
                             "the input file path, relative or absolute path.")
    parser.add_argument("-out", type=str, default="output.csv",
                        help="Output. Default is output.csv \n"
                             "The output file name containing the features, each sample\n"
                             "per row. ")
   # parser.add_argument("-lig", type=str, default="LIG",
   #                     help="Input, optional. Default is LIG. \n"
   #                          "The ligand molecule residue name (code, 3 characters) in the \n"
   #                          "complex pdb file. ")
    parser.add_argument("-start", type=float, default=0.1,
                        help="Input, optional. Default is 0.05 nm. "
                             "The initial shell thickness. ")
    parser.add_argument("-end", type=float, default=3.0,
                        help="Input, optional. Default is 3.05 nm. "
                             "The boundary of last shell.")
    parser.add_argument("-delta", type=float, default=0.05,
                        help="Input, optional. Default is 0.05 nm. "
                             "The thickness of the shells.")
    parser.add_argument("-n_shells", type=int, default=60,
                        help="Input, optional. Default is 60. "
                             "The number of shells for featurization. ")

    args = parser.parse_args()

    all_elements = ["H", "C", "O", "N", "S", "CAM", "OH","Du"]   #remove Br and add CAM (shikhar)
    keys = ["_".join(x) for x in list(itertools.product(all_elements, all_elements))]

    if rank == 0:
        if len(sys.argv) < 3:
            parser.print_help()
            sys.exit(0)

        # spreading the calculating list to different MPI ranks
        with open(args.inp) as lines:
            lines = [x for x in lines if ("#" not in x and len(x.split()) >= 1)].copy()
            inputs = [x.split()[0] for x in lines]

        inputs_list = []
        aver_size = int(len(inputs) / size)
        print(size, aver_size)
        for i in range(size-1):
            inputs_list.append(inputs[int(i*aver_size):int((i+1)*aver_size)])
        inputs_list.append(inputs[(size-1)*aver_size:])

    else:
        inputs_list = None

    inputs = comm.scatter(inputs_list, root=0)

    # defining the shell structures ... (do not change)
    n_cutoffs = list(np.linspace(args.start, args.end, args.n_shells))
    print(n_cutoffs)

    results = []
    ele_pairs =[]

    # computing the features now ...
    for p in inputs:
        fn = p
       # lig_code = args.lig

#        try:
            # the main function for featurization ...
        r, ele_pairs = generate_features(fn, n_cutoffs, all_elements, keys)
        print(rank, fn)

#        except:
#            print("error")
#            r = [0., ] * 64 * args.n_shells
#            print(rank, "Not successful. ", fn)

        results.append(r)

    # saving features to a file now ... 
    df = pd.DataFrame(results)
    try:
        df.index = inputs
    except:
        df.index = np.arange(df.shape[0])
    print(df.shape)
    col_n = []
    for i, n in enumerate(keys * args.n_shells):
        col_n.append(n+"_"+str(i))
    df.columns = col_n
    df.to_csv("rank%d_"%rank+args.out, sep=",", float_format="%.1f", index=True)

    print(rank, "Complete calculations. ")

