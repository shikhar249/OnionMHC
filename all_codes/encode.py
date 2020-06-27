import numpy as np

def one_hot(sequences):
    #if sequences is None:
    #    print("Error seq not given")
    #    return
    lengths = [len(x) for x in sequences]
    aa_list = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    mat = np.zeros((len(sequences), max(lengths), 20), dtype=int)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            mat[i, j, aa_list.index(aa)] += 1
    return mat

def blosum_encode(sequences):
    lengths = [len(x) for x in sequences]
    mat = np.zeros((len(sequences), max(lengths), 20), dtype=int)
    blosum_dict = {
        "A":[4,0,-2,-1,-2,0,-2,-1,-1,-1,-1,-2,-1,-1,-1,1,0,0,-3,-2],
"C":[0,9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],
"D":[-2,-3,6,2,-3,-1,-1,-3,-1,-4,-3,1,-1,0,-2,0,-1,-3,-4,-3],
"E":[-1,-4,2,5,-3,-2,0,-3,1,-3,-2,0,-1,2,0,0,-1,-2,-3,-2],
"F":[-2,-2,-3,-3,6,-3,-1,0,-3,0,0,-3,-4,-3,-3,-2,-2,-1,1,3],
"G":[0,-3,-1,-2,-3,6,-2,-4,-2,-4,-3,0,-2,-2,-2,0,-2,-3,-2,-3],
"H":[-2,-3,-1,0,-1,-2,8,-3,-1,-3,-2,1,-2,0,0,-1,-2,-3,-2,2],
"I":[-1,-1,-3,-3,0,-4,-3,4,-3,2,1,-3,-3,-3,-3,-2,-1,3,-3,-1],
"K":[-1,-3,-1,1,-3,-2,-1,-3,5,-2,-1,0,-1,1,2,0,-1,-2,-3,-2],
"L":[-1,-1,-4,-3,0,-4,-3,2,-2,4,2,-3,-3,-2,-2,-2,-1,1,-2,-1],
"M":[-1,-1,-3,-2,0,-3,-2,1,-1,2,5,-2,-2,0,-1,-1,-1,1,-1,-1],
"N":[-2,-3,1,0,-3,0,1,-3,0,-3,-2,6,-2,0,0,1,0,-3,-4,-2],
"P":[-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2,7,-1,-2,-1,-1,-2,-4,-3],
"Q":[-1,-3,0,2,-3,-2,0,-3,1,-2,0,0,-1,5,1,0,-1,-2,-2,-1],
"R":[-1,-3,-2,0,-3,-2,0,-3,2,-2,-1,0,-2,1,5,-1,-1,-3,-3,-2],
"S":[1,-1,0,0,-2,0,-1,-2,0,-2,-1,1,-1,0,-1,4,1,-2,-3,-2],
"T":[0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,0,-1,-1,-1,1,5,0,-2,-2],
"V":[0,-1,-3,-2,-1,-3,-3,3,-2,1,1,-3,-2,-2,-3,-2,0,4,-3,-1],
"W":[-3,-2,-4,-3,1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11,2],
"Y":[-2,-2,-3,-2,3,-3,2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1,2,7]}
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            mat[i, j, :] = blosum_dict[aa]
    return mat

def embed(sequences):
    
    lengths = [len(x) for x in sequences]
    
    emb_code = {"A": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6,"H": 7,"I": 8,"K": 9,"L": 10,"M": 11,"N": 12,"P": 13,"Q": 14,"R":15,"S":16,"T":17,"V":18,"W":19,"Y":20}
    
    mat = np.zeros((len(sequences), max(lengths)), dtype=int)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            mat[i, j] = emb_code[aa]
    return mat