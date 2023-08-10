# OnionMHC

The source code for our paper: [OnionMHC: A deep learning model for peptide — HLA-A*02:01 binding predictions using both structure and sequence feature sets](https://www.worldscientific.com/doi/10.1142/S2424913020500095)
### Peptide - HLA-A*02:01 binding prediction using structure and sequence feature sets

<p align="center">
  <img width="460" height="250" src="https://github.com/shikhar249/OnionMHC/blob/master/image.png">
</p>

### Required Modules
1. Tensorflow 2.0.0 <br />
2. Pandas 0.25.3 <br />
3. scikit-learn 0.21.3 <br />
4. scipy 1.3.2 <br />
5. Numpy 1.17.4 <br />

### Generating structure based features
Structure based features can be generated by running `generate_features_cam_o.py`. Each line in the input file `<input.dat>` contains the path to the pdb structures. 
```
python generate_features_cam_o.py -inp <input.dat> -out <output.csv>
```

Parallelizing generation of structure based features using mpirun

```
mpirun -np 16 python generate_features_cam_o.py -inp <input.dat> -out <output.csv>
```

### Making Predictions
1. Clone this repository <br />
```
git clone https://github.com/shikhar249/OnionMHC
```

2. Run onionmhc.py <br />
```
python onionmhc.py -struc <structure-based features file> -seq <sequence file> -mod path/to/models/fold{0..4}_model{0..2}_bls_lstm.h5 -out <output file>
```
The example of prediction results will be shown as:

| peptide_sequences | OnionMHC_score | Binding_affinity(nM) |
| -----------------| :-----------: | :---------: |
| FLIAYQPLL  |      0.901299    |          2.909335|
| NLLTTPKFT  |      0.483376    |        267.669274|
|GTHSWEYWG   |     0.062278     |     25487.509380 |
|... | ... | ...|

### Citations
Please cite our paper
```
@article{doi:10.1142/S2424913020500095,
author = {Saxena, Shikhar and Animesh, Sambhavi and Fullwood, Melissa J. and Mu, Yuguang},
title = {OnionMHC: A deep learning model for peptide — HLA-A*02:01 binding predictions using both structure and sequence feature sets},
journal = {Journal of Micromechanics and Molecular Physics},
volume = {05},
number = {03},
pages = {2050009},
year = {2020},
doi = {10.1142/S2424913020500095}
```
