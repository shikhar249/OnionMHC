# OnionMHC
### Peptide - HLA-A*02:01 binding prediction using structure and sequence feature sets

#### Required Modules
1. Tensorflow 2.0.0 <br />
2. Pandas 0.25.3 <br />
3. scikit-learn 0.21.3 <br />
4. scipy 1.3.2 <br />
5. Numpy 1.17.4 <br />

#### Making Predictions
`python onionmhc.py -struc <structure-based features file> -seq <sequence file> -mod path/to/models/fold{0..4}_model{0..2}_bls_lstm.h5 -out <output file>`
