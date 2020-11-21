# dpdilca
Family of algorithms for differentially private distance learning in categorical data.

The main file is dilca_dp_final.py, in 'algorithms' folder. Folder 'datasets' contains the real-world datasets used in our experiments.

The following files can be used for reproduce the experiments:
 
 - dilca_test.py: comparison between distances computed by DP-DILCA and non-private DILCA, for each attribute of the dataset.
 
 - dilca_test_objects.py: comparison between the pairwise objects distances computed by DP-DILCA and non-private DILCA.
 
 - Final_Expe_clustering.py: applies Ward's hierarchical clustering to the pairwise distance matrix computed by DP-DILCA.

 - Final_Expe_kNN.py: applies kNN classifier to the pairwise distance matrix computed by DP-DILCA.
 
 - Final_Expe_DP.py: applies DP-KMeans algorithm to the pairwise distance matrix computed by DP-DILCA (it requires the implementation of DP-KMeans provided in https://github.com/IBM/differential-privacylibrary)
 
 - test_synth.py: experiments on synthetic data. It assesses the quality of the context selection phase in a controlled environment.
