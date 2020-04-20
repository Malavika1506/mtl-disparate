# mtl-disparate with multiFC dataset

Requirements:

- Tensorflow 1.5
- Numpy 1.12.1
- sklearn 0.18.1
- scipy

Data organisation:

Please download the data folder from the .zip file in the mail and replace it with this data folder. 

- Foldername: data/evidence -- has each domain/task specific train,dev and test dataframes. (claim+evidence)
- Foldername: data/evidence_ranked -- has each domain/task specific train,dev and test dataframes. (claim+evidence_ranked)
- preproc/data_reader.py tests if the data readers work.


Steps to run:

- main.py trains models


Already computed Results:

- Foldername: results/only_claim -- has saved_features, log.txt, log_inds.txt for every domain/task. (claim_only)
- Foldername: results/evidence -- has saved_features, log.txt, log_inds.txt for every domain/task. (claim+evidence)
- Foldername: results/evidence_ranked -- has saved_features, log.txt, log_inds.txt for every domain/task. (claim+evidence_ranked)


Note: Still in progress