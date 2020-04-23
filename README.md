# mtl-disparate with multiFC dataset

Requirements:

- Tensorflow 1.5
- Numpy 1.12.1
- sklearn 0.18.1
- scipy

Data organisation:

Please download the data folder from the .zip file in the mail and replace it with this data folder. 

- Foldername: data/meta_domain_evi -- has each domain/task specific train,dev and test dataframes. (claim+evidence+meta)
- preproc/data_reader.py tests if the data readers work.


Steps to run:

- main.py trains models


PS: This is the one with the best scores out of all 6 variants: claim_only, claim+evidence and claim+evidence_ranked and claim_only+meta, claim+evidence+meta and claim+evidence_ranked+meta 

Note: Still in progress