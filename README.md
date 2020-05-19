Fake News Classification of Multi-source Dataset with DisparateLabel Space Using Multi-task Learning and Label Embedding
This is the code repository for the final project of course: DAT550, Spring 2020. 

Requirements:

- Tensorflow 1.5
- Numpy 1.12.1
- sklearn 0.18.1
- scipy

Data organisation:

- Due to confidentiality issues, the data is not included here. But it can be found at: https://competitions.codalab.org/competitions/21163#learn_the_details 

- preproc/data_reader.py tests if the data readers work. But this works only on our preprocessed data which will be provided on request. 
- The folder /explore has the python scripts used for the data exploratory works. 

Steps to run:

- Run main.py 

This folder consists of the code to run the best performing model variant: claim+evidence+meta
PS: There are totally 6 variants: claim_only, claim+evidence and claim+evidence_ranked and claim_only+meta, claim+evidence+meta and claim+evidence_ranked+meta 

Results:

- The domain-wise F1 Macro and Micro scores for all 6 variants can be found in scores.xlsx 