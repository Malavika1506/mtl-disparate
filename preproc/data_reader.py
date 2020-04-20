import csv, os
import xml.etree.ElementTree as ET
from collections import defaultdict

from sklearn.model_selection import train_test_split


from constants import ABBC,AFCK,BOVE,CHCT,CLCK,FAAN,FALY,FANI,FARG,GOOP,HOER,HUCA,MPWS,OBRY,\
    PARA,PECK,POMT,POSE,RANZ,SNES,THAL,THET,TRON,VEES,VOGO,WAST,\
        ABBC_LABELS,AFCK_LABELS,BOVE_LABELS,CHCT_LABELS,CLCK_LABELS,FAAN_LABELS,FALY_LABELS,FANI_LABELS,FARG_LABELS, \
        GOOP_LABELS,HOER_LABELS,HUCA_LABELS,MPWS_LABELS,OBRY_LABELS,PARA_LABELS,PECK_LABELS,POMT_LABELS,POSE_LABELS, \
            RANZ_LABELS,SNES_LABELS,THAL_LABELS,THET_LABELS,TRON_LABELS,VEES_LABELS,VOGO_LABELS,WAST_LABELS


def task2data_reader(task):
    if task == ABBC: 
	    return readABBC
    if task == AFCK: 
        return readAFCK
    if task == BOVE: 
        return readBOVE
    if task == CHCT: 
        return readCHCT
    if task == CLCK: 
        return readCLCK
    if task == FAAN: 
        return readFAAN
    if task == FALY: 
        return readFALY
    if task == FANI: 
        return readFANI
    if task == FARG: 
        return readFARG
    if task == GOOP: 
        return readGOOP
    if task == HOER: 
        return readHOER
    if task == HUCA: 
        return readHUCA
    if task == MPWS: 
        return readMPWS
    if task == OBRY: 
        return readOBRY
    if task == PARA: 
        return readPARA
    if task == PECK: 
        return readPECK
    if task == POMT: 
        return readPOMT
    if task == POSE: 
        return readPOSE
    if task == RANZ: 
        return readRANZ
    if task == SNES: 
        return readSNES
    if task == THAL: 
        return readTHAL
    if task == THET: 
        return readTHET
    if task == TRON: 
        return readTRON
    if task == VEES: 
        return readVEES
    if task == VOGO: 
        return readVOGO
    if task == WAST: 
        return readWAST
    raise ValueError('No data reader available for %s.' % task)


def readABBC(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("abbc",ABBC_LABELS)
def readAFCK(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("afck",AFCK_LABELS)
def readBOVE(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("bove",BOVE_LABELS)
def readCHCT(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("chct",CHCT_LABELS)
def readCLCK(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("clck",CLCK_LABELS)
def readFAAN(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("faan",FAAN_LABELS)
def readFALY(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("faly",FALY_LABELS)
def readFANI(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("fani",FANI_LABELS)
def readFARG(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("farg",FARG_LABELS)
def readGOOP(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("goop",GOOP_LABELS)
def readHOER(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("hoer",HOER_LABELS)
def readHUCA(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("huca",HUCA_LABELS)
def readMPWS(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("mpws",MPWS_LABELS)
def readOBRY(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("obry",OBRY_LABELS)
def readPARA(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("para",PARA_LABELS)
def readPECK(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("peck",PECK_LABELS)
def readPOMT(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("pomt",POMT_LABELS)
def readPOSE(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("pose",POSE_LABELS)
def readRANZ(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("ranz",RANZ_LABELS)
def readSNES(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("snes",SNES_LABELS)
def readTHAL(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("thal",THAL_LABELS)
def readTHET(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("thet",THET_LABELS)
def readTRON(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("tron",TRON_LABELS)
def readVEES(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("vees",VEES_LABELS)
def readVOGO(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("vogo",VOGO_LABELS)
def readWAST(datafolder="./data/evidence_ranked/",debug=True,num_instances=270):
    return read_task("wast",WAST_LABELS)




def read_task(task, labels, datafolder="./data/evidence_ranked/", debug=True, num_instances=270):
    stance=[]
    folder= ((task))+'/'

    data_train = {"seq1": [], "seq2": [], "stance": [], "labels": []}
    data_train = parse1(datafolder+folder, "train.csv", data_train, debug, num_instances)
    
    data_dev = {"seq1": [], "seq2": [], "stance": [], "labels": []}
    data_dev = parse1(datafolder+folder, "dev.csv", data_dev, debug, num_instances)
    
    #stance=data_train["stance"]
    data_test = {"seq1": [], "seq2": [], "stance": [], "labels": []}
    data_test = parse1(datafolder+folder, "test.csv", data_test, debug, num_instances)
    
    #print(sorted(data_train["labels"]))
    #print(sorted(labels))

    data_train["labels"] = sorted(labels)
    data_dev["labels"] = data_train["labels"]
    data_test["labels"] = data_train["labels"]

    #assert sorted(data_train["labels"]) == sorted(labels)
    #print(task)
    #print(sorted(labels))
    #print(data_test["labels"])
    '''
    print(data_train)
    print('*********************************************')
    print(data_dev)
    print('*********************************************')
    print(data_test)
    print('*********************************************')
    '''
    return data_train, data_dev, data_test

def parse1(datafolder, datafile_bodies,  data_dict, debug, num_instances):
    id2body = {}
    with open(os.path.join(datafolder, datafile_bodies), 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = -1
        for row in csvreader:
            i += 1
            if i == 0:
                continue
            k, domain,claimID,claim,snippet,label = row 
            '''
            # claim and evidence or evidence_ranked
            id2body[claimID] = snippet
            
            data_dict["seq1"].append(claim)
            data_dict["seq2"].append(id2body[claimID])
            data_dict["stance"].append(label)
            '''
            # Claim - only
            id2body[claimID] = claim
            
            data_dict["seq1"].append(claimID)
            data_dict["seq2"].append(id2body[claimID])
            data_dict["stance"].append(label)
            


    for lab in set(data_dict["stance"]):
        data_dict["labels"].append(lab)

    return data_dict









