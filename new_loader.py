import os
import csv
from sklearn.model_selection import train_test_split
import json

# paths of datasets
BENIGN_CENTRIC_PATH = 'datasets/benign_process_centric/'
RW_CENTRIC_PATH = 'datasets/ransomware_process_centric/'
RW_SINGLESPLIT_PATH = 'datasets/single_functional_splitting/'
RW_COMBINEDSPLIT1_PATH = 'datasets/combined_splitting_dlrd_wtrn/'
RW_COMBINEDSPLIT2_PATH = 'datasets/combined_splitting_dlwt_rdrn/'

NR_SPLITS_LIST_REDUCED = list(range(1,11)) 
NR_SPLITS_LIST = list(range(1,11)) + list(range(15, 55, 5))
BENIGN = 'benign'
MALIGN = 'ransomware'
BENIGN_LABEL = [1]
MALIGN_LABEL = [-1]

tierToTicks = {
    '1': [0.1, 0.13, 0.17, 0.22, 0.29, 0.37, 0.48, 0.63, 0.82, 1, 1.38,
        1.79, 2.3, 3, 3.9, 5, 6.65, 8.65, 11.25, 14.65, 19, 24.7, 32.1, 41, 54, 70.5, 91, 100],
    '2': [0.13,  0.22, 0.37,  0.63, 1, 1.79, 3, 5, 8.65, 14.65, 24.7, 41, 70.5, 100],
    '3': [0.17, 0.37, 0.82, 1.79, 3.9, 8.65, 19, 41, 100],
    '4': [0.22, 0.63, 1.79, 5, 14.65, 41, 100],
    '5': [0.29, 1, 3.9, 14.65, 54, 100],
    '6': [0.37, 1.79, 8.65, 41, 100],
    '7': [0.48, 3, 19, 100],
    '8': [0.63, 5, 41, 100],
    '9': [0.82, 8.65, 100],
    '10': [1, 14.65, 100],
    '11': [1.38, 24.7, 100],
    '12': [1.79, 41, 100],
    '13': [2.3, 70.5, 100],
    '14': [3, 100],
    '15': [3.9, 100],
    '16': [5, 100],
    '17': [6.65, 100],
    '18': [8.65, 100],
    '19': [11.25, 100],
    '20': [14.65, 100],
    '21': [19, 100],
    '22': [24.7, 100],
    '23': [32.1, 100],
    '24': [41, 100],
    '25': [54, 100],
    '26': [70.5, 100],
    '27': [91, 100],
    '28': [100]
}

def get_classic_dataset(type_of_dataset):
    # type_of_dataset can be BENIGN or MALIGN
    if type_of_dataset==BENIGN:
        path = BENIGN_CENTRIC_PATH
    elif type_of_dataset==MALIGN:
        path = RW_CENTRIC_PATH
    else:
        print('ERROR')
        return
    
    # initialization
    tier2ticks2fv = dict()
    for tier in tierToTicks:
        tier2ticks2fv[tier] = dict()
        for tick in range(len(tierToTicks[tier])):
            tick = str(tick)
            tier2ticks2fv[tier][tick] = list()

    for part in os.listdir(path):
        for tier in range(1,29):
            tier = str(tier)
            tier_folder = path + part + "/tier" + tier
            
            if not os.path.isdir(tier_folder): continue            
            
            for file in os.listdir(tier_folder):
                tick = file.split('.csv')[0].split("_")[1]
                file = tier_folder + "/" + file
            
                with open(file, newline='\n') as csv_file:                 
                    csv_reader = list(csv.reader(csv_file, delimiter=','))

                    tier2ticks2fv[tier][tick] += csv_reader
    
    return tier2ticks2fv

def get_single_dataset():
    # initialization
    nrproc2tier2ticks2fv = dict()
    for nrproc in NR_SPLITS_LIST:
        nrproc = str(nrproc)
        nrproc2tier2ticks2fv[nrproc] = dict()
        for tier in tierToTicks:
            nrproc2tier2ticks2fv[nrproc][tier] = dict()
            for tick in range(len(tierToTicks[tier])):
                tick = str(tick)
                nrproc2tier2ticks2fv[nrproc][tier][tick] = list()
            
    # SINGLE FUNCTIONAL SPLITTING DATASET
    for part in os.listdir(RW_SINGLESPLIT_PATH):
        for nrprocessfolder in os.listdir(RW_SINGLESPLIT_PATH + part):
            path1 = RW_SINGLESPLIT_PATH + part + '/' + nrprocessfolder
            nrproc = nrprocessfolder.split('nprocess')[1]
        
            for tierfolder in os.listdir(path1):
                tier = tierfolder.split('tier')[1]
                path2 = path1 + '/' + tierfolder + '/'
                for file in os.listdir(path2):
                    tick = file.split('.csv')[0]
                    file = path2 + '/' + file
                        
                    with open(file) as csv_file:
                        csv_reader = list(csv.reader(csv_file, delimiter=','))
                        # in this case, need to remove blank lines (newline argument doesn't work)
                        noblankfvs = [x for x in csv_reader if x != []]
                        
                        nrproc2tier2ticks2fv[nrproc][tier][tick] += noblankfvs
                        
    return nrproc2tier2ticks2fv

def get_single_dataset_2():
    '''Second method for single dataset that produces a dataset not depending on nrprocesses'''
    # initialization
    tier2ticks2fv = dict()
    for tier in tierToTicks:
        tier2ticks2fv[tier] = dict()
        for tick in range(len(tierToTicks[tier])):
            tick = str(tick)
            tier2ticks2fv[tier][tick] = list()
            
    # SINGLE FUNCTIONAL SPLITTING DATASET
    for part in os.listdir(RW_SINGLESPLIT_PATH):
        for nrprocessfolder in os.listdir(RW_SINGLESPLIT_PATH + part):
            path1 = RW_SINGLESPLIT_PATH + part + '/' + nrprocessfolder
            # nrproc = int(nrprocessfolder.split('nprocess')[1])
            # if nrproc > 10:
            #     continue
        
            for tierfolder in os.listdir(path1):
                tier = tierfolder.split('tier')[1]
                path2 = path1 + '/' + tierfolder + '/'
                for file in os.listdir(path2):
                    tick = file.split('.csv')[0]
                    file = path2 + '/' + file
                        
                    with open(file) as csv_file:
                        csv_reader = list(csv.reader(csv_file, delimiter=','))
                        # in this case, need to remove blank lines (newline argument doesn't work)
                        noblankfvs = [x for x in csv_reader if x != []]
                        
                        tier2ticks2fv[tier][tick] += noblankfvs
                        
    return tier2ticks2fv

def get_comb_dataset(number):
    # number can be 1 or 2
    if number==1:
        path = RW_COMBINEDSPLIT1_PATH
    elif number==2:
        path = RW_COMBINEDSPLIT2_PATH
    else:
        print("ERROR")
        return
        
    # initialization
    nrproc2tier2ticks2fv = dict()
    for nrproc in NR_SPLITS_LIST:
        nrproc = str(nrproc)
        nrproc2tier2ticks2fv[nrproc] = dict()
        for tier in tierToTicks:
            nrproc2tier2ticks2fv[nrproc][tier] = dict()
            for tick in range(len(tierToTicks[tier])):
                tick = str(tick)
                nrproc2tier2ticks2fv[nrproc][tier][tick] = list()
                
    # COMBINED SPLITTING (DL-DR, WT-RN) DATASET (1)
    # COMBINED SPLITTING (DL-WT, RD-RN) DATASET (2)
    for part in os.listdir(path):
        for nrprocessfolder in os.listdir(path + part):
            nrproc = nrprocessfolder.split('nprocess')[1]
        
            for tier_folder in os.listdir(path + part + "/" + nrprocessfolder):
                tier = tier_folder.split('tier')[1]
                tier_folder = path + part + "/" + nrprocessfolder + "/" + tier_folder
                for file in os.listdir(tier_folder):
                    tick = file.split('.csv')[0]
                    file = tier_folder + "/" + file
                    
                    with open(file, newline='\n') as csv_file:
                        csv_reader = list(csv.reader(csv_file, delimiter=','))
                        
                        nrproc2tier2ticks2fv[nrproc][tier][tick] += csv_reader
    
    return nrproc2tier2ticks2fv

def get_comb_dataset_2(number):
    '''Second method for combined dataset that produces a dataset not depending on nrprocesses'''
    # number can be 1 or 2
    if number==1:
        path = RW_COMBINEDSPLIT1_PATH
    elif number==2:
        path = RW_COMBINEDSPLIT2_PATH
    else:
        print("ERROR")
        return
        
    # initialization
    tier2ticks2fv = dict()
    for tier in tierToTicks:
        tier2ticks2fv[tier] = dict()
        for tick in range(len(tierToTicks[tier])):
            tick = str(tick)
            tier2ticks2fv[tier][tick] = list()
                
    # COMBINED SPLITTING (DL-DR, WT-RN) DATASET (1)
    # COMBINED SPLITTING (DL-WT, RD-RN) DATASET (2)
    for part in os.listdir(path):
        for nrprocessfolder in os.listdir(path + part):
            # nrproc = int(nrprocessfolder.split('nprocess')[1])
            # if nrproc > 10:
            #     continue
        
            for tier_folder in os.listdir(path + part + "/" + nrprocessfolder):
                tier = tier_folder.split('tier')[1]
                tier_folder = path + part + "/" + nrprocessfolder + "/" + tier_folder
                for file in os.listdir(tier_folder):
                    tick = file.split('.csv')[0]
                    file = tier_folder + "/" + file
                    
                    with open(file, newline='\n') as csv_file:
                        csv_reader = list(csv.reader(csv_file, delimiter=','))
                        
                        tier2ticks2fv[tier][tick] += csv_reader
    
    return tier2ticks2fv


def get_alltrainset_and_parametric_testsets():
    '''
            
    # per ogni tier-tick, si splitta il dataset corrispondente di single e di combined in trainset e testset
    # i testset si tengono tutti da parte, e si addestrano le reti con tutti gli altri dati
    # per ogni nr_processes, si si prende uno di questi e si usa come testset
    
    '''
    
    # getting all the 5 datasets
    benign_fv   = get_classic_dataset(BENIGN)
    print("benign loaded")
    rans_fv     = get_classic_dataset(MALIGN)
    print("malign loaded")
    single_fv   = get_single_dataset()    # different structure
    print("single loaded")
    comb1_fv    = get_comb_dataset(1)     # different structure
    print("comb1 loaded")
    comb2_fv    = get_comb_dataset(2)     # different structure
    print("comb2 loaded")
    
    nrproc2tiers2ticks2x_single     = dict()
    nrproc2tiers2ticks2x_comb       = dict()
    nrproc2tiers2ticks2y_single     = dict()
    nrproc2tiers2ticks2y_comb       = dict()
    tier2ticks2xtrain               = dict()
    tier2ticks2ytrain               = dict()
    for nrproc in NR_SPLITS_LIST:
        nrproc = str(nrproc)
        nrproc2tiers2ticks2x_single[nrproc]  = dict()
        nrproc2tiers2ticks2x_comb[nrproc]    = dict()
        nrproc2tiers2ticks2y_single[nrproc]  = dict()
        nrproc2tiers2ticks2y_comb[nrproc]    = dict()
        
        for tier in tierToTicks:
            tier = str(tier)
            nrproc2tiers2ticks2x_single[nrproc][tier]   = dict()
            nrproc2tiers2ticks2x_comb[nrproc][tier]     = dict()
            nrproc2tiers2ticks2y_single[nrproc][tier]   = dict()
            nrproc2tiers2ticks2y_comb[nrproc][tier]     = dict()
            tier2ticks2xtrain[tier]                     = dict()
            tier2ticks2ytrain[tier]                     = dict()
            
            for tick in range(len(tierToTicks[tier])):
                tick = str(tick)
                nrproc2tiers2ticks2x_single[nrproc][tier][tick] = list()
                nrproc2tiers2ticks2x_comb[nrproc][tier][tick]   = list()
                nrproc2tiers2ticks2y_single[nrproc][tier][tick] = list()
                nrproc2tiers2ticks2y_comb[nrproc][tier][tick]   = list()
                tier2ticks2xtrain[tier][tick]                   = list()
                tier2ticks2ytrain[tier][tick]                   = list()
    
    # merging the datasets
    for tier in tierToTicks:
        tier = str(tier)
        for tick in range(len(tierToTicks[tier])):
            tick = str(tick)
            # add a part of benign dataset and all rans dataset
            # try:
            #     benign_x_train, benign_x_test, benign_y_train, benign_y_test = train_test_split(benign_fv[tier][tick], BENIGN_LABEL * len(benign_fv[tier][tick]), test_size=0.33)
            # except:
            #     benign_x_train = benign_fv[tier][tick]
            #     benign_y_train = BENIGN_LABEL * len(benign_fv[tier][tick])
            #     benign_x_test = list()
            #     benign_y_test = list()
            
            tier2ticks2xtrain[tier][tick] += benign_fv[tier][tick] + rans_fv[tier][tick]
            tier2ticks2ytrain[tier][tick] += BENIGN_LABEL * len(benign_fv[tier][tick]) + MALIGN_LABEL * len(rans_fv[tier][tick])
            
            for nrproc in NR_SPLITS_LIST:
                nrproc = str(nrproc)
                # split corresponding single dataset if possible, else trainset empty
                try:
                    single_x_train, single_x_test, single_y_train, single_y_test = train_test_split(single_fv[nrproc][tier][tick], MALIGN_LABEL * len(single_fv[nrproc][tier][tick]), test_size=0.33)
                except:
                    single_x_train = single_fv[nrproc][tier][tick]
                    single_y_train = MALIGN_LABEL * len(single_fv[nrproc][tier][tick])
                    single_x_test = list()
                    single_y_test = list()
                    
                # split corresponding combined datasets if possible, else trainset empty
                try:
                    comb1_x_train, comb1_x_test, comb1_y_train, comb1_y_test = train_test_split(comb1_fv[nrproc][tier][tick], MALIGN_LABEL * len(comb1_fv[nrproc][tier][tick]), test_size=0.33)
                except:
                    comb1_x_train = comb1_fv[nrproc][tier][tick]
                    comb1_y_train = MALIGN_LABEL * len(comb1_fv[nrproc][tier][tick])
                    comb1_x_test = list()
                    comb1_y_test = list()
                    
                try:
                    comb2_x_train, comb2_x_test, comb2_y_train, comb2_y_test = train_test_split(comb2_fv[nrproc][tier][tick], MALIGN_LABEL * len(comb2_fv[nrproc][tier][tick]), test_size=0.33)
                except:
                    comb2_x_train = comb2_fv[nrproc][tier][tick]
                    comb2_y_train = MALIGN_LABEL * len(comb2_fv[nrproc][tier][tick])
                    comb2_x_test = list()
                    comb2_y_test = list()
                
                # add training parts to the main training set
                tier2ticks2xtrain[tier][tick]   += single_x_train + comb1_x_train + comb2_x_train
                tier2ticks2ytrain[tier][tick]   += single_y_train + comb1_y_train + comb2_y_train
                
                # save test parts
                nrproc2tiers2ticks2x_single[nrproc][tier][tick]   += single_x_test
                nrproc2tiers2ticks2x_comb[nrproc][tier][tick]     += comb1_x_test + comb2_x_test
                nrproc2tiers2ticks2y_single[nrproc][tier][tick]   += single_y_test
                nrproc2tiers2ticks2y_comb[nrproc][tier][tick]     += comb1_y_test + comb2_y_test

    # Serialize data into files:
    json.dump( tier2ticks2xtrain, open('obj/tier2ticks2xtrain.json', 'w' ) )
    json.dump( tier2ticks2ytrain, open('obj/tier2ticks2ytrain.json', 'w' ) )
    json.dump( nrproc2tiers2ticks2x_single, open('obj/nrproc2tiers2ticks2x_single.json', 'w' ) )
    json.dump( nrproc2tiers2ticks2x_comb, open('obj/nrproc2tiers2ticks2x_comb.json', 'w' ) )
    json.dump( nrproc2tiers2ticks2y_single, open('obj/nrproc2tiers2ticks2y_single.json', 'w' ) )
    json.dump( nrproc2tiers2ticks2y_comb, open('obj/nrproc2tiers2ticks2y_comb.json', 'w' ) )
    
    
    return tier2ticks2xtrain, tier2ticks2ytrain, nrproc2tiers2ticks2x_single, nrproc2tiers2ticks2x_comb, nrproc2tiers2ticks2y_single, nrproc2tiers2ticks2y_comb


def get_partial_trainset_and_parametric_testsets():
    '''
    Si ottiene un training set con tutti i dati fino ad un determinato nrsplit. Poi si creano i testset parametrici rispetto al numero di split.
    nrsplit depends on NR_SPLITS_LIST_REDUCED
    
                | benign    | classic ransom    | single and comb until nrsplit | single and comb after nrsplit | 
    trainset    | 100%      | 100%              | 67%                           | 0%                            |
    testset     | 0%        | 0%                | 33%                           | 100%                          |

    '''
    
    # getting all the 5 datasets
    benign_fv   = get_classic_dataset(BENIGN)
    print("benign loaded")
    rans_fv     = get_classic_dataset(MALIGN)
    print("malign loaded")
    single_fv   = get_single_dataset()    # different structure
    print("single loaded")
    comb1_fv    = get_comb_dataset(1)     # different structure
    print("comb1 loaded")
    comb2_fv    = get_comb_dataset(2)     # different structure
    print("comb2 loaded")
    
    tier2ticks2xtrain = dict()
    tier2ticks2ytrain = dict()
    nrsplit2tier2ticks2xtest  = dict()  # different structure
    nrsplit2tier2ticks2ytest  = dict()  # different structure
    for nrsplit in NR_SPLITS_LIST:
        nrsplit = str(nrsplit)
        nrsplit2tier2ticks2xtest[nrsplit] = dict()
        nrsplit2tier2ticks2ytest[nrsplit] = dict()
        for tier in tierToTicks:
            tier = str(tier)
            tier2ticks2xtrain[tier] = dict()
            tier2ticks2ytrain[tier] = dict()
            nrsplit2tier2ticks2xtest[nrsplit][tier] = dict()
            nrsplit2tier2ticks2ytest[nrsplit][tier] = dict()
            for tick in range(len(tierToTicks[tier])):
                tick = str(tick)
                tier2ticks2xtrain[tier][tick] = list()
                tier2ticks2ytrain[tier][tick] = list()
                nrsplit2tier2ticks2xtest[nrsplit][tier][tick] = list()
                nrsplit2tier2ticks2ytest[nrsplit][tier][tick] = list()
    
    # 
    for tier in tierToTicks:
        tier = str(tier)
        for tick in range(len(tierToTicks[tier])):
            tick = str(tick)
            # add all benign and all ransomware
            tier2ticks2xtrain[tier][tick] += benign_fv[tier][tick] + rans_fv[tier][tick]
            tier2ticks2ytrain[tier][tick] += BENIGN_LABEL * len(benign_fv[tier][tick]) + MALIGN_LABEL * len(rans_fv[tier][tick])
            
            # until nrsplit, 67% of single and combined in the training set and 33% in the testset
            for nrsplit in NR_SPLITS_LIST_REDUCED:
                nrsplit = str(nrsplit)
                # split corresponding single dataset if possible, else trainset empty
                try:
                    single_x_train, single_x_test, single_y_train, single_y_test = train_test_split(single_fv[nrsplit][tier][tick], MALIGN_LABEL * len(single_fv[nrsplit][tier][tick]), test_size=0.33)
                except:
                    single_x_train = single_fv[nrsplit][tier][tick]
                    single_y_train = MALIGN_LABEL * len(single_fv[nrsplit][tier][tick])
                    single_x_test = list()
                    single_y_test = list()
                    
                # split corresponding combined datasets if possible, else trainset empty
                try:
                    comb1_x_train, comb1_x_test, comb1_y_train, comb1_y_test = train_test_split(comb1_fv[nrsplit][tier][tick], MALIGN_LABEL * len(comb1_fv[nrsplit][tier][tick]), test_size=0.33)
                except:
                    comb1_x_train = comb1_fv[nrsplit][tier][tick]
                    comb1_y_train = MALIGN_LABEL * len(comb1_fv[nrsplit][tier][tick])
                    comb1_x_test = list()
                    comb1_y_test = list()
                    
                try:
                    comb2_x_train, comb2_x_test, comb2_y_train, comb2_y_test = train_test_split(comb2_fv[nrsplit][tier][tick], MALIGN_LABEL * len(comb2_fv[nrsplit][tier][tick]), test_size=0.33)
                except:
                    comb2_x_train = comb2_fv[nrsplit][tier][tick]
                    comb2_y_train = MALIGN_LABEL * len(comb2_fv[nrsplit][tier][tick])
                    comb2_x_test = list()
                    comb2_y_test = list()
                
                # add training parts to the main training set
                tier2ticks2xtrain[tier][tick]   += single_x_train + comb1_x_train + comb2_x_train
                tier2ticks2ytrain[tier][tick]   += single_y_train + comb1_y_train + comb2_y_train
                
                # save test parts
                nrsplit2tier2ticks2xtest[nrsplit][tier][tick]   += single_x_test + comb1_x_test + comb2_x_test
                nrsplit2tier2ticks2ytest[nrsplit][tier][tick]   += single_y_test + comb1_y_test + comb2_y_test
                
            # after nrsplit, 100% of single and combined in testset
            for nrsplit in NR_SPLITS_LIST:
                if nrsplit in NR_SPLITS_LIST_REDUCED: continue

                nrsplit = str(nrsplit)
                # save everything to testset
                single_y = MALIGN_LABEL * len(single_fv[nrsplit][tier][tick])
                comb1_y = MALIGN_LABEL * len(comb1_fv[nrsplit][tier][tick])
                comb2_y = MALIGN_LABEL * len(comb2_fv[nrsplit][tier][tick])
                nrsplit2tier2ticks2xtest[nrsplit][tier][tick]   += single_fv[nrsplit][tier][tick] + comb1_fv[nrsplit][tier][tick] + comb2_fv[nrsplit][tier][tick]
                nrsplit2tier2ticks2ytest[nrsplit][tier][tick]   += single_y + comb1_y + comb2_y
    
    return tier2ticks2xtrain, tier2ticks2ytrain, nrsplit2tier2ticks2xtest, nrsplit2tier2ticks2ytest
