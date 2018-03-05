import sys
from os import listdir

category = sys.argv[1]
path = "../TrainTestDatasets/%s" %(category)

all_train_ids = []
for fl in listdir(path):
    if fl.startswith("train"):
        train_fl = open("%s/%s" %(path, fl), "r")
        lst_train_fl = train_fl.read().split("\n")
        train_fl.close()
        if "" in lst_train_fl:
            lst_train_fl.remove("")
        all_train_ids.extend(lst_train_fl)

all_train_ids = set(all_train_ids)

for id in all_train_ids:
    print(id)