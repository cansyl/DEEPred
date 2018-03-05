"""
def getListofGOsCategory(long_category):
    all_go_terms_fl = open("go-basic.obo","r")
    lst_all_go_terms_fl = all_go_terms_fl.read().split("\n")
    all_go_terms_fl.close()
    go_term_dict=dict()
    go_term=""
    for line in lst_all_go_terms_fl:
        if line.startswith("id: "):
            go_term=line.split(" ")[1]
        if line =="namespace: %s" %(long_category):
            go_term_dict[go_term]=""
    return go_term_dict

mf_go_term_dict = getListofGOsCategory("molecular_function")
bp_go_term_dict = getListofGOsCategory("biological_process")
cc_go_term_dict = getListofGOsCategory("cellular_component")
"""
import os
import sys
from operator import itemgetter

category = sys.argv[1]
path = "../TrainTestDatasets/%s" %(category)
lst_GO_terms = []
category = sys.argv[1]
for fl in os.listdir(path):
    if fl.startswith("train"):
        go_term = fl.split("_")[1][:-4]
        train_fl = open(path+"/"+fl,"r")
        lst_train_fl = train_fl.read().split("\n")
        train_fl.close()

        if "" in lst_train_fl:
            lst_train_fl.remove("")

        lst_GO_terms.append([go_term,len(lst_train_fl)])

sorted_list = sorted(lst_GO_terms, key=itemgetter(1), reverse=True)

for line in sorted_list:
    if line[1]>29:
        print(line[0]+"\t"+str(line[1]))