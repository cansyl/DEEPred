import os
import sys
#path = "/Users/trman/Dropbox/Academic/METU/PhD/PhDProject/VirEnvDrugProject/%sDeepTrainFilesSinan" %(category)
#path = "/Users/trman/Desktop/NEWMFPRED/DEEPPRED/deepData/mf/spmap"
mapping_files_path ="../FeatureVectors"
#cafa3_vec_path = "/Users/trman/Desktop/NEWMFPRED/DEEPPRED/deepData/mf/spmap"
#target.10090.fasta_cafa3_id_number_mapping.tsv
vec_path = "../FeatureVectors"
#train.positive1.10090.fa.vec
"""
for fl in os.listdir(mapping_files_path):
    print(fl)
    taxon = fl.split(".")[1]
    taxon_specific_vec_file = open("bp_cafa3_%s.vecs" %(taxon),"w")
    mapping_fl = open("%s/%s" %(mapping_files_path,fl),"r")
    lst_mapping_fl = mapping_fl.read().split("\n")
    mapping_fl.close()
    if "" in lst_mapping_fl:
        lst_mapping_fl.remove("")
    mapping_dict = dict()
    
    for line in lst_mapping_fl:
        acc,num = line.split("\t")
        acc = acc.split(" ")[0]
        print(acc,num)
        mapping_dict[num] = acc
    
    count = 0
    for key in mapping_dict.keys():
        if count%1000==0:
            sys.stderr.write(str(count)+"\n")
        count+=1
        vec_file = open("%s/train.positive%s.%s.fa.vec" %(cafa3_vec_path,key,taxon),"r")
        str_vec_file = vec_file.read().split("\n")[0]
        vec_file.close()
        str_vec_file =str_vec_file.split(" ")
        #print(len(str_vec_file))
        if len(str_vec_file)==1862:
            res_str = mapping_dict[key] +"\t"
            for dim in str_vec_file[1:]:
                #print(dim)
                res_str += dim.split(":")[1]+"\t"
            #print(str_vec_file)
            taxon_specific_vec_file.write(res_str+"\n")
    taxon_specific_vec_file.close()
"""
category = sys.argv[1]
mapping_file =open("%s/%s_uniprot_number_mapping.tsv" %(mapping_files_path,category),"r")
lst_mapping_file = mapping_file.read().split("\n")
mapping_file.close()
lst_mapping_file.remove("")
mapping_dict = dict()
for line in lst_mapping_file:
    acc,num = line.split("\t")
    mapping_dict[num] = acc

count = 0
for key in mapping_dict.keys():
    
    if count%1000==0:
        sys.stderr.write(str(count))
    count+=1
    vec_file = open("%s/train.positive%s.fa.vec" %(vec_path,key),"r")
    str_vec_file = vec_file.read().split("\n")[0]
    vec_file.close()
    str_vec_file =str_vec_file.split(" ")
    res_str = mapping_dict[key] +"\t"
    for dim in str_vec_file[1:]:
        #print(dim)
        res_str += dim.split(":")[1]+"\t"
    #print(str_vec_file)
    print(res_str)
