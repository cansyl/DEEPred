import sys
feature_type = sys.argv[1]

r_features = open("../FeatureVectors/%sFeatures_uniprot_training_test_set.txt" %(feature_type),"r")
lst_r_features = r_features.read().split("[1] ")
r_features.close()
if "" in lst_r_features:
    lst_r_features.remove("")

for field in lst_r_features:
    lst_field = field.split("\n")
    if "" in lst_field:
        lst_field.remove("")
    prot_id = lst_field[0].split("|")[1]
    features = []
    count = 0
    for line in lst_field[1:]:
        if count%2==1:
            features.extend(line.split())
        count+=1
    #print(len(features))
    str_prot_feat_string = prot_id+"\t"
    for feat in features:
        str_prot_feat_string += feat+"\t"
    print(str_prot_feat_string)