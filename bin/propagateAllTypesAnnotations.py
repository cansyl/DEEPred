import go_dag_parser as dag
import sys
#biological_process
#molecular_function
#cellular_component
#id: GO:0000001
#name: mitochondrion inheritance
#namespace: biological_process

def getListofGOsCategory(short_category,long_category):

    all_go_terms_fl = open('./go-basic.obo', 'r')
    lst_all_go_terms_fl = all_go_terms_fl.read().split("\n")
    all_go_terms_fl.close()
    go_term_dict=dict()
    go_term=""
    category=""
    for line in lst_all_go_terms_fl:
        if line.startswith("id: "):
            go_term=line.split(" ")[1]
        if line =="namespace: %s" %(long_category):
            go_term_dict[go_term]=""
    return go_term_dict

mf_go_term_dict = getListofGOsCategory("mf","molecular_function")
bp_go_term_dict = getListofGOsCategory("bp","biological_process")
cc_go_term_dict = getListofGOsCategory("cc","cellular_component")

#category_list = ["Function","Process","Component"]

annot_fl_name = "../all_categories_manual_experimental_annots_29_08_2017_NOT_Propagated.tsv"
annot_fl = open(annot_fl_name,"r")
lst_annot_fl = annot_fl.read().split("\n")
annot_fl.close()

if "" in lst_annot_fl:
	lst_annot_fl.remove("")

propagated_all_categories_annot_set = set()
#for c in category_list:
#	propagated_category_annot_dict[c] = set()

go_ancestor_dict = dict()
#ID GO ID   Aspect  Evidence
for line in lst_annot_fl[1:]:
    #ID	Symbol	GO ID	GO Name	Aspect
    prot_id, symbol, go, go_name, category = line.split("\t")
    propagated_all_categories_annot_set.add("%s\t%s\t%s" %(prot_id,go,category))
    try:
            go_ancestor_dict[go]

    except:
            go_ancestor_dict[go] = dag.getAncestors(go)
    for ancestor in go_ancestor_dict[go]:
        category="UNKNOWN"
        #print(ancestor)
        try:
            mf_go_term_dict[ancestor]
            category = "Function"
        except:
            pass
        try:
            bp_go_term_dict[ancestor]
            category = "Process"
        except:
            pass
        try:
            cc_go_term_dict[ancestor]
            category = "Component"
        except:
            pass
        if category=="UNKNOWN":
            print(ancestor,"problem")
        else:
            propagated_all_categories_annot_set.add("%s\t%s\t%s" %(prot_id, ancestor, category))

print("Prot ID\tGO ID\tCategory")

for annot in propagated_all_categories_annot_set:
    print(annot)
