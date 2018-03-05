from random import shuffle


def getCAFAUniProtMapping(mappingFilePath):
    mapping_file = open(mappingFilePath,"r")
    lst_mapping_file = mapping_file.read().split("\n")
    mapping_file.close()

    if "" in lst_mapping_file:
        lst_mapping_file.remove("")

    uniprot_ids_set = set()

    for line in lst_mapping_file:
        cafaid, uid = line.split("\t")
        uniprot_ids_set.add(uid)
    return uniprot_ids_set
    #lst_uniprot_ids = list(uniprot_ids_set)
    #return lst_uniprot_ids

def getSequencesLessThan31Length():
    small_sequence_ids = set()
    all_swissprot_fasta_fl = open("../FastaFiles/swissprot_all_29_08_2017.fasta", "r")
    lst_sp_seqs = all_swissprot_fasta_fl.read().split("\n")
    all_swissprot_fasta_fl.close()

    if "" in lst_sp_seqs:
        lst_sp_seqs.remove("")

    sequence = ""
    header = ""
    isFirst = True
    count = 0
    for line in lst_sp_seqs:
        if isFirst:
            header = line
            isFirst = False
        if line.startswith(">"):
            if len(sequence) < 31:
                id = header.split("|")[1]
                #print(id)
                small_sequence_ids.add(id)
                count += 1
                #print(header, sequence)
            sequence = ""
            header = line
        else:
            sequence += line
    return small_sequence_ids
    #print(count)

small_sequence_id_set = getSequencesLessThan31Length()

mf_cafa_uniprot_id_set = getCAFAUniProtMapping("../Annots/MF_cafa_uniprot_id_mapping.txt")
bp_cafa_uniprot_id_set = getCAFAUniProtMapping("../Annots/BP_cafa_uniprot_id_mapping.txt")
cc_cafa_uniprot_id_set = getCAFAUniProtMapping("../Annots/CC_cafa_uniprot_id_mapping.txt")


path = "../TrainTestDatasets"
all_annot_fl = open("../Annots/all_categories_manual_experimental_annots_29_08_2017_Propagated.tsv","r")
lst_all_annots = all_annot_fl.read().split("\n")
all_annot_fl.close()

if "" in lst_all_annots:
    lst_all_annots.remove("")

go_prot_dict  = dict()

for line in lst_all_annots[1:]:
    prot_id, go_id, category = line.split("\t")
    try:
        go_prot_dict[go_id][0].add(prot_id)
    except:
        go_prot_dict[go_id] = [set(), category]
        go_prot_dict[go_id][0].add(prot_id)

for go in go_prot_dict.keys():
    go_prot_dict[go][0] = go_prot_dict[go][0] - small_sequence_id_set

for go in go_prot_dict.keys():
    if len(go_prot_dict[go][0])>=30:
        prot_annot_size = len(go_prot_dict[go][0])
        category = go_prot_dict[go][1]
        #print(category)
        test_size = int(len(go_prot_dict[go][0]) * (10 / 100))
        lst_go_annots = list(go_prot_dict[go][0])
        shuffle(lst_go_annots)
        set_go_annots = set(lst_go_annots)

        cafa_training_intersect = None
        if category=="Function":
            category= "MF"
            cafa_training_intersect = set_go_annots & mf_cafa_uniprot_id_set
            set_go_annots = set_go_annots - cafa_training_intersect
        elif category=="Process":
            category = "BP"
            cafa_training_intersect = set_go_annots & bp_cafa_uniprot_id_set
            set_go_annots = set_go_annots - cafa_training_intersect
        elif category=="Component":
            category = "CC"
            cafa_training_intersect = set_go_annots & cc_cafa_uniprot_id_set
            set_go_annots = set_go_annots - cafa_training_intersect
        else:
            print("Category Problem")
            pass
        #if len(cafa_training_intersect)!=0:
        #    print(len(cafa_training_intersect))

        lst_test_ids = []
        lst_train_ids = []

        for prot_id in cafa_training_intersect:
            lst_test_ids.append(prot_id)

        while len(lst_test_ids) < test_size:
            lst_test_ids.append(set_go_annots.pop())

        lst_train_ids = list(set_go_annots)

        tst_fl = open("%s/%s/test_%s.ids" %(path, category, go), "w")
        for prot in lst_test_ids:
            tst_fl.write(prot+"\n")
        tst_fl.close()

        train_fl = open("%s/%s/train_%s.ids" %(path, category, go), "w")
        for prot in lst_train_ids:
            train_fl.write(prot+"\n")
        train_fl.close()

        #GO:0045776_positive.ids
        print(go, len(lst_train_ids), test_size, prot_annot_size)
