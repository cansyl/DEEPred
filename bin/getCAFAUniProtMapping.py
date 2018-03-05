def getFasta(fastaFilePath):
    # fasta dict is a dcitionary keys are prot ids values are corresponding seqs
    fasta_dict = dict()

    prot_id = ""
    with open(fastaFilePath) as f:
        for line in f:
            line = line.split("\n")[0]
            if line.startswith(">"):
                prot_id = line.split("|")[1]
                fasta_dict[prot_id] = ""
            else:
                fasta_dict[prot_id] = fasta_dict[prot_id] + line

                # print(prot_id)
    # print(fasta_dict["P37642"])
    return fasta_dict


cafa_fasta_dict = getFasta("../Annots/cco_cafa_only_annot_prot.fasta")
uniprot_fasta_dict = getFasta("../Annots/all_over_30_sp_manual_experimental_prot_seqs.fasta")

cafa_uniprot_id_set = set()
for cprot in cafa_fasta_dict.keys():
    for uprot in uniprot_fasta_dict.keys():
        if cafa_fasta_dict[cprot]==uniprot_fasta_dict[uprot]:
            cafa_uniprot_id_set.add(cprot+"\t"+uprot)
            break

for pair in cafa_uniprot_id_set:
    print(pair)

