fasta_file = open("../FastaFiles/MF_deepred_training_sequences.fasta","r")
lst_fasta_fl = fasta_file.read().split("\n")
fasta_file.close()

lst_count_id = []
count=1
for line in lst_fasta_fl:
    if line.startswith(">"):
        u_id = line.split("|")[1]
        print("%s\t%d" %(u_id,count))
        count += 1