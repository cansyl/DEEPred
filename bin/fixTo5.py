import os
import sys

category = sys.argv[1]
path = "../GOTermFiles/%s" %(category)
subgroup_path = "%s/subgroups" %(path)
five_path = "%s/5" %(path)
new_fl = None
isFirst = True
for fl in os.listdir(subgroup_path):
	print(fl)
	if fl.startswith("%sGOTerms" %(category)):
		fl_count=1
		go_fl = open("%s/%s" %(subgroup_path,fl))
		a = go_fl.read().split("\n")
		go_fl.close()
		if "" in a:
			a.remove("")

		len_a = len(a)

		for i in range(len_a):
			if i%5==0 and len(a)<10:
				if isFirst:
					isFirst = False
					new_fl = open("%s/%s_%d.txt" %(five_path, fl.split(".txt")[0],fl_count), "w")
					while a:
						new_fl.write(a.pop()+"\n")
						#print(new_fl,a.pop())
						fl_count+=1
					break
				else:
					new_fl.close()
					new_fl = open("%s/%s_%d.txt" %(five_path, fl.split(".txt")[0],fl_count), "w")
					while a:
						new_fl.write(a.pop()+"\n")
						#print(new_fl,a.pop())
						fl_count+=1
					break

			elif i%5==0:
				if isFirst:
					isFirst = False
					new_fl = open("%s/%s_%d.txt" %(five_path, fl.split(".txt")[0],fl_count),"w")
					fl_count+=1
					#print(new_fl,a.pop())
					new_fl.write(a.pop()+"\n")
				else:
					new_fl.close()
					new_fl = open("%s/%s_%d.txt" %(five_path, fl.split(".txt")[0],fl_count),"w")
					fl_count+=1
					#print(new_fl,a.pop())
					new_fl.write(a.pop()+"\n")

			else:
				#print(new_fl,a.pop())
				new_fl.write(a.pop()+"\n")
		
