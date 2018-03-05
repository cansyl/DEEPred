import os
import os
import re, string; 
import subprocess

def getTerm(stream):
	block = []
	for line in stream:
		if line.strip() == "[Term]" or line.strip() == "[Typedef]":
			break
		else:
			if line.strip() != "":
				block.append(line.strip())
	return block

def parseTagValue(term):
	data = {}
	for line in term:
		tag = line.split(': ',1)[0]
		value = line.split(': ',1)[1]
		if not tag in data:
			data[tag] = []

		data[tag].append(value)
	return data

oboFile = open('go-basic.obo','r')

#keys are the goids
terms = {}

#remove the file header lines
getTerm(oboFile)

while 1:
	term = parseTagValue(getTerm(oboFile))
	if len(term) != 0:
		termID = term['id'][0]

		if 'is_a' in term:
			termParents = [p.split()[0] for p in term['is_a']]
			if not termID in terms:
				terms[termID] = {'p':[],'c':[]}

			terms[termID]['p'] = termParents

			for termParent in termParents:
				if not termParent in terms:
					terms[termParent] = {'p':[],'c':[]}
				terms[termParent]['c'].append(termID)
	else:
		break


def getDescendents(goid):
	recursiveArray = [goid]
	if goid in terms:
		children = terms[goid]['c']
		if len(children) > 0:
			for child in children:
				recursiveArray.extend(getDescendents(child))
	return set(recursiveArray)


def getAncestors(goid):
	recursiveArray = [goid]
	if goid in terms:
		#print("true")
		parents = terms[goid]['p']
		#print(parents)
		if len(parents) > 0:
			for parent in parents:
				#print(parent)
				recursiveArray.extend(getAncestors(parent))
	return set(recursiveArray)


#returns true if the query term is one of the children of the targer term
def isChild(query,target):
	if query in getDescendents(target):
		return True
	return False

#returns true if the query term is the parent of the targer term
def isParent(query, target):
	if query in getAncestors(target):
		return True
	return False


directParents = dict()
def getAllDirectParents():
	go_obo_file = open("go-basic.obo","r")	
	lst_obo_file = go_obo_file.read().split("\n")
	go_obo_file.close()
	go_id = ""
	tempDrctParents = []
	isFirst= True
	for line in lst_obo_file:
		if line.startswith("id: ") and isFirst==True:
			go_id=line[line.find("GO:"):]
			tempDrctParents=[]
			#print(go_id)
			isFirst=False
		elif line.startswith("id: ") and isFirst==False:
			directParents[go_id]=tempDrctParents
			#print("KEY : "+go_id)
			#print(directParents[go_id])
			tempDrctParents=[]
			go_id=line[line.find("GO:"):]
		elif line.startswith("is_a: "):
			sibling = line[line.find("GO:"):line.find("GO:")+10]
			tempDrctParents.append(sibling)

getAllDirectParents()

#for key in directParents.keys():
#	print(key,directParents[key])

def isSibling(targetTerm,otherTerm):
	#Bos mu diye kontrol et
	if directParents[targetTerm]!=[] and directParents[otherTerm]!=[]:
		if len(set(directParents[targetTerm]) & set(directParents[otherTerm]))!=0:
			#print(set(directParents[targetTerm]) & set(directParents[otherTerm]))
			return True
		else:
			return False
	else:
		return False
def getTerm(stream):
	block = []
	for line in stream:
		if line.strip() == "[Term]" or line.strip() == "[Typedef]":
			break
		else:
			if line.strip() != "":
				block.append(line.strip())
	return block

def parseTagValue(term):
	data = {}
	for line in term:
		tag = line.split(': ',1)[0]
		value = line.split(': ',1)[1]
		if not tag in data:
			data[tag] = []

		data[tag].append(value)
	return data

oboFile = open('go-basic.obo','r')

#keys are the goids
terms = {}

#remove the file header lines
getTerm(oboFile)

while 1:
	term = parseTagValue(getTerm(oboFile))
	if len(term) != 0:
		termID = term['id'][0]

		if 'is_a' in term:
			termParents = [p.split()[0] for p in term['is_a']]
			if not termID in terms:
				terms[termID] = {'p':[],'c':[]}

			terms[termID]['p'] = termParents

			for termParent in termParents:
				if not termParent in terms:
					terms[termParent] = {'p':[],'c':[]}
				terms[termParent]['c'].append(termID)
	else:
		break


def getDescendents(goid):
	recursiveArray = [goid]
	if goid in terms:
		children = terms[goid]['c']
		if len(children) > 0:
			for child in children:
				recursiveArray.extend(getDescendents(child))
	return set(recursiveArray)


def getAncestors(goid):
	recursiveArray = [goid]
	if goid in terms:
		#print("true")
		parents = terms[goid]['p']
		#print(parents)
		if len(parents) > 0:
			for parent in parents:
				#print(parent)
				recursiveArray.extend(getAncestors(parent))
	return set(recursiveArray)


#returns true if the query term is one of the children of the targer term
def isChild(query,target):
	if query in getDescendents(target):
		return True
	return False

#returns true if the query term is the parent of the targer term
def isParent(query, target):
	if query in getAncestors(target):
		return True
	return False


directParents = dict()
def getAllDirectParents():
	go_obo_file = open("go-basic.obo","r")	
	lst_obo_file = go_obo_file.read().split("\n")
	go_obo_file.close()
	go_id = ""
	tempDrctParents = []
	isFirst= True
	for line in lst_obo_file:
		if line.startswith("id: ") and isFirst==True:
			go_id=line[line.find("GO:"):]
			tempDrctParents=[]
			#print(go_id)
			isFirst=False
		elif line.startswith("id: ") and isFirst==False:
			directParents[go_id]=tempDrctParents
			#print("KEY : "+go_id)
			#print(directParents[go_id])
			tempDrctParents=[]
			go_id=line[line.find("GO:"):]
		elif line.startswith("is_a: "):
			sibling = line[line.find("GO:"):line.find("GO:")+10]
			tempDrctParents.append(sibling)

getAllDirectParents()

#for key in directParents.keys():
#	print(key,directParents[key])

def isSibling(targetTerm,otherTerm):
	#Bos mu diye kontrol et
	if directParents[targetTerm]!=[] and directParents[otherTerm]!=[]:
		if len(set(directParents[targetTerm]) & set(directParents[otherTerm]))!=0:
			#print(set(directParents[targetTerm]) & set(directParents[otherTerm]))
			return True
		else:
			return False
	else:
		return False
"""
def propagateAnnotsParents(inFile):
	lst_inFile = open(inFile,"r").readlines()
	lst_inFile.close()
	prot_dict = dict()
	for line in lst_inFile:
		prot_id,goid,aspect = line.split("\n")[0]a.plit("\t")
		try:
			prot_dict[prot_id].add(goid)
		except:
			prot_dict[prot_id] = set()
			prot_dict[prot_id].add(goid)
	print(len(prot_dict.keys()))
propagateAnnotsParents("associations.tsv")
"""

"""
def getPredFile(cafaid,category,goannot):
	path ="/Users/trman/Dropbox/UniGOPred_MSB/CAFA/CAFA2/Mydocs_CAFA2/BenchMarkPreds_"+category
	for fl in os.listdir(path):
		if fl.startswith(cafaid):
			pred_file = open(path+"/"+fl)
			pred_file.readline()
			lst_pred_file = pred_file.read().split("\n")
			pred_file.close()
			lst_pred_file.remove("")
			for line in lst_pred_file:
				gopred,wmean,spmap,blast,pepstats = line.split("\t")
				if gopred==goannot:
					print(cafaid+"\t"+gopred+"\t"+wmean)

def createMostSpecificPredFile(category):
	path ="/Users/trman/Dropbox/UniGOPred_MSB/CAFA/CAFA2/Mydocs_CAFA2/BenchMarkPreds_"+category
	specific_path = "/Users/trman/Dropbox/UniGOPred_MSB/CAFA/CAFA2/Mydocs_CAFA2/Specific_BenchMarkPreds_"+category
	trained_pred_file = open("BenchMarkPreds_"+category+"/T96060001309_ATD2B_HUMAN.preds")
	header = trained_pred_file.readline()
	lst_trained_pred_file = trained_pred_file.read().split("\n")
	trained_pred_file.close()
	
	trained_terms = []
	for line in lst_trained_pred_file:
		go = line.split("\t")[0]
		trained_terms.append(go)

	most_specific_terms = []
	for got in trained_terms:
		isSpecific = True
		for got2 in trained_terms:
			if got!=got2 and isParent(got,got2):
				isSpecific=False
				break
		if isSpecific:
			most_specific_terms.append(got)

	for fl in os.listdir(path):
		#if fl.startswith(cafaid):
		new_pred = open(specific_path+"/"+fl,"w")
		new_pred.write(header)

		pred_file = open(path+"/"+fl,"r")
		pred_file.readline()
		lst_pred_file = pred_file.read().split("\n")
		pred_file.close()
		lst_pred_file.remove("")
		
		for line in lst_pred_file:
			gopred,wmean,spmap,blast,pepstats = line.split("\t")
			if gopred in most_specific_terms:
				new_pred.write(line+"\n")
		new_pred.close()

createMostSpecificPredFile("BP")
mf_pred_file = open("BenchMarkPreds_MF/T96060001309_ATD2B_HUMAN.preds")
mf_pred_file.readline()
lst_mf_pred_file = mf_pred_file.read().split("\n")
mf_pred_file.close()

cc_pred_file = open("BenchMarkPreds_CC/T96060001309_ATD2B_HUMAN.preds")
cc_pred_file.readline()
lst_cc_pred_file = cc_pred_file.read().split("\n")
cc_pred_file.close()

bp_pred_file = open("BenchMarkPreds_BP/T96060001309_ATD2B_HUMAN.preds")
bp_pred_file.readline()
lst_bp_pred_file = bp_pred_file.read().split("\n")
bp_pred_file.close()


BPO_annots_file = open("./../CAFA2-master/benchmark/groundtruth/propagated_BPO.txt","r")
lst_BPO_annots_file  = BPO_annots_file.read().split("\n")
BPO_annots_file.close()

CCO_annots_file = open("./../CAFA2-master/benchmark/groundtruth/propagated_CCO.txt","r")
lst_CCO_annots_file = CCO_annots_file.read().split("\n")
CCO_annots_file.close()

MFO_annots_file = open("./../CAFA2-master/benchmark/groundtruth/propagated_MFO.txt","r")
lst_MFO_annots_file = MFO_annots_file.read().split("\n")
MFO_annots_file.close()

lst_MFO_annots_file.remove("")
lst_BPO_annots_file.remove("")
lst_CCO_annots_file.remove("")
lst_mf_pred_file.remove("")
lst_bp_pred_file.remove("")
lst_cc_pred_file.remove("")

bp_terms = []
for line in lst_bp_pred_file:
	go = line.split("\t")[0]
	bp_terms.append(go)

cc_terms = []
for line in lst_cc_pred_file:
	go = line.split("\t")[0]
	cc_terms.append(go)

mf_terms = []
for line in lst_mf_pred_file:
	go = line.split("\t")[0]
	mf_terms.append(go)

def getAnnots(category):
	if category=="MF":
		for annot in lst_MFO_annots_file:
			cafaid,go = annot.split("\t")
			if go in mf_terms:
				getPredFile(cafaid,"MF",go)
	elif category=="BP":
		for annot in lst_BPO_annots_file:
			cafaid,go = annot.split("\t")
			if go in bp_terms:
				getPredFile(cafaid,"BP",go)

	elif category=="CC":
		for annot in lst_CCO_annots_file:
			cafaid,go = annot.split("\t")
			if go in cc_terms:
				print(annot)	
				getPredFile(cafaid,"CC",go)

#getAnnots("BP")





"""
