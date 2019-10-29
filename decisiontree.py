from sys import argv
import math



dataList = []
fields = {}
fieldsList = []


def __main__():

	if len(argv) != 2:
		print("incorrect number of arguments given")
		exit()

	global fieldsList, dataList, fields

	with open(argv[1],'r') as data:
		fieldsList = data.readline().split()[:-1]
		
		for field in fieldsList:
			fields[field] = set()
		line = data.readline()
		
		while line:
			splitLine = line.split()
			for i,element in enumerate(splitLine[:-1]):
				currentField = fieldsList[i]
				fields[currentField].add(element)

			dataList.append(splitLine)
			line = data.readline()
	tree = makeTree(dataList, fieldsList.copy())
	
	print()
	printTree(tree)
	print()

def entropy(field, data):
	'''returns the entropy of choosing a given field to split on,
	given all the data passing through the parent'''

	fieldIndex = fieldsList.index(field)
	countsDict = {}
	for key in fields[field]:
		countsDict[key] = [0,0]
	for line in data:
		if line[-1] == "yes":
			countsDict[line[fieldIndex]][0] += 1
		else:
			countsDict[line[fieldIndex]][1] += 1
	
	entropy = 0
	for fieldState in countsDict.keys():
		yesses = countsDict[fieldState][0]
		nos = countsDict[fieldState][1]
		if not (yesses and nos):
			entropy += yesses + nos
			continue
		yesEntropy = yesses / (yesses + nos)
		noEntropy = nos / (yesses + nos)
		yesEntropy *= math.log(1.0/yesEntropy, 2)
		noEntropy *= math.log(1.0/noEntropy, 2)
		entropy += (yesEntropy + noEntropy) * (yesses + nos)

	return entropy

def makeTree(data, fieldsAvailable):

	# base case 1: no more attributes
	
	yesses = 0
	nos = 0
	for line in data:
		if line[-1] == "yes":
			yesses += 1
		else:
			nos += 1
	if not fieldsAvailable:
		if yesses >= nos:
			return "yes"
		return "no"

	# base case 2: all points are either yes or no
	for line in data:
		if line[-1] != data[0][-1]:
			break
	else:
		return data[0][-1] # returns a yes or no

	bestField = (fieldsAvailable[0],entropy(fieldsAvailable[0],data))
	for field in fieldsAvailable[1:]:
		curEntropy = entropy(field,data)
		if curEntropy < bestField[1]:
			bestField = (field, curEntropy)

	
	tree = {}
	bestField = bestField[0]
	fieldIndex = fieldsList.index(bestField)
	fieldsAvailable.remove(bestField)
	for fieldState in fields[bestField]:
		filteredData = list(filter(lambda x : x[fieldIndex] == fieldState, data.copy()))
		if not filteredData:
			if yesses >= nos:
				tree[bestField + " = " + fieldState] = 'yes'
			else:
				tree[bestField + " = " + fieldState] = 'no'
			continue

		tree[bestField + " = " + fieldState] = makeTree(filteredData, fieldsAvailable.copy())

	return tree

def printTree(tree, depth = 0):
	
	if type(tree) == str:
		print(': ' + tree)
		return
	else:
		print()

	for key in tree.keys():
		print("|\t"*depth + key, end = '')
		printTree(tree[key], depth + 1)




if __name__ == '__main__':
	__main__()