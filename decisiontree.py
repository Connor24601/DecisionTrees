from sys import argv
import math
import scipy.stats



# Stores data in list of lists format
dataList = []
# Stores dict of fields (unordered) and their possible attributes
fields = {}
# Stores list of fields (in order)
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
    print('Unpruned tree:')
    printTree(tree)
    print()

    print("Unpruned training set accuracy: ")
    numCorrect = 0
    total = 0
    for line in dataList:
        numCorrect += testValue(tree,line)
        total += 1
    print("We got {0} entries correct and {1} entries incorrect ({2:.2f}% accuracy).".format(numCorrect, total-numCorrect, numCorrect*100/total))

    pruned = pruneTree(tree)[0]
    print()
    print('Pruned tree:')
    printTree(pruned)
    print()

    numCorrect = 0
    total = 0
    print("Pruned training set accuracy: ")
    for line in dataList:
        numCorrect += testValue(pruned,line)
        total += 1
    print("We got {0} entries correct and {1} entries incorrect ({2:.2f}% accuracy).".format(numCorrect, total-numCorrect, numCorrect*100/total))

    numCorrectPruned = 0
    numCorrect = 0
    total = 0
    # Leave-one-out cross-validation
    for line in dataList:
        # Get the dataset with the current data point removed
        dataCopy = dataList.copy()
        dataCopy.remove(line)
        # Train the tree on the reduced dataset
        tree = makeTree(dataCopy, fieldsList.copy())
        # Test the tree on the left-out data point
        numCorrect += testValue(tree,line)
        prunedTree = pruneTree(tree)[0]
        numCorrectPruned += testValue(prunedTree,line)
        total += 1
    print("Unpruned leave-one-out cross-validation:")
    print("We got {0} entries correct and {1} entries incorrect ({2:.2f}% accuracy).".format(numCorrect, total-numCorrect, numCorrect*100/total))
    print("Pruned leave-one-out cross-validation:")
    print("We got {0} entries correct and {1} entries incorrect ({2:.2f}% accuracy).".format(numCorrectPruned, total-numCorrectPruned, numCorrectPruned*100/total))
    print()
    print("Pruning helped with cross-validation on the tennis dataset.")
    print("Pruning didn't help with cross-validation on the titanic dataset.")




def entropy(field, data):
    '''returns the entropy of choosing a given field to split on,
    given all the data passing through the parent'''
    # Count yesses and nos for each split
    fieldIndex = fieldsList.index(field)
    countsDict = {} # Stores yesses and nos for each branch of the split
    for fieldState in fields[field]:
        countsDict[fieldState] = [0,0]
    for line in data:
        # For each data point, increment yesses or nos for its split
        if line[-1] == "yes":
            countsDict[line[fieldIndex]][0] += 1
        else:
            countsDict[line[fieldIndex]][1] += 1
    # Calculate total entropy = weighted average of branch entropies
    entropy = 0
    for fieldState in fields[field]:
        # Calculate branch entropy = weighted average of yes/no information
        yesses = countsDict[fieldState][0]
        nos = countsDict[fieldState][1]
        if not (yesses and nos):
            entropy += yesses + nos
            continue
        yesInformation = yesses / (yesses + nos) # p of yes
        noInformation = nos / (yesses + nos) # p of no
        yesInformation *= math.log(1.0/yesInformation, 2) # info. of yes
        noInformation *= math.log(1.0/noInformation, 2) # info. of no
        entropy += (yesInformation + noInformation) * (yesses + nos)

    return entropy

def makeTree(data, fieldsAvailable):
    ''' Creates a decision tree using the ID3 algorithm; each non-terminal
    node is a dict of entries {'field = attribute': child}
    each terminal node is a tuple of a decision ('yes' or 'no') and
    total yes/no counts'''
    # Count total yesses and nos, since we'll need this later
    yesses = 0
    nos = 0
    for line in data:
        if line[-1] == "yes":
            yesses += 1
        else:
            nos += 1
    # Base case 1: no more attributes; choose based on total yesses vs nos
    if not fieldsAvailable:
        if yesses >= nos:
            return ('yes', (yesses, nos))
        else:
            return ('no', (yesses, nos))

    # Base case 2: data is homogeneous: all points are either yes or no
    # If data is not homogeneous, this loop will break, thus not triggering
    # the else statement
    for line in data:
        if line[-1] != data[0][-1]:
            break
    else:
        # Data is homogeneous
        return (data[0][-1], (yesses, nos))

    # Find split with lowest entropy
    bestField = (fieldsAvailable[0],entropy(fieldsAvailable[0],data))
    for field in fieldsAvailable[1:]:
        # Get entropy for splitting on this field
        splitEntropy = entropy(field,data)
        if splitEntropy < bestField[1]:
            bestField = (field, splitEntropy)


    tree = {}
    # Don't need to know entropy anymore
    bestField = bestField[0]
    fieldIndex = fieldsList.index(bestField)
    fieldsAvailable.remove(bestField)
    # Make children for the chosen split
    for fieldState in fields[bestField]:
        # Figure out which data to put in the given child
        filteredData = list(filter(lambda x : x[fieldIndex] == fieldState, data.copy()))
        # If the child doesn't get any data, set its decision = parent's
        if not filteredData:
            if yesses >= nos:
                tree[bestField + " = " + fieldState] = ('yes', (0, 0))
            else:
                tree[bestField + " = " + fieldState] = ('no', (0, 0))
            continue
        # Else, the child is the result of continuing to make a tree with 
        # its data
        tree[bestField + " = " + fieldState] = makeTree(filteredData, fieldsAvailable.copy())

    return tree

def printTree(tree, depth = 0):
    ''' Prints a tree, not including the exact yes/no counts at its leaves'''
    if type(tree) == tuple:
        if depth != 0:
            print(':', end=' ')
        print(tree[0])
        return
    else:
        print()

    for key in tree.keys():
        print("|\t"*depth + key, end = '')
        printTree(tree[key], depth + 1)

def pruneTree(tree):
    ''' Prunes a tree using chi-squared significance testing of its splits.'''
    # Base case: can't prune a leaf
    if type(tree) == tuple:
        return tree, tree[1]
    # Tells us what our significance threshhold is, assuming 95% confidence
    chiThreshhold = scipy.stats.chi2.ppf(0.95, len(tree.keys())-1)
    childTuples = [] # Holds yes/no counts for children
    yesses, nos = 0, 0 # Total yes/no count for (children of) current node
    # Prune children, then get their yes/no counts
    for key in tree.keys():
        tree[key], childTuple = pruneTree(tree[key])
        childTuples.append(childTuple)
        yesses += childTuple[0]
        nos += childTuple[1]
    # Expected probability of a yes, given random distribution of parent data
    yesProb = yesses/(yesses+nos)
    chi = 0 # Total significance of splits
    for childTuple in childTuples:
        expYes = yesProb*(childTuple[0]+childTuple[1])
        expNo = (1-yesProb)*(childTuple[0]+childTuple[1])
        # If child has no data, say that it's not significantly different
        if expYes == 0:
            yesSig = 0
            noSig = 0
        else:
            yesSig = ((childTuple[0] - expYes)**2)/expYes
            noSig = ((childTuple[1] - expNo)**2)/expNo
        chi += yesSig + noSig
    # If significance threshhold is not reached, prune
    if chi < chiThreshhold:
        if yesses >= nos:
            tree = ('yes', (yesses, nos))
        else:
            tree = ('no', (yesses, nos))
    return tree, (yesses, nos)

def testValue(tree, dataLine):
    ''' Determines whether or not the tree classifies a point correctly.'''
    # Base case: at a leaf, just check leaf label vs data label
    if type(tree) == tuple:
        return tree[0] == dataLine[-1]
    # Otherwise, figure out which branch to take
    for key in tree.keys():
        field, fieldState = key.split(" = ")
        fieldIndex = fieldsList.index(field)
        if dataLine[fieldIndex] == fieldState:
            return testValue(tree[key],dataLine)


if __name__ == '__main__':
    __main__()
