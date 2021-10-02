
class DecisionTreeClassifier:
    def __init__(self, max_depth: int, data: list[list[float]]):
        self.data = data
        self.max_depth = max_depth
        self.trueList = []
        self.falseList = []
        self.testPredict = []
        self.trainPredict = []
        self.yhat = []
        self.trainData = []
        combined, X, yhat, test_data, complete,self.trainData ,test= self.split_data()
        self.yhat = yhat
        # I used combined data bec I couldnt find a way that works with separated data
        # When I split the data and tree X and Y gone mad :((
        
        #rootNode = self.fit(combined, self.trainData)
      
        #rootNode = self.fit(combined, self.trainData)
        #self.testPredict = self.predict(test_data, rootNode, "test")
        #self.trainPredict = self.predict(combined, rootNode, "train")
    
    def fit(self, X: list[list[float]], y: list[int]):
        combine = self.combineData(X,y)
        decisionTree = self.constructTree(combine, 0)
        self.printTree(decisionTree, "")
        return decisionTree ,combine

    def predict(self, X: list[list[float]], root ):
        correct = 0
        predictedArray = []
        for item in X:
            isCorrect, label = self.isDataLabeledCorrect(item, root)
            predictedArray.append(label)
            if isCorrect:
                correct += 1
        print("Correct Labels: " + str(correct) +
                " Accuracy is: " + str((100*correct)/len(X)))
        return predictedArray
    def combineData(self,X: list[list[float]], y: list[int]):
        combined = []
        for i in range(len(X)):
            X[i].append(y[i])
        print(len(X))
        return X
    def isDataLabeledCorrect(self, array: list[float], node):
        # print(node)
        if isinstance(node, LeafNode):
            if(node.type == array[4]):
                return True, node.type
            else:
                return False, node.type
        elif array[node.questionAttNumber] <= node.questionNumber:
            return self.isDataLabeledCorrect(array, node.trueBranch)
        else:
            return self.isDataLabeledCorrect(array, node.falseBranch)
    #You just said split the data in notebook but I already did this in my code. Its not part of the model
    #But :/
    def split_data(self):
        data_train, data_test, Y_test, X_test, complete_data,test = [], [], [], [], [],[]
        self.Y, self.X, self.Y_train, self.X_train = [], [], [], []
        train = []
        # Train the DT Classifier using the first %80 of the data and test it with the remaining data.
        trainLimitInt = int((len(self.data) / 100) * 80)
        for i in range(150):
            if self.data[i][4] == 'Iris-setosa':
                complete_data.append(
                    [self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3],0])
                test.append(0)
            elif self.data[i][4] == 'Iris-versicolor':
                complete_data.append(
                    [self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3],1])
                test.append(1)
            elif self.data[i][4] == 'Iris-virginica':
                complete_data.append(
                    [self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3],2])
                test.append(2)

        for i in range(trainLimitInt):
            if self.data[i][4] == 'Iris-setosa':
                data_train.append(
                    [self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3]])
                train.append(0)
            elif self.data[i][4] == 'Iris-versicolor':
                train.append(1)
                data_train.append(
                    [self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3]])
            elif self.data[i][4] == 'Iris-virginica':
                data_train.append(
                    [self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3]])
                train.append(2)
            self.X.append([self.data[i][0], self.data[i][1],
                          self.data[i][2], self.data[i][3]])
        for i in range(trainLimitInt, len(self.data)):
            data_test.append(complete_data[i])
            Y_test.append(complete_data[i][4])
            X_test.append([complete_data[i][0], complete_data[i]
                          [1], complete_data[i][2], complete_data[i][3]])
        return data_train, X_test, Y_test, data_test, complete_data,train,test

    def splitTree(self, partArr: list[list[float]], question):
        trueList, falseList = [], []
        for item in partArr:
            if item[question.att] <= question.number:
                trueList.append(item)
            else:
                falseList.append(item)
        return trueList, falseList

    def weightedGiniCalculator(self, trueList, falseList, gini):
        trueGini = self.GiniImpurity(trueList)
        falseGini = self.GiniImpurity(falseList)
        prob = len(trueList) / (len(falseList) + len(trueList))
        return prob*trueGini + (1-prob)*falseGini

    def findQuestion(self, giniList: list[list[float]]):
        """Find the optimal gini and question for the given  dataset"""
        bestGini = 1
        bestQuestion = None
        gini = self.GiniImpurity(giniList)
        trueGini = 0
        falseGini = 0
        for index in range(4):
            for item in giniList:
                question = Question(item[index], index)
                trueList, falseList = self.splitTree(giniList, question)
                if len(trueList) == 0 or len(falseList) == 0:
                    continue

                gini = self.weightedGiniCalculator(trueList, falseList, gini)
                if gini <= bestGini:
                    bestGini = gini
                    bestQuestion = question
                    self.trueList = trueList
                    self.falseList = falseList
        return bestGini, bestQuestion

    def isClassPure(self, countData: list[list[float]]) -> bool:
        """If our dataset is pure and have only 1 label then there is no need for
        more decision and gini calculation"""
        if not countData:
            return True
        label = countData[0][4]
        return all(label == countData[i][4] for i in range(1, len(countData)))

    def countClass(self, countData: list[list[float]]):
        count0 = 0
        count1 = 0
        count2 = 0
        max = 0
        for countDatum in countData:
            if countDatum[4] == 0:
                count0 += 1
            elif countDatum[4] == 1:
                count1 += 1
            elif countDatum[4] == 2:
                count2 += 1
        if (count1 > count0):
            max = 1
        elif(count2 > count0):
            max = 2
        return ("["+str(count0)+", "+str(count1)+", "+str(count2)+"]"), max
        # return ("Number of Iris-setosa is => " + str(count0) + "\n" +
        #       "Number of Iris-versicolor is => " + str(count1)+"\n"+"Number of Iris-virginica is =>" + str(count2))

    def GiniImpurity(self, countData: list[list[float]]):
        count0 = 0
        count1 = 0
        count2 = 0
        length = len(countData)
        for i in range(length):
            if countData[i][4] == 0:
                count0 += 1
            elif countData[i][4] == 1:
                count1 += 1
            elif countData[i][4] == 2:
                count2 += 1
        return 1-((count0/length)**2)-((count1/length)**2)-((count2/length)**2)

    def constructTree(self, train_data: list[list[float]], depth: int):
        """Construct our tree with given maxDepth limit """
        gain, question = self.findQuestion(train_data)
        if (
            self.isClassPure(train_data) == True
            or self.isClassPure(train_data) != True
            and depth >= self.max_depth
        ):
            message, info = self.countClass(train_data)
            return(LeafNode(message, info))
        else:
            trueList, falseList = self.splitTree(train_data, question)
            gini = self.GiniImpurity(train_data)

            trueBranch = self.constructTree(trueList, depth+1)  # lesser
            falseBranch = self.constructTree(falseList, depth+1)  # greater
            return DecisionNode(gini, len(train_data), question, trueList, falseList, trueBranch, falseBranch)

    def printTree(self, node, spacing):
        """Classical Tree printer code that I found from online"""
        if isinstance(node, LeafNode):
            print(spacing + node.leafInfo)
            return
        print(spacing + node.questionAtt +
              " is <= " + str(node.questionNumber) + " gini= " + str(node.gini)+" samples= " + str(node.samples))

        print(spacing + '--> True:')
        self.printTree(node.trueBranch, spacing + " ")
        print(spacing + '--> False:')
        self.printTree(node.falseBranch, spacing + " ")


class DecisionNode:
    """Node that has Questions in it"""

    def __init__(self, gini: float, samples: int, question, trueList: list[list[float]], falseList: list[list[float]], trueBranch, falseBranch) -> None:
        self.gini = gini
        self.samples = samples
        self.questionAttNumber = question.att
        if question.att == 0:
            self.questionAtt = "Sepal Length"
        elif question.att == 1:
            self.questionAtt = "Sepal Width"
        elif question.att == 2:
            self.questionAtt = "Petal Length"
        elif question.att == 3:
            self.questionAtt = "Petal Width"
        self.questionNumber = question.number
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.trueList = trueList  # number that are lesser than our Number
        self.falseList = falseList  # number that are higher than our Number


class LeafNode:
    """Node that contains information about data"""

    def __init__(self, leafInfo: str, type: int) -> None:
        """like [0, 2, 0] 0 occ of type 0 / 2 occ of type 1 etc."""
        self.leafInfo = leafInfo
        self.type = type


class Question:
    """Question that contains which att has <= relation with data"""

    def __init__(self, number, att) -> None:
        self.number = number
        self.att = att


pass

   # X, y = ...
   # X_train, X_test, y_train, y_test = ...

   # clf = DecisionTreeClassifier(max_depth=5)
   # clf.fit(X_train, y_train)
   # yhat = clf.predict(X_test)
