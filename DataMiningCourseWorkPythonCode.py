



import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.model_selection import train_test_split as TTS
from keras.layers import Dense as D
from keras.models import Sequential as Seq



# location data set
# calculating the distance between two points
def dist(p1, p2):
    sumTotal = 0

    for c in range(len(p1)):
        sumTotal = sumTotal + pow((p1[c] - p2[c]),2)

    return math.sqrt(sumTotal)

# finding the minimum distance btween a point and a set of points
def minDistPos(point, matrix):
    minPos = -1
    minValue = float("inf")

    for rowPos in range(len(matrix)):
        d = dist(point,matrix[rowPos,:])

        if (d < minValue):
            minValue = d
            minPos = rowPos

    return minPos
        
# finding the sum of the distances between two matrices
def sumDist(m1, m2):
    sumTotal = 0
    
    for pos in range(len(m1)):
        sumTotal = sumTotal + dist(m1[pos,:],m2[pos,:])

    return sumTotal

# standardising the data
def standard(data):
    standardData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        sigma = np.std(data[:,j])
        mu = np.mean(data[:,j])

        for i in range(rows):
            standardData[i,j] = (data[i,j] - mu)/sigma

    return standardData

# loading the data

dataraw = []

datafile = open("/Users/ramakrishna/Desktop/location2.csv", "r")
while True:
    theline = datafile.readline()
    if len(theline) == 0:
        break
    readdata = theline.split(",")
    for pos in range(len(readdata)):
        readdata[pos] = float(readdata[pos])
    dataraw.append(readdata)
    
datafile.close()

location = np.array(dataraw)
location

standardisedDataL = standard(location)
standardisedDataL

columns = 2

# Number of clusters
k = 2

# Initial centroids
C = np.random.random((k,columns))

# scale the random numbers
for f in range(columns):
    maxValue = standardisedDataL[:,f].max()
    minValue = standardisedDataL[:,f].min()

    for c in range(k):
        C[c,f] = minValue + C[c,f] * (maxValue - minValue)
    
C_old = np.zeros(C.shape)

clusters = np.zeros(len(standardisedDataL))

distCentroids = float("inf")

threshold = 0.1

while distCentroids > threshold:
    for i in range(len(standardisedDataL)):
        clusters[i] = minDistPos(standardisedDataL[i], C)

    C_old = C.copy()

    for i in range(k):
        points = np.array([])

        for j in range(len(location)):
            if (clusters[j] == i):
                if (len(points) == 0):
                    points = standardisedDataL[j,:].copy()
                else:
                    points = np.vstack((points,standardisedDataL[j,:]))

        C[i] = np.mean(points, axis=0)
        
    distCentroids = sumDist(C, C_old)

    
centroids = C

group1 = np.array([])

group2 = np.array([])

for d in standardisedDataL:
    if (dist(d, centroids[0,:]) < dist(d, centroids[1,:])):
        if (len(group1) == 0):
            group1 = d
        else:
            group1 = np.vstack((group1,d))
    else:
        if (len(group2) == 0):
            group2 = d
        else:
            group2 = np.vstack((group2,d))

plt.figure(figsize=(6,4))

#plotting the location data

plt.plot(group1[:,0],group1[:,1],'r.')
plt.plot(group2[:,0],group2[:,1],'g.')

plt.plot(centroids[0,0],centroids[0,1],'rx')
plt.plot(centroids[1,0],centroids[1,1],'gx')






# mushroom dataset with missing values
#loading the dataset
mush = pd.read_csv("/Users/ramakrishna/Desktop/mushroom2.csv")

# converting the categorical string variables to float
labelencoder=LabelEncoder()
for column in mush.columns:
    mush[column] = labelencoder.fit_transform(mush[column])

mushF = mush.astype(float)

mushFA = np.array(mushF)

# filling up the missing values with the imputation method
# to get the median value of the vector
def getMedian(vector):
    newVector = np.array([])
    for item in vector:
        if (not np.isnan(item)):
            newVector = np.append(newVector ,item)
    return np.median(newVector)

# replacing the NaN values in the new vector
def replace(vector , value):
    newVector = vector.copy()
    for pos in range(len(vector)):
        if (np.isnan(vector[pos])):
            newVector[pos] = value
            return newVector
# calculating the median and replacing the NaN with the median value
def replaceByMedian(data):
    newData = data.copy()
    for columnPos in range(len(data[0,:])):
        for rowPos in range(len(data)):
            if (np.isnan(data[rowPos,columnPos])):
                # Found a NaN. We must now get the median of the column, 
                # and replace all NaNs with the median value.
                median = getMedian(data[:,columnPos]) 
                newData[:,columnPos] = replace(data[:,columnPos],median)
                break 
    return newData

# performing imputation on the mushroom data set
mushIm = replaceByMedian(mushFA)



#assigning the labels and parameters to the imputed mushroom data set
x = mushIm[:, 1:23]
y = mushIm[:, 0:1]

# splitting the imputed data into training and test data sets
MTrainX, MTestX, MTrainY, MTestY = TTS(x, y, test_size=.2, random_state=5)

# creating a model to perform neural network classification
model = Seq()
model.add(D(1, input_dim=22,  activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(MTrainX, MTrainY, validation_data=(MTestX, MTestY), epochs=100, batch_size=25)






# Abalone imbalanced data set
# importing the abalone imabalced data set
abln = pd.read_csv("/Users/ramakrishna/Desktop/abalone.csv")


# converting the categorical string variables to float
labelencoder=LabelEncoder()
for column in abln.columns:
    abln[column] = labelencoder.fit_transform(abln[column])

ablnF = abln.astype(float)
ablnFA = np.array(ablnF)

# assigning the labels and parameters
AData = ablnFA[:, 0:8]
ALabels = ablnFA[:,8]


# splitting the data and labels into training and testing
ATrainX, ATestX, ATrainY, ATestY = TTS(AData, ALabels, test_size=0.2, random_state=0)

# original labels count
unique, count = np.unique(ATrainY, return_counts=True)
print('Original labels count: ', unique, count)

# applying Smote from imblearn.oversampling to generate new data
S = SMOTE(random_state=2)
ATrainX_res, ATrainY_res = S.fit_sample(ATrainX, ATrainY)

# labels count afetr oversampling
unique1, count1 = np.unique(ATrainY_res, return_counts=True)
print('labels count after oversampling: ', unique1, count1)


# classification of the oversampled data by using k nearest neighbour classifier
classifier = neighbors.KNeighborsClassifier(15)
classifier.fit(ATrainX_res, ATrainY_res)

# Prediction from the model and accuracy
P = classifier.predict(ATrainX)
accuracy = classifier.score(ATrainX, ATrainY)
print('The oversampled training model accurately predicts: ' + str(accuracy))

# generating an confusion matrix for testing data
TrainConfMatrix = confusion_matrix(ATrainY, P)
TrainConfMatrix
print("The recall metric for the Training data: {}%".format(100*TrainConfMatrix[1,1]/(TrainConfMatrix[1,0]+TrainConfMatrix[1,1])))

#classification of oversampled testing data by using knn
P1 = classifier.predict(ATestX)
accuracy1 = classifier.score(ATestX, ATestY)
print('The oversampled test model accurately predicts: ' + str(accuracy1))

# Put the result into a confusion matrix for test data
TestConfMatrix = confusion_matrix(ATestY, P1)
TestConfMatrix
print("The recall metric for the test data: {}%".format(100*TestConfMatrix[1,1]/(TestConfMatrix[1,0]+TestConfMatrix[1,1])))



