import sys, pickle
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

def loadData():
    movieNames = {}
    with open("ml-100k/u.ITEM", encoding='ascii', errors="ignore") as file:
        for line in file:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

# Running locally with as many threads as logical cores on machine
conf = SparkConf().setMaster("local[*]").setAppName("ALSTest")
sc = SparkContext(conf = conf)
# Set the directory under which RDDs are going to be checkpointed
sc.setCheckpointDir('checkpoint')


nameDict = loadData()

data = sc.textFile("file:///C:/Users/N0tE/Documents/GitHub/spark_learning/recruitment_task/ml-100k/u.data")

ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()

# Build simple recommendation model using Alternating Least Squares
rank = 10
# Lowered numIterations to ensure it works on lower-end systems
numIterations = 6
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))


# Creating output numpy arrays
userFeatures = model.userFeatures().collect()
productFeatures = model.productFeatures().collect()

userFeatureNumpyArray = []
productFeatureNumpyArray = []

# Appending to NumPy arrays is catastrophically slower than appending to ordinary lists
for feature in userFeatures:
    element = (feature[0], np.asarray(feature[1]))
    userFeatureNumpyArray.append(element)

for feature in productFeatures:
    element = (feature[0], np.asarray(feature[1]))
    productFeatureNumpyArray.append(element)

dt = np.dtype([('id', np.int32), ('predictions', np.float64, (10,))])

userFeatureNumpyArray = np.array(userFeatureNumpyArray, dtype=dt)
itemFeatureNumpyArray = np.array(productFeatureNumpyArray, dtype=dt)

pickle_output = {'userFeature': userFeatureNumpyArray, 'itemFeature': itemFeatureNumpyArray}

pickle.dump(pickle_output, open("modelals.p","wb"))