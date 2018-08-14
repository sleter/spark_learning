from pyspark import SparkConf, SparkContext
import collections

conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    cust_id = int(fields[0])
    item_id = int(fields[1])
    amount_spent = float(fields[2])
    return (cust_id, amount_spent)

lines = sc.textFile("file:///C:/Users/N0tE/Documents/GitHub/spark_learning/customer-orders.csv")
rdd = lines.map(parseLine)
totalsByCustId = rdd.reduceByKey(lambda x, y: x+y)
results = totalsByCustId.collect()
sortedResults =  sorted(results, key=lambda x: x[1])
for result in sortedResults:
    print(result)
