#libraries for modeling
from multiprocessing.pool import ThreadPool
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession, Window, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
import itertools
from itertools import repeat
import pickle
import pyspark
import copy

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#libraries for plotting
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class CreateBestModel:
    def __init__(self, algo, avgprecision, avgrecall, avgfscore, hyperparams, best_model):
        self.algo = algo
        self.avgPrecision = avgprecision
        self.avgFScore = avgfscore
        self.avgRecall = avgrecall
        self.hyperParams = hyperparams
        self.model = best_model

#function-based
def sample(df, sampling_method, ratio):

    notfraud = df.select('*').where(df.Class == 0.0)
    fraud = df.select('*').where(df.Class == 1.0)

    if sampling_method == "over":
        nrows = notfraud.select("Class").count()
        sample_size = int(nrows*ratio/(1-ratio))
        sampled = fraud.rdd.takeSample(True, sample_size, 46)
        fraud = sqlContext.createDataFrame(sampled)

    elif sampling_method == "under":
        nrows = fraud.select("Class").count()
        sample_size = int(nrows*(1-ratio)/ratio)
        sampled = notfraud.rdd.takeSample(False, sample_size, 46)
        notfraud = sqlContext.createDataFrame(sampled)

    sampled = fraud.union(notfraud)
    fraud_count = sampled.select("Class").where(sampled.Class == 1.0).count()
    tot_count = sampled.select("Class").count()
    fraud_ratio = fraud_count / tot_count

    print("train after sampling: " + str(tot_count))
    print("fraud ratio: " + str(fraud_ratio))

    #shuffle undersampled dataframe
    nrows = sampled.select("Class").count()
    shuffled = sampled.rdd.takeSample(False, nrows)
    shuffled_df = sqlContext.createDataFrame(shuffled)

    return shuffled_df

def generateParamGrid(*args):
    grid = list(itertools.product(*args))
    return grid

def generateClassifier(algo, params, features):

    ############################################################################
    #TODO: complete this section

    def lr(params,features):
        lrClassifier = LogisticRegression(featuresCol = 'features',
                                          labelCol = 'Class',
                                          threshold=params[0])
                                          #maxIter=params[0],
                                          #regParam=params[1],
                                          #elasticNetParam=params[2])
        return lrClassifier


    def gbm(params,features):
        gbmClassifier = GBTClassifier(featuresCol = 'features',
                                      labelCol = 'Class',
                                      maxDepth = params[0],
                                      minInfoGain = params[1])
        return gbmClassifier

    def rf(params,features):
        rfClassifier = RandomForestClassifier(featuresCol='features',
                                              labelCol='Class',
                                              maxDepth=params[0],
                                              minInfoGain=params[1],
                                              numTrees=params[2])

        return rfClassifier

    def mlp(params,features):
        input_layers = len(features)
        layers = [input_layers, params[1], 2]
        print(layers)
        mlpClassifier = MultilayerPerceptronClassifier(featuresCol = 'features',
                                                       labelCol = 'Class',
                                                       maxIter = params[0],
                                                       layers = layers,
                                                       stepSize = params[2])
        return mlpClassifier

    def svm(params, features):
        return

    def xg(params,features):
        return
    ############################################################################

    getClassifier = {
        'lr':lr,
        'gbm':gbm,
        'rf':rf,
        'mlp':mlp,
        'svm':svm,
        'xg':xg}

    return getClassifier[algo](params,features)

def crossValidate(df, folds, k, classifier, features, sampling_method, ratio, pool):

    def build(fold, df, classifier, features, sampling_method, ratio):

        #undersample notfraud
        validation = fold
        train = df.subtract(fold)
        train = sample(train, sampling_method, ratio)
        vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
        vector_train = vectorAssembler.transform(train)
        vector_validate = vectorAssembler.transform(validation)
        model = classifier.fit(vector_train)
        pred = model.transform(vector_validate)
        pos = pred.filter(pred.prediction == 1.0).count()
        if pos != 0:
            precision = pred.filter(pred.Class == pred.prediction).filter(pred.Class == 1.0).count() / pos
        else:
            precision = 0
        fraud = pred.filter(pred.Class == 1.0).count()
        if fraud != 0:
            recall = pred.filter(pred.Class == pred.prediction).filter(pred.Class == 1.0).count() / fraud
        else:
            recall = 0
        precision_recall = precision + recall
        if precision_recall != 0:
            f_score = 2 * precision * recall /(precision_recall)
        else:
            f_score = 0
        print("\n precision, recall, f_score: " + str(precision) + ", " + str(recall) + ", " + str(f_score))
        return [precision, recall, f_score]

    #call multiprocessing here
    cvperformance = pool.map(lambda fold: build(fold, df, classifier, features, sampling_method, ratio), folds)

    #calculate metrics
    precision_sum = sum([x[0] for x in cvperformance])
    recall_sum = sum([x[1] for x in cvperformance])

    avg_precision = precision_sum/k
    avg_recall = recall_sum/k
    avg_fscore = 2 * avg_precision * avg_recall /(avg_precision+avg_recall)
    return [avg_precision,avg_recall,avg_fscore]

def gridSearch(df, folds, k, algo, grid, features, sampling_method, ratio, pool):

    best_hyper = None
    best_precision = 0
    best_recall = 0
    best_fscore = 0

    for i in range(len(grid)):
        params = list(grid[i])
        classifier = generateClassifier(algo, params, features)
        modelPerformance = crossValidate(df, folds, k, classifier, features, sampling_method, ratio, pool)
        if modelPerformance[2] > best_fscore:
            best_hyper = params
            best_precision = modelPerformance[0]
            best_recall = modelPerformance[1]
            best_fscore = modelPerformance[2]

    return best_hyper, best_precision, best_recall, best_fscore

def trainBestModel(df,algo,features,params):
    vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
    classifier = generateClassifier(algo, params, features)
    vector_train = vectorAssembler.transform(df)
    bestmodel = classifier.fit(vector_train)
    return bestmodel

def tune(df, k, stratification_flag, sampling_method, ratio, modelobj_flag, features, algo, *args):

    pool = ThreadPool(2)

    #reduce df dimenions to include features and class
    cols = features+['Class', 'index']
    df = df.select(cols)
    df = df.select(*(F.col(c).cast("double").alias(c) for c in df.columns))
    #df.drop("index")

    folds = []

    if stratification_flag == False:

        tot_count = df.select("Class").count()
        n = int(tot_count / k)

        #create sub-dataframe iteratively
        fold_start = 1
        fold_end = n
        for i in range(k):
            fold = df.select('*').where(df.index.between(fold_start, fold_end))
            folds.append(fold)
            fold_start = fold_end + 1
            fold_end = fold_start + n
            if i == k-2:
                end = tot_count

    #ensure each fold has the same number of records and same fraud ratio
    if stratification_flag == True:

        fraud = df.select("*").where(df.Class == 1.0)
        #shuffle undersampled dataframe
        nrows = fraud.select("Class").count()
        shuffled = fraud.rdd.takeSample(False, nrows)
        fraud = sqlContext.createDataFrame(shuffled)
        #add row index to dataframe
        fraud = fraud.withColumn('dummy', F.lit('7'))
        fraud = fraud.withColumn("temp_index", F.row_number().over(Window.partitionBy("dummy").orderBy("dummy")))
        fraud = fraud.drop('dummy')
        fraud_count = fraud.select("Class").count()
        each_fraud = int(fraud_count/k)

        notfraud = df.select("*").where(df.Class == 0.0)
        nrows = notfraud.select("Class").count()
        shuffled = notfraud.rdd.takeSample(False, nrows)
        notfraud = sqlContext.createDataFrame(shuffled)
        #add row index to dataframe
        notfraud = notfraud.withColumn('dummy', F.lit('7'))
        notfraud = notfraud.withColumn("temp_index", F.row_number().over(Window.partitionBy("dummy").orderBy("dummy")))
        notfraud = notfraud.drop('dummy')
        notfraud_count = notfraud.select("Class").count()
        each_notfraud = int(notfraud_count/k)

        fraud_start = 1
        fraud_end = each_fraud
        notfraud_start = 1
        notfraud_end = each_notfraud

        for i in range(k):
            fraud_fold  = fraud.select('*').where(fraud.temp_index.between(fraud_start, fraud_end))
            notfraud_fold = notfraud.select('*').where(notfraud.temp_index.between(notfraud_start, notfraud_end))
            fold = fraud_fold.union(notfraud_fold).drop("temp_index")
            folds.append(fold)
            fraud_start = fraud_end + 1
            fraud_end = fraud_start + each_fraud
            notfraud_start = notfraud_end + 1
            notfraud_end = notfraud_start + each_notfraud
            if i == k-2:
                fraud_end = fraud_count
                notfraud_end = notfraud_count

    #generate hyperparam combo
    grid = generateParamGrid(*args)

    #conduct grid search:
    best_hyper, best_precision, best_recall, best_fscore = gridSearch(df, folds, k, algo, grid, features, sampling_method, ratio, pool)

    if modelobj_flag == True:
        #generate a model obj
        best_model = trainBestModel(trimmed_df,algo,features,best_hyper)
        modelobj = CreateBestModel(algo, best_precision, best_recall, best_fscore, best_hyper, best_model)
        return modelobj

    return best_hyper, best_precision, best_recall, best_fscore

def save(modelobj, filename):

    modelobj = modelobj
    pickle.dump(modelobj, open(filename, "wb"))

def load(filename):

    modelobj = pickle.load(open(filename, "rb"))
    return modelobj

#############Example###########

sc=pyspark.SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

df = sqlContext.read.csv("modeling.csv", header = True)
features = ['V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
hiddenlayer = int((len(features) + 2) / 2)
cv_hyper, cv_precision, cv_recall, cv_fscore = tune(df, 5, True, 'None', 0, False, features, 'mlp', [100], [hiddenlayer], [0.03])
print("avg precision:", cv_precision)
print("avg recall:", cv_recall)
print("avg f-score:", cv_fscore)

sqlContext.clearCache()
