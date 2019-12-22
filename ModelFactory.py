"""
###############################################################################
A suite of functions that integrated grid search, random/stratified k fold
cross-validation, and sampling during cross-validation. Users should customize
the code to meet their needs :)
###############################################################################
"""

from multiprocessing.pool import ThreadPool
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession, Window, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
import itertools
from itertools import repeat
import pickle
import pyspark

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier


class CreateBestModel:
    def __init__(self, algo, avgprecision, avgrecall, avgfscore, hyperparams,
    ootmodel, ootprecision, ootrecall, ootfscore):
        self.algo = algo
        self.gsPrecision = avgprecision
        self.gsFScore = avgfscore
        self.gsRecall = avgrecall
        self.hyperParams = hyperparams
        self.model = ootmodel
        self.ootPrecision = ootprecision
        self.ootFScore = ootfscore
        self.ootRecall = ootrecall

def sample(df, sampling_method, ratio):
    """
    implementation of random sampling
    Example:
        >>> sampled = sample(df, "under", 0.05)
    :param df: data for sampling
    :type df: : pyspark dataframe
    :param sampling_method: "over" for oversampling minority class,
                            "under" for undersampling majority class, "None"
    :type sampling_method: str
    :param ratio: targeted fraud ratio after sampling.
    :type ratio: float
    :returns: sampled dataframe
    """

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
    else:
        return df

    sampled = fraud.union(notfraud)

    #shuffle undersampled dataframe
    nrows = sampled.select("Class").count()
    shuffled = sampled.rdd.takeSample(False, nrows, 46)
    shuffled_df = sqlContext.createDataFrame(shuffled)

    return shuffled_df

def generateParamGrid(*args):

        """
        implementation of a hyperparameter grid for grid search
        Example:
            >>> gird = generateParamGrid([1,2], [3,4])
        :param *args: a sequence of params for hyperparams tuning.
                ex. [values for params1], [values for params2],...
        :type *args: list
        :returns: cartesian product of the sequence of params
                ex. [[1,3], [1,4], [2,3], [2,4]]
        """

    grid = list(itertools.product(*args))
    return grid

def generateClassifier(algo, params, features):

    """
    interface of creating a classifier
    Example:
        >>> mlp = generateClassifier("mlp", [15,100,0.03], ["V1", "V2"])
    :param algo: algorithm for modeling. "lr", "svm", "rf", "gbm", "mlp"
    :type algo: str
    :param params: hyperparameters for the classifier
    :type params: list
    :param features: features for training
    :type features: list

    :returns: classifier object
    """

    def lr(params,features):
        print(params)
        if len(params) > 2:
            lrClassifier = LogisticRegression(featuresCol = 'features',
                                          labelCol = 'Class',
                                          threshold=params[0],
                                           maxIter=params[1],
                                           weightCol=params[2])
                                          #regParam=params[2])
                                          #elasticNetParam=params[2])
        else:
            lrClassifier = LogisticRegression(featuresCol = 'features',
                                          labelCol = 'Class',
                                          threshold=params[0],
                                           maxIter=params[1])
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
        if len(params) > 3:
            svmClassifier = LinearSVC(featuresCol = 'features',
                         labelCol='Class',
                         maxIter=params[0],
                         regParam=params[1],
                         tol =params[2],
                         weightCol=params[3]
                         )

        else:
            svmClassifier = LinearSVC(featuresCol = 'features',
                         labelCol='Class',
                         maxIter=params[0],
                         regParam=params[1],
                         tol =params[2]
                         )

        return svmClassifier

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

    """
    implementation of k fold cross-validation (private function)
    Example:
        >>> metrics = crossValidate(df, folds, 5, mlp, ["V1", "V2"], "under", 0.05, pool)
    :param df: data for modeling purpose
    :type df: : pyspark dataframe
    :param folds: collection of data split into folds
    :type: folds: list of dataframes
    :param k: number of folds for cross validation
    :type k: int
    :param sampling_method: "over" for oversampling minority class,
                            "under" for undersampling majority class, "None"
    :type sampling_method: str
    :param ratio: targeted fraud ratio after sampling
    :type ratio: float, between 0 and 1

    :returns: a list of metrics after cross-validation
    """

    def build(fold, df, classifier, features, sampling_method, ratio):

        validation = fold
        train = df.subtract(fold)
        train = sample(train, sampling_method, ratio)
        fraud_count = train.select("Class").where(train.Class == 1.0).count()
        tot_count = train.select("Class").count()
        fraud_ratio = fraud_count / tot_count
        print("train: " + str(tot_count))
        print("fraud ratio: " + str(fraud_ratio))

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
    if avg_precision+avg_recall == 0:
        avg_fscore = 0
    else:
        avg_fscore = 2 * avg_precision * avg_recall /(avg_precision+avg_recall)
    return [avg_precision,avg_recall,avg_fscore]

def gridSearch(df, folds, k, algo, grid, features, sampling_method, ratio, pool):

    """
    implementation of grid search (private function)
    Example:
        >>> best_hyper, best_precision, best_recall, best_fscore = gridSearch(df, folds, k, algo, grid, features, sampling_method, ratio, pool)

    :param df: data for modeling purpose
    :type df: : pyspark dataframe
    :param folds: collection of data split into folds
    :type: folds: list of dataframes
    :param k: number of folds for cross validation
    :type k: int
    :param algo: algorithm for modeling. "lr", "svm", "rf", "gbm", "mlp"
    :type algo: str
    :param grid: hyperparameter grid
    :type grid: nested list
    :param features: features for training
    :type features: list
    :param sampling_method: "over" for oversampling minority class,
                            "under" for undersampling majority class, "None"
    :type sampling_method: str
    :param ratio: targeted fraud ratio after sampling.
    :type ratio: float
    :returns: a list of metrics after cross-validation
    """

    best_hyper = None
    best_precision = 0
    best_recall = 0
    best_fscore = 0

    for i in range(len(grid)):
        params = list(grid[i])
        print(params)
        classifier = generateClassifier(algo, params, features)
        modelPerformance = crossValidate(df, folds, k, classifier, features, sampling_method, ratio, pool)
        if modelPerformance[2] > best_fscore:
            best_hyper = params
            best_precision = modelPerformance[0]
            best_recall = modelPerformance[1]
            best_fscore = modelPerformance[2]

    return best_hyper, best_precision, best_recall, best_fscore

def ootTest(traindf,testdf,algo,features,params):

    """
    deprecated
    """

    vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
    classifier = generateClassifier(algo, params, features)
    vector_train = vectorAssembler.transform(traindf)
    vector_test = vectorAssembler.transform(testdf)
    ootmodel = classifier.fit(vector_train)
    pred = ootmodel.transform(vector_test)
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

    return ootmodel, precision, recall, f_score

def tune(df, k, stratification_flag, sampling_method, ratio, modelobj_flag, features, algo, *args):

    """
    Entry point of this suite of functions. returns cv metrics or a model object
    Example:
        >>> cv_hyper, cv_precision, cv_recall, cv_fscore = tune(df, 5, True,
        'None', 0, False, features, 'mlp', [300], [15], [0.03])
    :param df: data for modeling purpose
    :type df: : pyspark dataframe
    :param k: number of folds for cross validation
    :type k: int
    :param stratification_flag: specifies whether fraud ratio is fixed for each fold. True for stratification
    :type stratification_flag: boolean
    :param sampling_method: "over" for oversampling minority class,
                            "under" for undersampling majority class, "None"
    :type sampling_method: str
    :param ratio: targeted fraud ratio after sampling.
    :type ratio: float
    :param modelobj_flag: specifies whether to return a model object for out of time test.
                          if False, returns cv performancce
    :type modelobj_flag: float
    :param features: features for training
    :type features: list
    :param algo: algorithm for modeling. "lr", "svm", "rf", "gbm", "mlp"
    :type algo: str
    :param *args: a sequence of params for hyperparams tuning. ex. [values for params1], [values for params2],...
    :type *args: list
    :returns: model object, cross validation metrics, selected hyperparams if multiples are given.
              modelobj_flag dependent.
    """

    pool = ThreadPool(2)

    #reduce df dimenions to include features and class
    cols = features+['Class', 'index']
    df = df.select(cols)
    df = df.select(*(F.col(c).cast("double").alias(c) for c in df.columns))
    df.cache()
    #df.drop("index")

    ########################ClassWeights#################################
    if algo in ["lr", "svm"] and ["ClassWeigts"] in args:
        #add class weight
        balance_ratio = args[-1][0]
        df=df.withColumn("classWeights", when(df.Class == 1.0,balance_ratio).otherwise(1-balance_ratio))
    ########################ClassWeights#################################

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

    if stratification_flag == True:
        fraud = df.select("*").where(df.Class == 1.0)
        #shuffle undersampled dataframe
        nrows = fraud.select("Class").count()
        shuffled = fraud.rdd.takeSample(False, nrows, 46)
        fraud = sqlContext.createDataFrame(shuffled)
        #add row index to dataframe
        fraud = fraud.withColumn('dummy', F.lit('7'))
        fraud = fraud.withColumn("temp_index", F.row_number().over(Window.partitionBy("dummy").orderBy("dummy")))
        fraud = fraud.drop('dummy')
        fraud_count = fraud.select("Class").count()
        each_fraud = int(fraud_count/k)

        notfraud = df.select("*").where(df.Class == 0.0)
        nrows = notfraud.select("Class").count()
        shuffled = notfraud.rdd.takeSample(False, nrows, 46)
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
        traindf = sample(df, sampling_method, ratio)
        testdf = sqlContext.read.csv("oot.csv", header = True)
        cols = features+['Class', 'index']
        testdf = testdf.select(cols)
        testdf = testdf.select(*(F.col(c).cast("double").alias(c) for c in testdf.columns))
        model, precision, recall, fscore = ootTest(traindf, testdf, algo,features,best_hyper)

        modelobj = CreateBestModel(algo, best_precision, best_recall, best_fscore, best_hyper,
                                   model, precision, recall, fscore)
        return modelobj

    return best_hyper, best_precision, best_recall, best_fscore

def save(content, filename):

    pickle.dump(content, open(filename, "wb"))

def load(filename):

    content = pickle.load(open(filename, "rb"))
    return content

def generateStratifiedFolds(df,k):

    """
    deprecated
    """

    return folds
