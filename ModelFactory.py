from multiprocessing.pool import ThreadPool

Class CreateBestModel(self, algo, avgfscore, avgrecall, hyperparams, best_model, result):
  def __init__(self, algo, avgfscore, avgrecall, hyperparams, best_model, result):
    self.algo = algo
    self.avgFScore = avgfscore
    self.avgRecall = avgrecall
    self.hyperParams = hyperparams
    self.model = best_model
    self.gsResult = result

Class ModelFactory(self, df, n_thread):
  def __init__(self,df):
    self.folds = df.ramdomsplit[0.2,0.2,0.2,0.2,0.2]
    self.pool = ThreadPool(n_thread)
    self.modelobj = None

  @private class method
  def _generateParamGrid(self, *args):
    grid = itertools.product(*args)
    return grid

  @private class method
  def _generateClassifier(self, algo, params):

    ############################################################################
    #TODO: complete this section

    def lr(self, params):
      return lr classifier

    def gbm(self, params):
      return gbm classifier

    def rf(self,params)
      return rf classifer

    def mlp(self, params)
      return mlp classifier

    def svm(self,params):
      return mlp classifier
    ############################################################################

    getClassifier = {
    'lr':lr
    'gbm':gbm
    'rf':rf
    'mlp':mlp
    'svm':svm
    'xg':xg}

    return getClassifier[algo](self,params)

  @private class method
  def _crossValidate(self, folds, classifier, undersample_ratio):
    recall_sum = 0
    precision_sum = 0
    F1_sum = 0
    for fold in folds:
      validation = fold
      ############################################################################
      #TODO: translate to pyspark sql
      training = modeling_df - validation
      undersampled_training = undersample(training, non-fraud=0.7)
      model = undersampled_training.fit
      validation_result = model.test(validation)
      recall = validation_result.recall
      precision = validation_result.precision
      ############################################################################
      F1 = 2 * precision * recall / (precision + recall)
      recall_sum += recall
      precision_sum = precision
    metrics = [recall_sum, precision_sum, F1_sum]
    avgmetrics[:] = [metric / 5 for metric in metrics]
    return avgmetrics

  def _collectResult(self, folds, algo, params, undersample_ratio):
    params = list(params)
    classifier = self._generateClassifier(algo, params)
    modelPerformance = params + self._crossvalidate(folds, classifier, undersampe_ratio)
    return modelPerformance

  @private class method
  def _gridSearch(self，algo, grid, undersampe_ratio):
    folds = self.folds

    #for params in grid:
      #classifier = generateClassifier(algo, list(params))
      #modelPerformance = paramcombo + crossvalidate(folds, classifier, undersampe_ratio)
    #ind = avgFScore.indexOf(max(avgFscore))
    #return result[ind][:-3]

    result = self.pool.map(lambda params: self._collectResult(folds, algo, params, undersample_ratio), grid)
    return result

  @private class method
  def _getBestModel(self,algo,params):
    #TODO: convert to pyspark
    ############################################################################
    classifer = _generateClassifier(algo, bestHyperParams)
    df = join(self.folds)
    bestmodel = classifier.fit(df)
    return bestmodel
    ############################################################################

  @public class method
  def tune(self, algo, undersample_ratio, *args):
    #generate hyper parameter grid
    grid = self._generateParamGrid(*args)

    #conduct grid search:
    results = gridSearch(self，algo, parameterGrid, undersampe_ratio)
    max_fscore = 0
    besthyperparams = None
    for result in results:
      if result[-1] > max_fscore:
        max_fscore = result[-1]
        besthyperparams = result[:-3]
    best_model = self._getBestModel(algo,besthyperparams)
    self.modelobj = CreateBestModel(algo, avgfscore, avgrecall, besthyperparams, best_model, result)
    return self.modelobj

  #TODO: convert to pyspark code
  ############################################################################
  @public class method
  def save(self):
    modelobj = self.modelobj
    save modelobj as modelobj.algo.dat
  ############################################################################
