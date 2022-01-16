from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes, GBTClassifier


class PySparkModels:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params

    def initialize_model(self, classifier: str):
        def logistic_regression_model() -> LogisticRegression:
            mlr = LogisticRegression(maxIter=self.hyper_params['maxIter'],
                                     regParam=self.hyper_params['regParam'],
                                     elasticNetParam=self.hyper_params['elasticNetParam'])
            return mlr

        def random_forest_model() -> RandomForestClassifier:
            rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                        numTrees=self.hyper_params['numTrees'],
                                        maxDepth=self.hyper_params['maxDepth'],
                                        maxBins=self.hyper_params['maxBins'])
            return rf

        def gradient_boost_model() -> GBTClassifier:
            gb = GBTClassifier(maxDepth=self.hyper_params['maxDepth'],
                               maxBins=self.hyper_params['maxBins'],
                               maxIter=self.hyper_params['maxIter'])
            return gb

        if classifier in ['lr_tf', 'lr_tf_idf']:
            return logistic_regression_model()
        elif classifier == 'rf':
            return random_forest_model()
        elif classifier == 'gb':
            return gradient_boost_model()
        else:
            raise ValueError(f'Unsupported Classifier:{classifier}')
