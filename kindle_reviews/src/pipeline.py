import argparse
import os.path
import sys
import logging
from datetime import datetime
from traceback import format_exc

from pyspark.ml import Pipeline

from kindle_reviews.libs.models import PySparkModels
from kindle_reviews.libs.pre_processing import PreProcessing
from kindle_reviews.libs.transformer import Transformer
from kindle_reviews.libs.utils.data import load_from_csv, train_test_split, sava_data_parquet, load_from_parquet
from kindle_reviews.libs.utils.helper import read_yaml, spark_start, stop_spark, build_transformer, build_pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def get_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_config", help=" Transformer columns YAML config", type=str, required=True)
    parser.add_argument("--appname", help="Spark application name", type=str, required=False,
                        default='Sentimental Analysis')
    parser.add_argument("--dataset_name", help="Dataset name", type=str, required=True)
    parser.add_argument("--dataset_path", help="Dataset path", type=str, required=True)
    parser.add_argument("--model_hyper_param", help="Model Hyper parameters YAML file", type=str, required=True)
    parser.add_argument("--classifier", help="Classifier name, e.g.: Logistic Regression-lr_tf,lr_tf_idf",
                        type=str, required=True)
    parser.add_argument("--output_path", help="Output folder", type=str, required=True)
    args = parser.parse_args(args)
    return args


def main():
    try:
        start_time = datetime.now()
        task = sys.argv[0][:-3]
        logging.info(f'{task} starts at {start_time}')
        args = get_args(sys.argv[1:])
        transformer_config = read_yaml(args.tf_config)
        hyper_params = read_yaml(args.model_hyper_param)
        config_map = dict()
        config_map.update(vars(args))
        print(config_map)
        spark_session = spark_start(spark_config=config_map)
        df = load_from_csv(file_path=config_map['dataset_path'], spark=spark_session)
        pre_processing = PreProcessing(df=df)
        pre_processing.clean_text()
        pre_processing.create_label()
        pre_processing.select_features()
        print('Pre-processing dataset schema:{}'.format(pre_processing.data.schema))
        # print(pre_processing.data.count())
        train, test = train_test_split(df=pre_processing.data)
        transformer = Transformer(transformer_columns=transformer_config)
        transform_chain = build_transformer(classifier=config_map['classifier'], transformer=transformer)
        pyspark_model = PySparkModels(hyper_params=hyper_params[config_map['classifier']])
        pipeline_stages = build_pipeline(transform_chain=transform_chain, classifier=config_map['classifier'],
                                         pyspark_model=pyspark_model)
        pipeline = Pipeline(stages=pipeline_stages)
        # Train start here
        model = pipeline.fit(train)
        model_path = os.path.join(config_map['output_path'], 'models', config_map['classifier'])
        model.save(model_path)
        # Test starts here
        predictions = model.transform(test)
        # Evaluation starts here
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
        print('Accuracy:{}'.format(evaluator.evaluate(predictions)))
        if config_map['classifier'] != 'gb':
            training_summary = model.stages[-1].summary
            # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
            print(training_summary.roc.show())
            print("Area Under ROC: " + str(training_summary.areaUnderROC))
            print("Precision by label:")
            for i, prec in enumerate(training_summary.precisionByLabel):
                print("label %d: %s" % (i, prec))

            print("Recall by label:")
            for i, rec in enumerate(training_summary.recallByLabel):
                print("label %d: %s" % (i, rec))

            print("F-measure by label:")
            for i, f in enumerate(training_summary.fMeasureByLabel()):
                print("label %d: %s" % (i, f))

            accuracy = training_summary.accuracy
            falsePositiveRate = training_summary.weightedFalsePositiveRate
            truePositiveRate = training_summary.weightedTruePositiveRate
            fMeasure = training_summary.weightedFMeasure()
            precision = training_summary.weightedPrecision
            recall = training_summary.weightedRecall
            print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
                  % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
        # Save model artifacts
        prediction_path = os.path.join(config_map['output_path'], 'predictions', config_map['classifier'],
                                       'prediction.parquet')
        predictions_subset = predictions.select("id", "prediction", "label")
        sava_data_parquet(prediction_path, df=predictions_subset)
        train_path = os.path.join(config_map['output_path'], 'predictions', config_map['classifier'],
                                  'train.parquet')
        test_path = os.path.join(config_map['output_path'], 'predictions', config_map['classifier'],
                                 'validation.parquet')
        sava_data_parquet(test_path, df=test)
        sava_data_parquet(train_path, df=train)
        stop_spark(spark=spark_session)
    except (FileNotFoundError, Exception) as e:
        logging.error('Failed to execute pipeline')
        logging.error(format_exc())
        raise e


if __name__ == "__main__":
    main()
