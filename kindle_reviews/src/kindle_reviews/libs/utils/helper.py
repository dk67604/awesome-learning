import yaml
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

from kindle_reviews.libs.models import PySparkModels
from kindle_reviews.libs.transformer import Transformer
from kindle_reviews.libs.utils.spark_class import PySparkClass


def col_lower(column: str):
    def _(df: DataFrame) -> DataFrame:
        return df.withColumn(column, F.lower(F.col(column)))

    return _


def regex_replace(column: str, regex: str, new_value: str):
    def _(df):
        return df.withColumn(column, F.regexp_replace(column, regex, new_value))

    return _


def stemmer_udf(stemmer):
    return F.udf(lambda x: stem(x, stemmer), ArrayType(StringType()))


def stem(in_vec, stemmer):
    out_vec = []
    for t in in_vec:
        t_stem = stemmer.stem(t)
        if len(t_stem) > 2:
            out_vec.append(t_stem)
    return out_vec


def read_yaml(file_path: str) -> dict:
    """
    A method load the YAML configuration from given file path
    :param file_path:
    :return: dict object
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def spark_start(spark_config: dict) -> SparkSession:
    """
    Start a spark session
    :param spark_config:
    :return:
    """
    if isinstance(spark_config, dict):
        return PySparkClass(config=spark_config).sparkStart()


def stop_spark(spark) -> None:
    """
    End Spark session
    :param spark:
    :return:
    """
    spark.stop() if isinstance(spark, SparkSession) else None


def build_transformer(transformer: Transformer, classifier: str) -> list:
    transform_chain = []
    if classifier == 'lr_tf':
        transform_chain.append(transformer.get_tokenizer())
        transform_chain.append(transformer.remove_stopwords())
        transform_chain.append(transformer.count_vectorizer())
    elif classifier == 'lr_tf_idf':
        transform_chain.append(transformer.get_tokenizer())
        transform_chain.append(transformer.remove_stopwords())
        transform_chain.append(transformer.hashing_tf())
        transform_chain.append(transformer.idf())
    elif classifier == 'rf':
        transform_chain.append(transformer.get_tokenizer())
        transform_chain.append(transformer.remove_stopwords())
        transform_chain.append(transformer.count_vectorizer())
    elif classifier == 'gb':
        transform_chain.append(transformer.get_tokenizer())
        transform_chain.append(transformer.remove_stopwords())
        transform_chain.append(transformer.count_vectorizer())
    else:
        raise ValueError(f'Unsupported Classifier{classifier}')
    return transform_chain


def build_pipeline(transform_chain: list, pyspark_model: PySparkModels, classifier: str) -> list:
    pipeline_stages = []
    model = pyspark_model.initialize_model(classifier=classifier)
    for item in transform_chain:
        pipeline_stages.append(item)
    pipeline_stages.append(model)
    return pipeline_stages
