from typing import Tuple

from pyspark.sql import SparkSession, DataFrame


def load_from_csv(file_path: str, spark: SparkSession) -> DataFrame:
    """
    A method loads the data from the CSV file
    :param file_path: Location of the CSV format file
    :param spark: SparkSession object
    :return: a Spark dataframe
    """
    df = spark.read.csv(path=file_path, inferSchema=True, header=True)
    print('Dataframe Schema:{}'.format(df.columns))
    return df


def load_from_parquet(file_path: str, spark: SparkSession) -> DataFrame:
    """
    A method loads the data from the Parquet file
    :param file_path: Location of the CSV format file
    :param spark: SparkSession object
    :return: a Spark dataframe
    """
    df = spark.read.parquet(file_path)
    print('Dataframe Schema:{}'.format(df.columns))
    return df


def sava_data_parquet(file_path: str, df: DataFrame) -> None:
    """
    A method to save spark dataframe in parquet format
    :param file_path:
    :param df:
    :return:
    """
    df_pd = df.toPandas()
    df_pd.to_parquet(file_path, engine='pyarrow')


def train_test_split(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    A method to split the dataset
    :param df:
    :return:
    """
    train_split, test_split = df.randomSplit(weights=[0.80, 0.20], seed=13)
    return train_split, test_split
