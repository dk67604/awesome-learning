import logging

from pyspark.sql import SparkSession
import sys


class PySparkClass:

    def __init__(self, config: dict):
        self.config = config

    def sparkStart(self) -> SparkSession:
        """ create spark session from dict configuration"""

        try:
            def createBuilder(appname: str) -> SparkSession.Builder:
                """ create spark session"""
                builder: SparkSession.Builder = SparkSession. \
                    builder \
                    .appName(appname)
                return configSpark(builder=builder)

            def createSession(builder: SparkSession.Builder) -> SparkSession:
                if isinstance(builder, SparkSession.Builder):
                    return builder.getOrCreate()

            def configSpark(builder: SparkSession.Builder):
                if isinstance(builder, SparkSession.Builder):
                    builder \
                        .config("spark.pyspark.python", "python3") \
                        .config("spark.pyspark.virtualenv.enabled", "true") \
                        .config("spark.pyspark.virtualenv.type", "native") \
                        .config("spark.pyspark.virtualenv.bin.path", "/usr/bin/virtualenv")

                return builder

            _builder = createBuilder(self.config['appname'])
            _spark = createSession(_builder)
            logging.info('Spark Session created successfully.')

            return _spark

        except (ValueError, Exception) as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
