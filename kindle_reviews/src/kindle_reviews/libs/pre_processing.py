from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F
from kindle_reviews.libs.utils.helper import col_lower, regex_replace


class PreProcessing:
    def __init__(self, df: DataFrame) -> None:
        self.data = df

    def clean_text(self) -> None:
        df = (self.data
              .transform(col_lower(column='reviewText'))
              .transform(regex_replace(column='reviewText', regex='^rt ', new_value=''))
              .transform(regex_replace(column='reviewText', regex='(https?\\://)\\S+', new_value=''))
              .transform(regex_replace(column='reviewText', regex='[^a-zA-Z0-9\\s]', new_value=''))
              )
        self.data = df

    def create_label(self) -> None:
        df = self.data.withColumn('label', F.when(F.col('overall') < 4, 0)
                                  .otherwise(F.lit(1)))
        self.data = df

    def select_features(self) -> None:
        self.data = self.data.na.drop()
        self.data = self.data.withColumn("monotonically_increasing_id", F.monotonically_increasing_id())
        window = Window.orderBy(F.col('monotonically_increasing_id'))
        self.data = self.data.withColumn('id', F.row_number().over(window))
        self.data = self.data.select(['id', 'reviewText', 'label'])
