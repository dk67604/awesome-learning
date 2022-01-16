import logging

from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StopWordsRemover, VectorAssembler, CountVectorizer
from nltk.stem.porter import *


class Transformer:
    def __init__(self, transformer_columns: dict):
        logging.info('Transformer initialized')
        self.transformer_columns = transformer_columns
        self.stemmer = PorterStemmer()

    def remove_stopwords(self) -> StopWordsRemover:
        remover = StopWordsRemover(inputCol=self.transformer_columns['stopwords']['inputCol'],
                                   outputCol=self.transformer_columns['stopwords']['outputCol'])
        return remover

    def get_tokenizer(self) -> Tokenizer:
        """
        A method for tokenizer transformer
        :return: Tokenizer object
        """
        tokenizer = Tokenizer(inputCol=self.transformer_columns['tokenizer']['inputCol'],
                              outputCol=self.transformer_columns['tokenizer']['outputCol'])
        return tokenizer

    def count_vectorizer(self) -> CountVectorizer:
        count_vectors = CountVectorizer(inputCol=self.transformer_columns['vectorizer']['inputCol'],
                                        outputCol=self.transformer_columns['vectorizer']['outputCol'],
                                        vocabSize=10000, minDF=5)
        return count_vectors

    def hashing_tf(self) -> HashingTF:
        """
        A method for hashing term frequency transformer
        :return:
        """
        hashingTF = HashingTF(inputCol=self.transformer_columns['hashing']['inputCol'],
                              outputCol=self.transformer_columns['hashing']['outputCol'],
                              numFeatures=10000)
        return hashingTF

    def idf(self) -> IDF:
        """
        A method for inverse document frequency transformer
        :return:
        """
        idf = IDF(inputCol=self.transformer_columns['idf']['inputCol'],
                  outputCol=self.transformer_columns['idf']['outputCol'],
                  minDocFreq=8)
        return idf

    def vector_assembler(self) -> VectorAssembler:
        vec_assembler = VectorAssembler(inputCols=self.transformer_columns['vector_assembler']['inputCols'],
                                        outputCol=self.transformer_columns['vector_assembler']['outputCol'])
        return vec_assembler
