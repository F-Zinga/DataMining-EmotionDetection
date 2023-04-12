from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import numpy as np


class WordVectorTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, model="en_core_web_lg"):
        self.model = model

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        nlp = spacy.load(self.model)

        docs = []

        for doc in X:

            filtered = [token.text for token in nlp(doc) if not token.is_stop  and not token.is_digit and not token.is_punct and not token.is_bracket and not token.like_num and not token.like_url and not token.is_quote]

            docs.append(filtered)

        final_docs = []

        for d in docs:

            j = ' '.join(d)
            final_docs.append(j)

        return np.concatenate([nlp(doc).vector.reshape(1,-1) for doc in final_docs])