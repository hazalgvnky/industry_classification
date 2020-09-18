import pandas as pd
import os
import sys


#CELL_1

data = pd.read_csv("/Users/macbookretina/Desktop/jotform.csv",nrows=1000)
data = data.drop_duplicates(subset=['form_id'])




#CELL_2
msk = data['industry'] == 'SOLAR ENERGY COMPANY'
data.loc[msk, 'industry'] = 'SOLAR ENERGY'

msk = data['industry'] == 'Religious, Grantmaking, Civic, Professional, and Similar Organizations'
data.loc[msk, 'industry'] = 'Non-Profit Organizations'

msk = data['industry'] == 'Advertising, Marketing & Research'
data.loc[msk, 'industry'] = 'Advertising'

msk = data['industry'] == 'Advertising & Public Relations'
data.loc[msk, 'industry'] = 'Advertising'

msk = data['industry'] == 'Public Relations & Advertising'
data.loc[msk, 'industry'] = 'Advertising'

msk = data['industry'] == 'OIL & GAS'
data.loc[msk, 'industry'] = 'Consumable Fuels'

msk = data['industry'] == 'Oil, Gas & Consumable Fuels'
data.loc[msk, 'industry'] = 'Consumable Fuels'

msk = data['industry'] == 'ELECTRICITY'
data.loc[msk, 'industry'] = 'Electricity'

msk = data['industry'] == 'ENERGY COMPANY'
data.loc[msk, 'industry'] = 'Energy'

msk = data['industry'] == 'SOLAR ENERGY'
data.loc[msk, 'industry'] = 'Solar Energy'

msk = data['industry'] == 'WATER COMPANY'
data.loc[msk, 'industry'] = 'Water'


# CELL_3


import numpy as np
import multiprocessing as mp

import string
import re
import spacy
import en_core_web_sm
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from normalise import normalise
from langdetect import detect

nlp = en_core_web_sm.load()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="AmE",
                 user_abbrevs={},
                 n_jobs=1):

        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        if __name__ == '__main__':
            with mp.Pool(cores) as pool:
                print(pool.map(self._preprocess_part, data_split))
                data = pd.concat(pool.map(self._preprocess_part, data_split))

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        text_f = self.non_en(text)
        text_s = self.remove_short_words(text_f)
        normalized_text = self._normalize(text_s)
        doc = nlp(normalized_text)
        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return self._lemmatize(removed_stop_words)

    def _normalize(self, text):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _remove_punct(self, doc):
        return [t for t in doc if t.text not in string.punctuation]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])
    
    def non_en(self, doc):
        if(detect(doc)=="en"):
            return doc
        else:
            return ''
    def remove_short_words(self,doc):
        text = doc.lower()
        text = re.sub(r'[^A-Za-z]', ' ', text)
        shortword = re.compile(r'\W*\b\w{1,3}\b')
        text = shortword.sub('', text)
        text = ' '.join(text.split())
        return text

#CELL_4


keyCounts = pd.DataFrame(columns= ["industry","key"])
keyCounts["industry"] = data["industry"].values
keyCounts["key"] = TextPreprocessor(n_jobs=-1).transform(data['text'])


#CELL_5

file = open("/Users/macbookretina/Desktop/veriler kopyası.csv","w")
for index, row in keyCounts.iterrows():
    file.write(row['industry']+','+row['text']+'\n')
file.close()
