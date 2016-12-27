import io
import os
import random
import pickle
import urllib.request

from zipfile import ZipFile
from collections import Counter, defaultdict
from nltk import classify, NaiveBayesClassifier


class GenderPredictor():
    def __init__(self):
        counts = Counter()
        self.feature_set = []
        self.url = ('https://github.com/downloads/clintval/'
                    'gender_predictor/names.zip')

        for name_results in self._get_USSSA_data:
            name, male_counts, female_counts = name_results

            if male_counts == female_counts:
                continue

            features = self._name_features(name)
            gender = 'M' if male_counts > female_counts else 'F'
            counts.update([gender])

            m_prob = male_counts / sum([male_counts, female_counts])
            m_prob = 0.01 if m_prob == 0 else 0.99 if m_prob == 1 else m_prob

            features['m_prob'] = m_prob
            features['f_prob'] = 1 - m_prob
            self.feature_set.append((features, gender))

        print('{M:,} male names\n{F:,} female names'.format(**counts))

    def classify(self, name):
        return(self.classifier.classify(self._name_features(name.upper())))

    def train_and_test(self, percent_to_train=0.80):
        random.shuffle(self.feature_set)
        partition = int(len(self.feature_set) * percent_to_train)
        train = self.feature_set[:partition]
        test = self.feature_set[partition:]

        self.classifier = NaiveBayesClassifier.train(train)
        print("classifier accuracy: {:0.2%}".format(
            classify.accuracy(self.classifier, test)))

    def _name_features(self, name):
        return({
            'last_is_vowel': (name[-1] in 'AEIOUY'),
            'last_letter': name[-1],
            'last_three': name[-3:],
            'last_two': name[-2:]})

    @property
    def _get_USSSA_data(self):
        path = './gp_data/'

        if os.path.isdir(path) is False:
            os.makedirs(path)

        if os.path.exists(path + 'names.pickle') is False:
            names = defaultdict(lambda: {'M': 0, 'F': 0})
            print('names.pickle does not exist... creating')

            if os.path.exists(path + 'names.zip') is False:
                print('names.zip does not exist... downloading')
                urllib.request.urlretrieve(self.url, path + 'names.zip')

            with ZipFile(path + 'names.zip') as infiles:
                for filename in infiles.namelist():
                    with io.TextIOWrapper(infiles.open(filename)) as infile:
                        for row in infile:
                            name, gender, count = row.strip().split(',')
                            names[name.upper()][gender] += int(count)

            data = [(n, names[n]['M'], names[n]['F']) for n in names]

            with open(path + 'names.pickle', 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
                print('names.pickle saved')
        else:
            with open(path + 'names.pickle', 'rb') as handle:
                data = pickle.load(handle)
                print('import complete')
        return(data)
