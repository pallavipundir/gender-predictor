#gender_predictor

This package is a Python 3 port of Stephen Holiday's [genderPredictor](https://github.com/sholiday/genderPredictor) `nltk.NaiveBayesClassifier` wrapper. It has been rewritten mostly for personal use and is now used in a variety of unreleased classification projects of mine. More details to come on how I have used it...

Data was downloaded from the [U.S. Social Security](https://www.ssa.gov/oact/babynames/limits.html) website portal for *Beyond the Top 100 Names*. Names included are U.S. nationwide baby names from 2015 with at least five occurences.

To install this package:

```bash
pip install git+git://github.com/clintval/gender_predictor.git
```

Usage is simple. For first time the class is instantiated in the current working directory, the raw data will need to be downloaded and pickled. Only the file `names.pickle` is needed.

```python
>>> from gender_predictor import GenderPredictor

>>> gp = GenderPredictor()
>>> gp.train_and_test()

"names.pickle does not exist... creating"
"names.zip does not exist... downloading"
"names.pickle saved"
"32,031 male names"
"56,347 female names"
"classifier accuracy: 96.96%"
```

To classify a name:

```python
>>> gp.classify('Aldo')

"M"
```

```python
>>> gp.classify('Kaylee')

"F"
```

To change the directory `gender_predictor` uses for data storage you may reset `gender.predictor.PATH` before instantiating the classifier class.

```python
>>> import gender_predictor

>>> print(gender_predictor.PATH)

"./gender_prediction/"

>>> gender_predictor.PATH = '~/data/project01/'
>>> print(gender_predictor.PATH)

"~/data/project01/"
```
