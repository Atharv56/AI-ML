from numpy.lib.histograms import histogram
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
from tflearn.layers.core import activation
stemmer = LancasterStemmer()
import numpy
import tensorflow as tf
import tflearn
import random
import json

with open("/Users/atharvsalian/Desktop/Github/AI-ML/Artificial Intelligence/Pickup Lines Helper/pickuplines.json") as file:
    data = json.load(file)

