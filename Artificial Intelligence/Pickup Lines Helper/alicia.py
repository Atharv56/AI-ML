from cProfile import label
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

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs_x.append(word)
        docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)
training = []
output  = []
out_empty = [0 for _ in range (len(labels))]

for x,doc in enumerate(docs_x):
    bag = []
    word = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in word:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(training, output, n_epoch = 500, batch_size = 8, show_metric = True)
model.save('Pickup Lines/model.tflearn')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

        return numpy.array(bag)

def chat():
    print("You like this woman so be confident. I am here to help you communicate")
    print("Where did you meet this beautiful woman")
    while True:
        inp = input("Location: ")
        if inp.lower() == 'quit' or inp.lower() == 'she left':
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        for tg in data['intents']:
            if tg['tag'] == tag:
                response = tg['responses']
        print(random.choice(response))

chat()


