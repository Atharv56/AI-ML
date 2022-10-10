import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
data = pd.read_csv("D:\Code\Python ML\Linear-Regression\student-mat.csv", sep = ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "traveltime", "famrel", "goout", "Dalc", "Walc", "health"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
linear = linear_model.LinearRegression()


with open("studentmodel.pickle", 'wb') as f:
    pickle.dump(linear, f)



pickle_in = open("D:\Code\Python ML\Linear-Regression\studentmodel.pickle", "rb")

linear = pickle.load(pickle_in)
acc = linear.score(x_test, y_test)
print(acc)

print("Co: ",  linear.coef_)
print('Intercept: ' , linear.intercept_)
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p ="G1"
style.use("ggplot")
plt.plot(predictions, 'o')
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show() 