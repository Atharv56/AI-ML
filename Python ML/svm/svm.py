import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
#svm support vector machine
cancer = datasets.load_breast_cancer()


x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

classes = ['malignant', 'benign']
#SVC support vector classification
clf = svm.SVC(kernel ='linear', C = 2)
clf.fit(x_train, y_train)
y_prediction = clf.predict(x_test)
print(y_prediction)
acc = metrics.accuracy_score(y_test, y_prediction)
print(acc)
