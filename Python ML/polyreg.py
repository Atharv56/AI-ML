#Method to predict the future for a scattered data
from sklearn.metrics import r2_score
import numpy
import matplotlib.pyplot as plt
#Used when linear regression will not fit
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1,25,100)
print(r2_score(y, mymodel(x)))
speed = mymodel(17) #Predicts the speed at 5 p.m.
print(speed)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
"""R-Squared: value ranges from 0 to 1, where 0 means no
    relationship, and 1 means 100% related"""