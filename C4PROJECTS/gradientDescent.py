import numpy as np
from computeCost import computeCost

def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    return theta updated
    """
    m=len(y)
    for i in range(num_iters):
        predictions = np.dot(X,theta)
        error = np.dot(X.T,predictions-y)
        descent= alpha * 1/m * error
        theta= theta-descent
        if(i%100==0):
            print('Cost function after ',i,' iterations is ', computeCost(X,y,theta)," and theta is ",theta)
    return theta
#x=np.array([1,2,3,5,7,6])
#y=3.5*x
#print(" X.shape ")
#print(x.shape)
#print(y.shape)
#gradientDescent(x,y,4,0.001,1000)