import numpy as np
def computeCost(X,y,theta):
    """
    Take in a numpy array X,y, theta and generate the cost function using theta as parameter in a linear regression model
    """
    # np.dot(A,B) => gives dot product of A and B
    # np.power(A,n) => returns array with each element raised to the power n
    # np.sum(A) => Returns scalar with every element in A summed up
    m=len(y)
    #print(X)
    #print(y)
    #print(theta)
    predictions=np.dot(X,theta)
    square_err=(predictions-y)**2
    return 1/(2*m) * np.sum(square_err)
x=np.array([1,2,3,5,7,6,29,23,22,64])
y=np.array(3.5*x)
theta=5 # tried with 5,4,3,3.2,3.5
#print(computeCost(x,y,theta))