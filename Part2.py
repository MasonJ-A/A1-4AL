import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("datasets/training_data.csv")
dataframe = pd.DataFrame(data)
age = []
length = []
diameter = []
height = []
whole_weight = []
shucked_weight = []
viscera_weight = []
shell_weight = []
for row in dataframe.iterrows():
    age.append(row[1]["Rings"]+1.5)
    length.append(row[1]["Length"])
    height.append(row[1]["Height"])
    whole_weight.append(row[1]["Whole_weight"])
    shucked_weight.append(row[1]["Shucked_weight"])
    viscera_weight.append(row[1]["Viscera_weight"])
    shell_weight.append(row[1]['Shell_weight'])

fig, ax= plt.subplots()


#ax.scatter(age,length)#polynomial almost logorythmic
#ax.scatter(age,height)#Polynomial almost logorythmic
#ax.scatter(age,whole_weight)#slightly polynomial, kinda like a scatter shot that curves upwards
#ax.scatter(age,shucked_weight)#slightly polynomial, kinda like a scatter shot that curves upwards
#ax.scatter(age,viscera_weight)#slightly polynomial, kinda like a scatter shot that curves upwards
#ax.scatter(age, shell_weight)#slightly polynomial, kinda like a scatter shot that curves upwards
#all weights behave similarly I'm probably just gonna use the whole weight
#plt.show()

class linear_regression2():
    def  __init__(self, x_1:list, x_2:list, x_3:list, y_:list) -> None:
        self.target = np.array(y_)
        self.input = np.array([np.array(x_1),np.array(x_2),np.array(x_3)])
        self.n = len(self.target)
        self.beta = np.array([[0],[0],[0],[0]])

    def preprocess(self,):
        hmean = []
        hstd = []
        x_train = []
        X = []
        for i in self.input:
            hmean.append(np.mean(i))
            hstd.append(np.std(i))
            x_train.append((i-np.mean(i))/np.std(i))
        
        X = (np.column_stack((np.ones(len(x_train[0])),x_train[0],x_train[1],x_train[2])))
        
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd
        Y = (np.column_stack(y_train)).T
        return X, Y
    
    def predict(self, X_test,beta):
        #predict using beta

        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)

    def train(self,r,epochs,X,Y):
       
        for i in range(epochs):
            gradient = -2/self.n*X.T.dot(Y-X.dot(self.beta))
            self.beta = self.beta - r*gradient

        return self.beta
    
l_reg = linear_regression2(length,height,whole_weight,age)

X,Y = l_reg.preprocess()

beta = l_reg.train(0.1,100,X,Y)
print(beta)
