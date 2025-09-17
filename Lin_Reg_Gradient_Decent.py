import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("datasets/gdp-vs-happiness.csv")

by_year = (data[data['Year']==2018]).drop(columns=["World regions according to OWID","Code"])
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2021 international $)']).notna()]
happiness=[]
gdp=[]
for row in df.iterrows():
    if row[1]['Cantril ladder score']>4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2021 international $)'])

class gradient_decent():
    
    def  __init__(self, x_:list, y_:list) -> None:
        self.input = np.array(y_)
        self.target = np.array(x_)
        self.n = len(self.target)
        self.beta = np.array([[0],[0]])
    def preprocess(self,):

        #normalize the values
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean)/hstd

        #arrange in matrix format
        X = np.column_stack((np.ones(len(x_train)),x_train))

        #normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd

        #arrange in matrix format
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
    

GD =gradient_decent(happiness,gdp)
X,Y = GD.preprocess()
epochs = [10,50,100,500,1000]
rates = [0.0001,0.002,0.01,0.0005,0.4]
betas=[]
Y_predicts = []
for i in range(5):
    betas.append(GD.train(rates[i],epochs[i] ,X,Y))
    Y_predicts.append(GD.predict(X,betas[i]))

X_ = X[...,1].ravel()



#set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(X_,Y)

#display the line predicted by beta and X
ax.plot(X_,Y_predicts[0],color='r')
ax.plot(X_,Y_predicts[1],color='b')
ax.plot(X_,Y_predicts[2],color='g')
ax.plot(X_,Y_predicts[3],color='y')
ax.plot(X_,Y_predicts[4],color='c')

#set the x-labels
ax.set_xlabel("Happiness")

#set the x-labels
ax.set_ylabel("GDP per capita")

#set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

#show the plot
plt.show()