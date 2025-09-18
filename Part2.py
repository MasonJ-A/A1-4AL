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
plt.show()


