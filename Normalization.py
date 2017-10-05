import pandas as pd 
import numpy as np
import os


os.chdir('Data_csv')
path = os.getcwd()
os.chdir('Equal_csv')


Data_Type = 'emphasis'                                       # Change here as required for each class
f = open('b_'+Data_Type+'_Equal_Data.csv','r+')
Dataset = pd.read_csv(f)
target = Dataset['target']
target = target.values
temp = Dataset.drop('target',1)

# Removing zeros from dataset

data = []
for y in range(1,301):
    sum=0.0
    total=0
    temp_list = []
    for f in temp[str(y)]:
        if f!=0.0:
            sum+=f
            total+=1 
    Av = sum/total 
    Av = round(Av,3)       
    for g in temp[str(y)]:
        if g==0.0:
            g=Av
        temp_list.append(g)    
    data.append(temp_list)  

temp1 = np.array(data).transpose()

Max = temp1.max(axis=0)
Min = temp1.min(axis=0)
mean = temp1.mean(axis=0)
STD = temp1.std(axis=0)

print len(Max)
Row = temp1.shape[0]
Col = temp1.shape[1]
target = target.reshape((Row,1))

# Z score standardization

for j in range(Col):
    for i in range(Row):
        temp1[i,j] = (temp1[i,j]-mean[j])/STD[j]

temp1 = np.append(temp1,target,axis = 1)  

col = []
for d in range(1,301):
    col.append(str(d))

row = []    
for d in range(1,Row+1):
    row.append(d)    

col = col + ['target']
df = pd.DataFrame(temp1,index = row,columns= col)  
df['target'] = [int(s) for s in df['target']]
os.chdir(path)
os.chdir('Equal_Norm_csv')
f = open('b_'+Data_Type+'_NZE_Data.csv','w')
df.to_csv(f)
f.close()