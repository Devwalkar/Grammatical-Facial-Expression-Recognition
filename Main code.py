import tensorflow as tf
import numpy as np
from numpy import array
import pandas as pd
from sklearn.model_selection import train_test_split
import os

#######################################################################################################
# Network Parameters
test_train_split = 0.2
RANDOM_SEED = 128
STD_DEV = 0.14
epoch = 500
initial_learning_rate = 0.001
beta = 0.05
decay_step = 7000
decay_ratio = 0.9

#####################################################################################################
# Dataset initialization

os.chdir('Data_csv')
os.chdir('Equal_Norm_csv')
f = open('a_relative_NZE_Data.csv','r+')
Dataset = pd.read_csv(f)
target = Dataset['target']
target = target.values
temp = Dataset.drop('target',1)
data = temp.as_matrix()
data = np.delete(data,0,1)

for u in range(len(data)):
    for c in range(300):
       data[u][c] = data[u][c]*100

# OUTPUT labels generation
Label = []
for o in range(len(target)):
    if target[o]==0:
        Label.append([1.0,0.0])
    if target[o]==1:
        Label.append([0.0,1.0])                
Label = np.array(Label)  

# Train test data split
train_X, test_X, train_y, test_y = train_test_split(data,Label, test_size=test_train_split, random_state=RANDOM_SEED)


x_size = train_X.shape[1]               # Number of input nodes: 300 features and 1 bias
no_samples = train_X.shape[0]           # number of train samples

##########################################################################################
# INPUT OUTPUT VARIABLES

X = tf.placeholder("float", [None, x_size])
Y = tf.placeholder("float", [None, 2])

############################################################################################
# Intialization and propagation functions
initializer = tf.contrib.layers.xavier_initializer(seed = RANDOM_SEED)


def weight_initialize_first_layer():
    #D = tf.Variable(tf.random_normal([3,1],stddev = (float(2)/4)**0.5))
    D = tf.Variable(initializer((2,1)))
    #D = tf.get_variable("D1",shape = (2,1),initializer = initializer,regularizer = regularizer)
    return D

def weight_initialize_second_layer(Shape):
    #D = tf.Variable(tf.random_normal(Shape,stddev = (float(2)/sum(Shape))**0.5))
    D = tf.Variable(initializer(Shape))
    #D = tf.get_variable("D2",shape = Shape,initializer = initializer,regularizer = regularizer)
    return D

def weight_initialize_third_layer():
    #D = tf.Variable(tf.random_normal([10,10],stddev = (float(2)/ 20)**0.5))
    D = tf.Variable(initializer((10,10)))
    #D = tf.get_variable("D3",shape = (10,10),initializer = initializer,regularizer = regularizer)
    return D

def weight_initialize_output_layer():
    #D = tf.Variable(tf.random_normal([10,2],stddev = (float(2)/12)**0.5))
    D = tf.Variable(initializer((10,2)))
    #D = tf.get_variable("D4",shape = (10,2),initializer = initializer,regularizer = regularizer)
    return D

def bais(shape):
    #E = tf.Variable(tf.random_normal(shape,stddev = STD_DEV))
    E = tf.Variable(initializer(shape))
    #E = tf.get_variable("E",shape = shape,initializer = initializer,regularizer = regularizer)
    return E 

def forwardprop12(X,Y):
    #D = tf.Variable(tf.random_normal([1,1],stddev = (float(2)/4)**0.5))
    M = tf.add(tf.nn.tanh(tf.matmul(X,Y)),bais([1,1]))
    return M

def forwardprop3(X,Y):
    #D = tf.Variable(tf.random_normal([1,2],stddev = (float(2)/12)**0.5))
    M = tf.add(tf.nn.tanh(tf.matmul(X,Y)),bais([1,2]))
    return M    

################################################################################################
#FIRST LAYER

first_layer = []
first_layer_weights = []
for i in range(0,300,3):
    Z = weight_initialize_first_layer()
    M = forwardprop12(tf.reshape(X[0,i:i+2],(1,2)),Z)
    first_layer.append(M)
    first_layer_weights.append(Z)

##################################################################################################
#SECOND LAYER

second_layer = []
second_layer_weights = []

#LEFT_EYE
A1 = weight_initialize_second_layer([8,1])
B1 = forwardprop12(tf.reshape(first_layer[0:8],(1,8)),A1)
second_layer.append(B1)
second_layer_weights = tf.reshape(A1,(1,8))

#RIGHT_EYE
A2 = weight_initialize_second_layer([8,1])
B2 = forwardprop12(tf.reshape(first_layer[8:16],(1,8)),A2)
second_layer.append(B2)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A2,(1,8)))],axis=1)

#LEFT_EYEBROW
A3 = weight_initialize_second_layer([10,1])
B3 = forwardprop12(tf.reshape(first_layer[16:26],(1,10)),A3)
second_layer.append(B3)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A3,(1,10)))],axis=1)

#RIGHT_EYEBROW
A4 = weight_initialize_second_layer([10,1])
B4 = forwardprop12(tf.reshape(first_layer[26:36],(1,10)),A4)
second_layer.append(B4)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A4,(1,10)))],axis=1)

#NOSE
A5 = weight_initialize_second_layer([12,1])
B5 = forwardprop12(tf.reshape(first_layer[36:48],(1,12)),A5)
second_layer.append(B5)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A5,(1,12)))],axis=1)

#MOUTH
A6 = weight_initialize_second_layer([20,1])
B6 = forwardprop12(tf.reshape(first_layer[48:68],(1,20)),A6)
second_layer.append(B6)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A6,(1,20)))],axis=1)

#FACE_CONTOUR
A7 = weight_initialize_second_layer([19,1])
B7 = forwardprop12(tf.reshape(first_layer[68:87],(1,19)),A7)
second_layer.append(B7)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A7,(1,19)))],axis=1)

#LEFT+RIGHT+IRIS+NOSE_TIP
A8 = weight_initialize_second_layer([3,1])
B8 = forwardprop12(tf.reshape(first_layer[87:90],(1,3)),A8)
second_layer.append(B8)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A8,(1,3)))],axis=1)

#line above left eyebrow
A9 = weight_initialize_second_layer([5,1])
B9 = forwardprop12(tf.reshape(first_layer[90:95],(1,5)),A9)
second_layer.append(B9)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A9,(1,5)))],axis=1)

#line above right eyebrow
A10 = weight_initialize_second_layer([5,1])
B10 = forwardprop12(tf.reshape(first_layer[95:],(1,5)),A10)
second_layer.append(B10)
second_layer_weights = tf.concat([second_layer_weights,(tf.reshape(A10,(1,5)))],axis=1)

############################################################################################
# Third layer

#C1 = weight_initialize_third_layer()
#third_layer = forwardprop3(tf.reshape(second_layer,(1,10)),C1)
#third_layer_weights = tf.reshape(C1,(1,100))
#third_layer = tf.nn.dropout(third_layer,0.8,seed = RANDOM_SEED)

############################################################################################
#OUTPUT_LAYER

OUT = weight_initialize_output_layer()
OUTPUT = forwardprop3(tf.reshape(second_layer,(1,10)),OUT)
OUT_weights = tf.reshape(OUT,[1,20])

###########################################################################################
# Regularization

weights = tf.concat([tf.reshape(first_layer_weights,(1,200)),second_layer_weights],axis=1)
#weights = tf.concat([weights,third_layer_weights],axis = 1)
weights = tf.concat([weights,OUT_weights],axis=1)
'''
tf.add_to_collection("weights",tf.reshape(first_layer_weights,[200]))
tf.add_to_collection("weights",tf.reshape(second_layer_weights,[100]))
tf.add_to_collection("weights",OUT_weights)
'''
regularizer = tf.nn.l2_loss(weights)
############################################################################################
#COST_FUNCTION

global_step = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,decay_step,decay_ratio)

Loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits= OUTPUT) + beta*regularizer)

updates = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(Loss,global_step=global_step)

##########################################################################################
#TRAINING_MODEL

sess =  tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

for e in range(epoch):
    for r in range(no_samples):
     [_,L] = sess.run([updates,Loss] ,feed_dict={X:train_X[r].reshape((1,300)), Y:train_y[r].reshape((1,2))})
    print 'Epoch:%d  Loss:%.3f'%(e+1,L)  


#############################################################################################
# Testing for training set

sum = 0
for t in range(len(train_X)):
    J = sess.run(OUTPUT, feed_dict={X:train_X[t].reshape((1,300)), Y:train_y[t].reshape((1,2))})
    print J,train_y[t]
    if (J[0,0]>J[0,1]) & (train_y[t][0]>train_y[t][1]):
        sum+=1
    if (J[0,0]<J[0,1]) & (train_y[t][0]<train_y[t][1]):
        sum+=1      


# Testing for test set

TP = 0
FN = 0
TN = 0
FP = 0

for t in range(len(test_X)):
    J = sess.run(OUTPUT, feed_dict={X:test_X[t].reshape((1,300)), Y:test_y[t].reshape((1,2))})
    print J,test_y[t]
    if (J[0,0]>J[0,1]): 
       if (test_y[t][0]>test_y[t][1]):
         TN+=1
       else:
         FN+=1    
    if (J[0,0]<J[0,1]): 
        if (test_y[t][0]<test_y[t][1]):
         TP+=1
        else:
         FP+=1 

Precision = float(TP)/(TP+FP)
Recall = float(TP)/(TP+FN)
F_score = (2*(Precision*Recall))/(Precision+Recall)

print 'Test set Accuracy: %.3f  Total: %d' %(float(TP+TN)/len(test_X)*100,len(test_X))
print 'Training set Accuracy: %.3f   Total:%d' %(float(sum)/len(train_X)*100,len(train_X)) 
print 'F score:%.3f'%(F_score)
print 'Precision:%.3f'%(Precision)
print 'Recall:%.3f'%(Recall)
print 'False negatives:%d'%(FN)
sess.close()    
