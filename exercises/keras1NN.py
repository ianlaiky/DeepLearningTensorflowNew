from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
# step 1 load your data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels


L1 = 1024
L2 = 512
L3 = 256
L4 = 128
L5 = 64
L6 = 32
L7 = 20
L8 = 15
L9 = 13


# Step 2: Define the Model
model = Sequential()
model.add(Dense(L1,input_dim=784,activation='relu'))
model.add(Dense(L2,activation='relu'))
model.add(Dense(L3,activation='relu'))
model.add(Dense(L4,activation='relu'))
model.add(Dense(L5,activation='relu'))
model.add(Dense(L6,activation='relu'))
model.add(Dense(L7,activation='relu'))
model.add(Dense(L8,activation='relu'))
model.add(Dense(L9,activation='relu'))
model.add(Dense(10,activation='softmax'))
print(model.summary())




# step 3 compile your model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



# step 4 train your model

model.fit(X_train,y_train,epochs=2)


# step 5 evaluate your model
loss,acc = model.evaluate(X_test,y_test)
print("Accuracy = ",acc)
print("Loss = ",loss)

# step 6 save your model

