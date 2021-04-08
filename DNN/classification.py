import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
#% matplotlib inline

import seaborn as sns

sns.set(style="darkgrid")

print(tf.__version__)
cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety','output']
cars = pd.read_csv(r'Data/car_evaluation.csv', names=cols, header=None)

#cars.head()

#plot_size = plt.rcParams["figure.figsize"]
#plot_size [0] = 8
#plot_size [1] = 6
#plt.rcParams["figure.figsize"] = plot_size

#cars.output.value_counts().plot(kind='pie', autopct='%0.05f%%', colors=['lightblue', 'lightgreen', 'orange', 'pink'], explode=(0.05, 0.05, 0.05,0.05))

price = pd.get_dummies(cars.price, prefix='price')
maint = pd.get_dummies(cars.maint, prefix='maint')

doors = pd.get_dummies(cars.doors, prefix='doors')
persons = pd.get_dummies(cars.persons, prefix='persons')

lug_capacity = pd.get_dummies(cars.lug_capacity, prefix='lug_capacity')
safety = pd.get_dummies(cars.safety, prefix='safety')

labels = pd.get_dummies(cars.output, prefix='condition')

X = pd.concat([price, maint, doors, persons, lug_capacity, safety] , axis=1)

labels.head()

y = labels.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
print("X.shape[1]", X.shape[1])
print("y.shape[1]", y.shape[1])
input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

epoch_num = 10

history = model.fit(X_train, y_train, batch_size=8, epochs=epoch_num, verbose=1, validation_split=0.2)

loss = history.history["loss"]
acc = history.history["acc"]
val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]

print(acc)
print(val_acc)
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

#plt.show()

plt.plot([i for i in range(epoch_num)], acc)        # training loss
plt.plot([i for i in range(epoch_num)], val_acc)        # validate loss
plt.title("Acc")
plt.legend(["Train", "Val"])
plt.show()

plt.plot([i for i in range(epoch_num)], loss)        # training loss
plt.plot([i for i in range(epoch_num)], val_loss)        # validate loss
plt.title("Loss")
plt.legend(["Train", "Val"])
plt.show()