from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

def view_loss_acc(epoch_num, history):
   loss = history.history["loss"]
   acc = history.history["acc"]
   val_loss = history.history["val_loss"]
   val_acc = history.history["val_acc"]

   print(acc)
   print(val_acc)

   import matplotlib.pyplot as plt

   plt.plot([i for i in range(epoch_num)], acc)  # training loss
   plt.plot([i for i in range(epoch_num)], val_acc)  # validate loss
   plt.title("Acc")
   plt.legend(["Train", "Val"])
   plt.show()

   plt.plot([i for i in range(epoch_num)], loss)  # training loss
   plt.plot([i for i in range(epoch_num)], val_loss)  # validate loss
   plt.title("Loss")
   plt.legend(["Train", "Val"])
   plt.show()
   return

def view_wrong_case(model):
   predictions = model.predict(X_test[0:100])

   import numpy as np

   wrong_pred = np.argmin(np.argmax(predictions, axis=1) == y_test[0:100])
   wrong_pred

   element = wrong_pred
   #plt.imshow(X_test[element].reshape(28,28))
   #plt.show()
   print("Label for the element", element, ":", y_test[element])
   print("Prediction for the element:",  element, ":", np.argmax(predictions[element]))
   return

def network(n_input, epoch_num, X_train, y_train):
   num_hidden1 = 128
   num_hidden2 = 64
   num_hidden3 = 32
   num_hidden4 = 16
   n_output = 10  # 0,1,2, ..., 9 -- number charactors

   dropout = 0.5

   model = Sequential([
      Dense(num_hidden1, activation='relu', input_shape=(n_input,)),
      Dense(num_hidden2, activation='relu'),
      Dropout(rate=dropout),
      Dense(num_hidden3, activation='relu'),
      Dense(num_hidden4, activation='relu'),
      Dense(n_output, activation='softmax')
   ])

   model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

   print(model.summary())
   batch_size = 64
   history = model.fit(
      x=X_train,
      y=to_categorical(y_train),
      epochs=epoch_num,
      batch_size=batch_size,
      shuffle=True,
      validation_split=0.2
   )
   return model, history

def load_data():
   (X_train, y_train), (X_test, y_test) = mnist.load_data()

   n_input = 28 * 28 # input feature (28x28 pixels)

   X_train = X_train.reshape((-1, n_input))
   X_test = X_test.reshape((-1, n_input))

   return X_train, y_train, X_test, y_test, n_input


if __name__ == "__main__":

   X_train, y_train, X_test, y_test, n_input = load_data()

   epoch_num = 20

   model, history = network(n_input, epoch_num, X_train, y_train)

   view_loss_acc(epoch_num, history)

   eval = model.evaluate(X_test, to_categorical(y_test))
   print("Test Loss:", eval[0])
   print("Test Accuracy:", eval[1])

   model.save(r'SavedModels/hwr.model')


