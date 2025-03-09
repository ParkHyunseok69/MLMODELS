import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

letters_data = pd.read_csv("A_Z Handwritten Data.csv").values
X = letters_data[:, 1:].reshape(-1, 28, 28)
y = letters_data[:, 0]

X = X/255.0
y = tf.keras.utils.to_categorical(y, 26)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(26, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=16)

loss, acc = model.evaluate(X_test, y_test)
print(f'Accuracy: {acc:.2f}')
print(f'Loss: {loss:.2f}')