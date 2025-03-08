import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf




(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
tf.keras.layers.BatchNormalization()
tf.keras.layers.Dropout(0.9)
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
tf.keras.layers.BatchNormalization()
tf.keras.layers.Dropout(0.9)
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
tf.keras.layers.BatchNormalization()
tf.keras.layers.Dropout(0.9)
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 30)

loss, acc = model.evaluate(test_images, test_labels)
print(f'Accuracy: {acc:.2f}')
print(f'Loss: {loss:.2f}')
