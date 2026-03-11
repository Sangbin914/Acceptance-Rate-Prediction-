import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('gpascore.csv')
data = data.dropna()
x_train = []
y_train = data['admit'].values
for i, rows in data.iterrows():
    x_train.append([rows['gre'], rows['gpa'], rows['rank']])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), epochs=2000)

perdict = model.predict([[380, 3.61, 2], [660, 3.67, 2], [800, 4.0, 1]])
print(perdict)
