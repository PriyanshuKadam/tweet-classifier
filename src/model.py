import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers

def predict(X, y, X_test):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1]

    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=6, batch_size=512, validation_data=(X_val, y_val)) 

    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    return predicted_classes