import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the train dataset
df_Arrhythmia_train = pd.read_csv("dataset/mitbih_train.csv")
df_Arrhythmia_train.columns = range(df_Arrhythmia_train.shape[1])
df_Arrhythmia_train.rename(columns={187: 'Labels'}, inplace=True)

# Load the test dataset
df_Arrhythmia_test = pd.read_csv("dataset/mitbih_test.csv")
df_Arrhythmia_test.columns = range(df_Arrhythmia_test.shape[1])
df_Arrhythmia_test.rename(columns={187: 'Labels'}, inplace=True)


# Set x and y in train data
X = df_Arrhythmia_train.drop(columns=['Labels'])
y = df_Arrhythmia_train['Labels']

# SMOTE-Tomek for upsampling train data
smote_tomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Set x and y in test data
X_1 = df_Arrhythmia_test.drop(columns=['Labels'])
y_1 = df_Arrhythmia_test['Labels']

# SMOTE-Tomek for upsampling test data
smote_tomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
X_resampled_test, y_resampled_test = smote_tomek.fit_resample(X_1, y_1)

# Create resampled dataframe
df_resampled = pd.DataFrame(X_resampled)
df_resampled_test = pd.DataFrame(X_resampled_test)

# Set X-train and y-train
X_train = df_resampled.drop(columns=['Labels'])
y_train = df_resampled['Labels']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Set X-test and y-test
X_test = df_resampled_test.drop(columns=['Labels'])
y_test = df_resampled_test['Labels']

# Initialize OneHotEncoder
encoder_rnn = OneHotEncoder(sparse=False)

# Reshape y_test to a 2D array (required by OneHotEncoder)
y_test_reshaped_RNN = y_test.to_numpy().reshape(-1, 1)

# Fit and transform y_test to one-hot encoded format
y_test_onehot_RNN = encoder_rnn.fit_transform(y_test_reshaped_RNN)

# Convert one-hot encoded labels to single integer labels for y_test
y_test_classes_RNN = np.argmax(y_test_onehot_RNN, axis=1)

# Initialize the StandardScaler & applied StandardScaler
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_valid_normalized = scaler.transform(X_valid)
X_test_normalized = scaler.transform(X_test)

# Define the LSTM model
RNN_model = Sequential([
    LSTM(64, input_shape=(187,1), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(128, activation='relu'),
    Dropout(0.1),  # Adding dropout for regularization
    Dense(5, activation='softmax')
])

# Compile the model
RNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
RNN_model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history_RNN = RNN_model.fit(X_train_normalized.reshape(-1, 187, 1), y_train, epochs=150, batch_size=128, validation_data=(X_valid_normalized, y_valid), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = RNN_model.evaluate(X_valid_normalized, y_valid)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Extract accuracy and loss data from the training history
history_dict = history_RNN.history

# Create subplots: 1 row, 2 columns
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the training and validation loss
ax[0].plot(history_dict['loss'], label='Training Loss', color='blue')
ax[0].plot(history_dict['val_loss'], label='Validation Loss', color='red')
ax[0].set_title('Training and Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot the training and validation accuracy
ax[1].plot(history_dict['accuracy'], label='Training Accuracy', color='blue')
ax[1].plot(history_dict['val_accuracy'], label='Validation Accuracy', color='red')
ax[1].set_title('Training and Validation Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

# Show the plots
plt.tight_layout()
plt.show()

# Define a function to display the confusion matrix and classification report
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


# Predict on test data and print the confusion matrix
Y_pred = RNN_model.predict(X_test_normalized)
y_pred = np.argmax(Y_pred, axis=1)
print_confusion_matrix(y_test_classes_RNN, y_pred)




