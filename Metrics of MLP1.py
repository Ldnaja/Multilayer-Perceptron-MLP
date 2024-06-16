# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt

data_path = '' #Include the dataset path mammographic_masses.data

# Handle missing values by replacing them with the median of each column
data = pd.read_csv(data_path, header=None, na_values="?")
data.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
data.fillna(data.median(), inplace=True)

# Select relevant features and target variable
X = data[['Age', 'Shape', 'Margin', 'Density']]
y = data['Severity']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create the MLP model
def create_model(input_dim, neurons, dropout=0.0, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation='relu', input_dim=input_dim, kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Configurations to test
configs = [
    {'neurons': 10, 'epochs': 100, 'dropout': 0.0, 'learning_rate': 0.001, 'label': 'No Dropout, 100 Epoch, 10 neuron'},
    {'neurons': 10, 'epochs': 200, 'dropout': 0.5, 'learning_rate': 0.001, 'label': 'Dropout 0.5, 200 epochs, 10 neuron'},
    {'neurons': 15, 'epochs': 150, 'dropout': 0.4, 'learning_rate': 0.1, 'label': 'Dropout 0.4, 150 epochs, 15 neuron, lr=0.1'},
    {'neurons': 10, 'epochs': 100, 'dropout': 0.3, 'learning_rate': 0.01, 'label': 'Dropout 0.3, 100 epochs, 10 neuron, lr=0.01'},
    {'neurons': 20, 'epochs': 200, 'dropout': 0.2, 'learning_rate': 0.0001, 'label': 'Dropout 0.2, 200 epochs, 25 neuron, lr=0.0001'}
]

results = []

# Train and evaluate the model using 5-fold cross-validation for each configuration
for config in configs:
    accuracies, precisions, recalls, f1s, specificities = [], [], [], [], []
    histories = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = create_model(input_dim=X_train.shape[1],
                             neurons=config['neurons'],
                             dropout=config['dropout'],
                             learning_rate=config['learning_rate'])

        history = model.fit(X_train, y_train, epochs=config['epochs'], verbose=0, validation_data=(X_test, y_test))
        histories.append(history)

        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        specificities.append(specificity)

    # Store the results for each configuration
    results.append({
        'config': config['label'],
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1_score': np.mean(f1s),
        'specificity': np.mean(specificities),
        'histories': histories
    })

# Print the average performance metrics for each configuration
for result in results:
    print(f"Configuration: {result['config']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1_score']:.4f}")
    print(f"Specificity: {result['specificity']:.4f}\n")

# Plot accuracy convergence curves for each configuration
plt.figure(figsize=(12, 8))
for result in results:
    for i, history in enumerate(result['histories']):
        plt.plot(history.history['accuracy'], label=f"{result['config']} - Fold {i+1}")

plt.title('Model accuracy convergence for different configurations')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Plot bar charts for performance metrics comparison
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
plt.figure(figsize=(16, 12))

for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    values = [result[metric] for result in results]
    labels = [result['config'] for result in results]
    bars = plt.barh(labels, values, color='skyblue')
    plt.xlabel(metric.capitalize())
    plt.title(f'Comparison of {metric.capitalize()}')

    # Add numbers to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center')

plt.tight_layout()
plt.show()
