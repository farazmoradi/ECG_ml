import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import hydra
from plot_tools import plot_results
import mlflow
import mlflow.keras
import optuna
import argparse
import pickle
import numpy as np



# loading pickle file
with open('../../data/processed/01_data_stft.pkl', "rb") as open_file:
    loaded_list = pickle.load(open_file)

# Access learning_rate and batch_size
learning_rate_op = [0.001, 0.002]
batch_size_op = [128,256]
n_trial_op = 1
nepochs    = 2

print(learning_rate_op)

# extracting test and train
X_train = loaded_list['X_train']
X_test = loaded_list['X_test']
y_train = loaded_list['y_train']
y_test = loaded_list['y_test']


# Convert y_train to integer type
y_train = y_train.astype(int)

class_counts = np.bincount(y_train)

for class_label, count in enumerate(class_counts):
    if class_label==0:
        t=count
    print(f"Class {class_label} has {count} trials")
    if class_label!=0:
        print(f"Class {class_label} fraction to class 0 is {t/count}")
        

X_train, X_val, y_train, y_val = train_test_split(X_train, to_categorical(y_train), test_size=0.2, random_state=42,shuffle=True,stratify=to_categorical(y_train))
print(X_train.shape)

num_classes = 5


def create_model(trial):
    
    
    num_classes=5
    
    model = Sequential([
        Conv2D(16,(4,4),activation='relu',input_shape=X_train.shape[1:]),
        MaxPooling2D((2, 2)),
        Conv2D(32, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])


    return model

# Create a subclass of tf.keras.callbacks.Callback to log metrics in mlflow
class MlflowCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(logs, step=epoch)

# Define objective function for Optuna
def objective(trial):
    with mlflow.start_run():
        
        # Define hyperparameters to be optimized
        learning_rate = trial.suggest_loguniform('learning_rate', learning_rate_op[0], learning_rate_op[1])
        batch_size = trial.suggest_categorical('batch_size', batch_size_op)
        
        # Create, train, and evaluate model
        model = create_model(trial)
        
        model.compile(optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])   
    
        
                # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=3)

        # Train the model with early stopping
        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val),
                            epochs=nepochs, 
                            batch_size=batch_size,
                            callbacks=[early_stopping])

        # Evaluate our model
        loss, acc = model.evaluate(X_test, to_categorical(y_test), verbose=2)
        print('Test accuracy:', acc)
        
        plot_results(history,X_test,model,num_classes,y_test)
        
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Save the model
        mlflow.keras.log_model(model, "model")
        

    return acc  # Return accuracy as the metric to be optimized

    

# Create Optuna study and optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trial_op)  # You can increase n_trials for a more exhaustive search

# Print Optuna's best parameters
print("Optimizer: ", study.best_params)

