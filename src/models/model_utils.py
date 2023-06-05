from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV

def train_rf_with_pca(data, label, PCA_T=False, nPCA=4,PCA_VAR_plot=True,gridsearch=True):
    # Apply PCA if required
    if PCA_T:
        pca = PCA(n_components=nPCA)
        data = pca.fit_transform(data)
  
    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=42,shuffle=True, stratify=label)
    
    # Instantiate and train the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if gridsearch:
        # Define the parameter grid
        param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [2, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
        # 'bootstrap': [True, False]
        }
        
        # Create a base model
        rf_gr = RandomForestClassifier(random_state=42)

        # Instantiate the grid search model
        rf = GridSearchCV(estimator=rf_gr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        # You can get the best parameters like this
        
        print("Grid serach has started")

    
    rf.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_val_pred = rf.predict(X_val)
  
    # Get feature importances
    if PCA_VAR_plot:
      
        # Print the explained variance ratio of the PCA components
        print('Explained variance ratio: ', pca.explained_variance_ratio_)

        # Plot the explained variance
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.show()

    
    acc = predicted_results(y_val,y_val_pred)
  
    return rf, acc

def test_rf_with_PCA(rf, data, label, PCA_T=False,nPCA=2,PCA_VAR_plot=False):
    
    if PCA_T:
        pca = PCA(n_components=nPCA)
        data = pca.fit_transform(data)
    
    y_pred = rf.predict(data)
    acc = predicted_results(label,y_pred)
    
        # Get feature importances
    if PCA_VAR_plot:
      
        # Print the explained variance ratio of the PCA components
        print('Explained variance ratio: ', pca.explained_variance_ratio_)

        # Plot the explained variance
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.show()

    return acc


def predicted_results(y,y_pred):
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred,average='macro')
    recall = recall_score(y, y_pred,average='macro')  # Sensitivity is also known as recall
    precision = precision_score(y, y_pred,average='macro')
        
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Create a confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Use seaborn to plot the confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    plt.title('Confusion matrix without normalization')

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')

    plt.show()
    
    return accuracy