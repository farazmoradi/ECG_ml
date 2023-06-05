import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import seaborn as sns

def plot_results(history,X_test,model,num_classes,y_true,fig_label):
    


    # Plot the training/validation accuracy and loss curves
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('reports/figures/plt_loss_{}.jpg'.format(fig_label),dpi=200)

    

    # Compute predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes,average='macro')
    recall = recall_score(y_true, y_pred_classes,average='macro')  # Sensitivity is also known as recall
    precision = precision_score(y_true, y_pred_classes,average='macro')
        
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.savefig('reports/figures/confusion_matrix_{}.jpg'.format(fig_label),dpi=200)

    

    return