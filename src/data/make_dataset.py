import pandas as pd
import pickle

train_data = pd.read_csv('data/raw/mitbih_train.csv',header=None)
test_data  = pd.read_csv('data/raw/mitbih_test.csv',header=None)

X_train = train_data.iloc[:, :-1].values
X_test = test_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

print('Number of training examples: ',len(train_data))
print('Number of test examples: ',len(test_data))


# write to a pickle file
with open("data/interim/01_data_laoded.pkl", "wb") as file:
    pickle.dump({'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}, file)
    
