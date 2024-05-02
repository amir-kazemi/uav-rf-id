import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os

class Loader:
    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter=",")
        self.x = np.transpose(self.data[0:2025,:])
        self.labels = {
            "C2": self._get_label(2048),
            "C4": self._get_label(2049),
            "C10": self._get_label(2050)
        }
    
    def _get_label(self, index):
        label = np.transpose(self.data[index:index+1,:])
        return label.astype(int)

    @staticmethod
    def encode(datum):
        return to_categorical(datum)

    @staticmethod
    def decode(datum):
        y = np.zeros((datum.shape[0],1))
        for i in range(datum.shape[0]):
            y[i] = np.argmax(datum[i])
        return y
    
    

class Splitter:
    def __init__(self, class_name, data_percentage):
        self.class_name = class_name
        self.data_percentage=sorted(data_percentage, reverse=True)

    def split_and_save(self, dl, kfold_splits=5):
        
        # initial reduction of the dataset - we use 10% of the original dataset after cleaning
        Y = Loader.encode(dl.labels[self.class_name])
        Y = Loader.decode(Y)
        X = dl.x
        reducer = StratifiedKFold(n_splits=20, shuffle=True, random_state=0)
        for train, test in reducer.split(X, Y):
            x=X[test]
            y=Y[test]
            break
        
        # We keep 20% of the remaining data as our test
        # We further split the remaining 80% to get different percentages
        kfold = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=0)
        fold_counter=1
        for train, test in kfold.split(x, y):

            
            # 20 percent of reduced data as the test data set
            x_test = x[test]
            y_test = y[test]
            # 80 percent of reduced data as ground truth
            x_train = x[train]
            y_train = y[train]
            
            # We iterate over the data_percentages in reverse order
            for percentage in self.data_percentage:
                sub_folder = self._get_sub_folder(fold_counter,percentage)
                self._create_folder(sub_folder)
                
                # Adjust the number of splits based on the current percentage, with shuffle set to False
                #kfold_2 = StratifiedKFold(n_splits=int(100/percentage), shuffle=False)
                kfold_2 = StratifiedShuffleSplit(n_splits=1, test_size=percentage/100.0, random_state=0)
                for train_2, test_2 in kfold_2.split(x_train, y_train):
                    x_to_aug = x_train[test_2]
                    y_to_aug = y_train[test_2]

                    # We save the current percentage of 80% of the reduced dataset
                    self._save_data(sub_folder, x_train, y_train, x_test, y_test, x_to_aug, y_to_aug)
                    break
            
            fold_counter += 1

    def _get_sub_folder(self, counter,percentage):
        return './data/{class_name}P{data_percentage}K{fold_num}/'.format(
            class_name=self.class_name, 
            data_percentage='{:02d}'.format(percentage), 
            fold_num=str(counter)
        )

    def _create_folder(self, sub_folder):
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

    def _save_data(self, sub_folder, x_train, y_train, x_test, y_test, x_to_aug, y_to_aug):
        np.save(sub_folder+'X_train.npy', x_train)
        np.save(sub_folder+'Y_train.npy', np.squeeze(y_train))
        np.save(sub_folder+'X_test.npy', x_test)
        np.save(sub_folder+'Y_test.npy', np.squeeze(y_test))
        np.save(sub_folder+'X_train_subsampled.npy', x_to_aug)
        np.save(sub_folder+'Y_train_subsampled.npy', np.squeeze(y_to_aug))
        