import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


class ClassificationMetrics:
    def __init__(self, n_runs=5):
        self.n_runs = n_runs

    def classification(self, X_train, Y_train, X_test, Y_test):
        
        # Step 1: Split your original training data into training and validation sets
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, Y_train, test_size=0.2)

        # Step 2: Apply SMOTE to only the training part
        k_neighbors = 5
        smote_done = False
        while not smote_done:
            try:
                smote = SMOTE(random_state=0, k_neighbors=k_neighbors)
                X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_part, y_train_part)
                smote_done = True
            except:
                k_neighbors -= 1
            

        # Step 3: Initialize the XGBClassifier
        xgb_classifier = XGBClassifier(
            random_state=0,
            use_label_encoder=False,
            objective='multi:softprob',
            eval_metric=['mlogloss', 'merror'],
            n_estimators = 300,
        )

        # Step 4:  Fit the model on the oversampled training data, including early stopping
        xgb_classifier.fit(X_train_oversampled, y_train_oversampled, 
                           eval_set=[(X_val, y_val)], 
                           early_stopping_rounds=10, 
                           eval_metric=['mlogloss', 'merror'],
                           verbose=False)

        # Step 5: Get predictions
        Y_pred = xgb_classifier.predict(X_test)
        
        # Step 6: Get metrics
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        accuracy = accuracy_score(Y_test, Y_pred)

        return precision, recall, f1, accuracy

    def average_metrics(self, X_train, Y_train, X_test, Y_test):
        metrics = np.zeros(4)
        for _ in range(self.n_runs):
            precision, recall, f1, accuracy = self.classification(X_train, Y_train, X_test, Y_test)
            metrics += np.array([precision, recall, f1, accuracy])
        return metrics / self.n_runs
