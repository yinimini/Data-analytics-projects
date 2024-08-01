import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import time
from snapml import DecisionTreeClassifier, SupportVectorMachine
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cred_data = pd.read_csv("creditcard.csv")
print("There are " + str(len(cred_data)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(cred_data.columns)) + " columns in the dataset.")

#inflate the dataset
big_cred_data = pd.DataFrame(np.repeat(cred_data.values, 10, axis=0), columns = cred_data.columns)

#get the set of distinct classes
labels = big_cred_data.Class.unique()

#get the count of each class
sizes = big_cred_data.Class.value_counts().values

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

plt.hist(big_cred_data.Amount.values, 6)
plt.show()
print("Minimum amount value is ", np.min(big_cred_data.Amount.values))
print("Maximum amount value is ", np.max(big_cred_data.Amount.values))
print("90 procent of the transactions have an amount less or equal than ", np.percentile(cred_data.Amount.values, 90))

# dataset proprocessing 
big_cred_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_cred_data.iloc[:,1:30]) #column 0 is time, column 31 is class
data_matrix = big_cred_data.values
X = data_matrix[:, 1:30]
y = data_matrix[:, 30]

X = normalize(X, norm='l1')

print('X.shape=', X.shape, 'y.shape=', y.shape)

# train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 42, stratify = y)
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

#data is very inbalanced, gotta fix it, otherwise model fucked
w_train = compute_sample_weight('balanced', y_train)
print("Class distribution in the training set:")
print(pd.Series(w_train).value_counts())

'''
Decision Tree Classifier
'''
#train a Decision Tree Classifier using scikit-learn
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time_dtc = time.time()-t0
print("[Scikit-Learn DTC] Training time (s):  {0:.5f}".format(sklearn_time_dtc))


# train a Decision Tree Classifier model using Snap ML
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time_dtc = time.time()-t0
print("[Snap ML DTC] Training time (s):  {0:.5f}".format(snapml_time_dtc))


# Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time_dtc/snapml_time_dtc
print('[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

# run inference and compute the probabilities of the test samples 
# to belong to the class of fraudulent transactions
sklearn_pred_dtc = sklearn_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic 
# Curve (ROC-AUC) score from the predictions
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred_dtc)
print('[Scikit-Learn DTC] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

# run inference and compute the probabilities of the test samples
# to belong to the class of fraudulent transactions
snapml_pred_dtc = snapml_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic
# Curve (ROC-AUC) score from the prediction scores
snapml_roc_auc = roc_auc_score(y_test, snapml_pred_dtc)   
print('[Snap ML DTC] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))



'''
instatiate a scikit-learn SVM model
to indicate the class imbalance at fit time, set class_weight='balanced'
for reproducible output across multiple function calls, set random_state to a given integer value
'''
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

# train a linear Support Vector Machine model using Scikit-Learn
t0 = time.time()
sklearn_svm.fit(X_train, y_train)
sklearn_time_svm = time.time() - t0
print("[Scikit-Learn svm] Training time (s):  {0:.2f}".format(sklearn_time_svm))

# to set the number of threads used at training time, one needs to set the n_jobs parameter
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
# print(snapml_svm.get_params())

# train an SVM model using Snap ML
t0 = time.time()
model = snapml_svm.fit(X_train, y_train)
snapml_time_svm = time.time() - t0
print("[Snap ML svm] Training time (s):  {0:.2f}".format(snapml_time_svm))



# compute the Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time_svm/snapml_time_svm
print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))

# run inference using the Scikit-Learn model
# get the confidence scores for the test samples
sklearn_pred_svm = sklearn_svm.decision_function(X_test)

# evaluate accuracy on test set
acc_sklearn  = roc_auc_score(y_test, sklearn_pred_svm)
print("[Scikit-Learn svm] ROC-AUC score:   {0:.3f}".format(acc_sklearn))

# run inference using the Snap ML model
# get the confidence scores for the test samples
snapml_pred_svm = snapml_svm.decision_function(X_test)

# evaluate accuracy on test set
acc_snapml  = roc_auc_score(y_test, snapml_pred_svm)
print("[Snap ML svm] ROC-AUC score:   {0:.3f}".format(acc_snapml))

'''
# get the confidence scores for the test samples
sklearn_pred = sklearn_svm.decision_function(X_test)
snapml_pred  = snapml_svm.decision_function(X_test)
'''

# evaluate the hinge loss from the predictions
loss_snapml = hinge_loss(y_test, snapml_pred_svm)
print("[Snap ML svm] Hinge loss:   {0:.3f}".format(loss_snapml))

# evaluate the hinge loss metric from the predictions
loss_sklearn = hinge_loss(y_test, sklearn_pred_svm)
print("[Scikit-Learn svm] Hinge loss:   {0:.3f}".format(loss_snapml))


def plot_roc_curve(y_score, y_test, label):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
plt.figure()

#ROC SVM models
plot_roc_curve(sklearn_pred_svm, y_test, 'Scikit-Learn SVM')
plot_roc_curve(snapml_pred_svm, y_test, 'Snap ML SVM')

# Decision Tree models
plot_roc_curve(sklearn_pred_dtc, y_test, 'Scikit-Learn Decision Tree')
plot_roc_curve(snapml_pred_dtc, y_test, 'Snap ML Decision Treee')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

#Training Time Comparison
models = ['Decision Tree', 'SVM']
sklearn_times = [sklearn_time_dtc, sklearn_time_svm]  # Replace with actual times
snapml_times = [snapml_time_dtc, snapml_time_svm]  # Replace with actual times

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, sklearn_times, width, label='Scikit-Learn')
rects2 = ax.bar(x + width/2, snapml_times, width, label='Snap ML')

ax.set_ylabel('Training Time (s)')
ax.set_title('Training Time by Model and Library')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.show()

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Plot confusion matrix for the given model and test data.
    
    Parameters:
    model: The trained model
    X_test: Features of the test set
    y_test: True labels of the test set
    model_name: The name of the model (for the plot title)
    """
    # Predict labels
    y_pred_labels = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    
    # Plot confusion matrix
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
    cm_disp.plot()
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrix(sklearn_dt, X_test, y_test, 'Scikit-Learn DTC')
plot_confusion_matrix(snapml_dt, X_test, y_test, 'Snap ML DTC')
plot_confusion_matrix(sklearn_svm, X_test, y_test, 'Scikit-Learn SVM')
plot_confusion_matrix(snapml_svm, X_test, y_test, 'Snap ML SVM')


'''
# confusion matrix
# Predictions using predict method to get binary class labels
# For Scikit-Learn Decision Tree
sklearn_dt_pred_labels = sklearn_dt.predict(X_test)
cm_sklearn = confusion_matrix(y_test, sklearn_dt_pred_labels)
cm_sklearn_disp = ConfusionMatrixDisplay(confusion_matrix = cm_sklearn, display_labels = ['Legitimate', 'Fraud'])
cm_sklearn_disp.plot()
plt.title('Confusion Matrix: Scikit-Learn DTC')
plt.show() 

# For Snap ML Decision Tree
snapml_dt_pred_labels = snapml_dt.predict(X_test)
cm_snapml = confusion_matrix(y_test, snapml_dt_pred_labels)
cm_snapml_disp = ConfusionMatrixDisplay(confusion_matrix = cm_snapml, display_labels = ['Legitimate', 'Fraud'])
cm_snapml_disp.plot()
plt.title('Confusion Matrix: Snap ML DTC')
plt.show() 

# For Scikit-Learn SVM
sklearn_svm_pred_labels = sklearn_svm.predict(X_test)
cm_sklearn_svm = confusion_matrix(y_test, sklearn_svm_pred_labels)
cm_sklearn_svm_disp = ConfusionMatrixDisplay(confusion_matrix = cm_sklearn_svm, display_labels = ['Legitimate', 'Fraud'])
cm_sklearn_svm_disp.plot()
plt.title('Confusion Matrix: Scikit-Learn SVM')
plt.show() 

# For Snap ML SVM
snapml_svm_pred_labels = snapml_svm.predict(X_test)
cm_snapml_svm = confusion_matrix(y_test, snapml_svm_pred_labels)
cm_snapml_svm_disp = ConfusionMatrixDisplay(confusion_matrix = cm_snapml_svm, display_labels = ['Legitimate', 'Fraud'])
cm_snapml_svm_disp.plot()
plt.title('Confusion Matrix: Snap ML SVM')
plt.show()
'''