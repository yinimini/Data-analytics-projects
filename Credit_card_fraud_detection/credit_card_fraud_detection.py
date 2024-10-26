import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import time
from snapml import DecisionTreeClassifier as SnapMLDecisionTreeClassifier, SupportVectorMachine
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset from a CSV file.

    Args:
    - file_path(str): Path to the CSV file.

    Returns:
    - data(pd.DataFrame): DataFrame containing the dataset.
    """
    data = pd.read_csv(file_path)
    print(f"There are {len(data)} observations in the credit card fraud dataset.")
    print(f"There are {len(data.columns)} columns in the dataset.")
    return data


def inflate_dataset(data: pd.DataFrame, factor: int = 10) -> pd.DataFrame:
    """
    Inflate the dataset by repeating each observation.

    Args:
    - data(pd.DataFrame): DataFrame to be inflated.
    - factor(int): Number of times to repeat each observation.

    Returns:
    - Inflated DataFrame.
    """
    return pd.DataFrame(np.repeat(data.values, factor, axis=0), columns=data.columns)


def plot_target_distribution(data: pd.DataFrame) -> None:
    """
    Plot the distribution of the target variable.

    Args:
    - data(pd.DataFrame): DataFrame containing the dataset.
    """
    labels = data.Class.unique()
    sizes = data.Class.value_counts().values

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.3f%%')
    ax.set_title('Target Variable Value Counts')
    plt.show()


def plot_transaction_amounts(data: pd.DataFrame) -> None:
    """
    Plot histogram of transaction amounts and display some statistics.

    Args:
    - data(pd.DataFrame): DataFrame containing the dataset.
    """
    plt.hist(data.Amount.values, bins=6)
    plt.show()
    print("Minimum amount value is ", np.min(data.Amount.values))
    print("Maximum amount value is ", np.max(data.Amount.values))
    print("90 percent of the transactions have an amount less or equal than ",
          np.percentile(data.Amount.values, 90))


def preprocess_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the dataset by scaling features and normalizing.

    Args:
    - data(pd.DataFrame): DataFrame containing the dataset.

    Returns:
    - X, y(tuple) Tuple containing the feature matrix and target vector.
    """
    data.iloc[:, 1:30] = StandardScaler().fit_transform(data.iloc[:, 1:30])
    X = normalize(data.values[:, 1:30], norm='l1')
    y = data.values[:, 30]
    print(f'X.shape={X.shape}, y.shape={y.shape}')
    return X, y


def split_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and test sets.

    Args:
    - X(np.ndarray): Feature matrix.
    - y(np.ndarray): Target vector.

    Returns:
    - X_train, X_test, y_train, y_test(tuple): Tuple containing training and test sets for features and target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f'X_train.shape={X_train.shape}, Y_train.shape={y_train.shape}')
    print(f'X_test.shape={X_test.shape}, Y_test.shape={y_test.shape}')
    return X_train, X_test, y_train, y_test


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute class weights for the training set.

    Args:
    - y_train(np.ndarray): Target vector for the training set.

    Returns:
    - w_train(np.ndarray):Array of computed sample weights.
    """
    w_train = compute_sample_weight('balanced', y_train)
    print("Class distribution in the training set:")
    print(pd.Series(w_train).value_counts())
    return w_train


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray) -> tuple:
    """
    Train Decision Tree classifiers using Scikit-Learn and Snap ML.

    Args:
    - X_train(np.ndarray): Training features.
    - y_train(np.ndarray): Training target.
    - w_train(np.ndarray): Sample weights for the training set.

    Returns:
    - sklearn_dt, snapml_dt, sklearn_time_dtc, snapml_time_dtc (Tuple):Tuple containing the trained Scikit-Learn and Snap ML Decision Trees.
    """
    # Train a Decision Tree Classifier using Scikit-Learn
    sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)
    t0 = time.time()
    sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
    sklearn_time_dtc = time.time() - t0
    print(f"[Scikit-Learn DTC] Training time (s):  {sklearn_time_dtc:.5f}")

    # Train a Decision Tree Classifier using Snap ML
    snapml_dt = SnapMLDecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
    t0 = time.time()
    snapml_dt.fit(X_train, y_train, sample_weight=w_train)
    snapml_time_dtc = time.time() - t0
    print(f"[Snap ML DTC] Training time (s):  {snapml_time_dtc:.5f}")

    return sklearn_dt, snapml_dt, sklearn_time_dtc, snapml_time_dtc


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Evaluate the model using ROC-AUC score.

    Args:
    - model: The trained model to evaluate.
    - X_test(np.ndarray): Test features.
    - y_test(np.ndarray): Test target.

    Returns:
    - roc_auc(float): ROC-AUC score.
    """
    predictions = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, predictions)
    return roc_auc


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train SVM classifiers using Scikit-Learn and Snap ML.

    Argss:
    - X_train(np.ndarray): Training features.
    - y_train(np.ndarray): Training target.

    Returns:
    - Tuple containing the trained Scikit-Learn and Snap ML SVM models.
    """
    # Train a linear Support Vector Machine model using Scikit-Learn
    sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
    t0 = time.time()
    sklearn_svm.fit(X_train, y_train)
    sklearn_time_svm = time.time() - t0
    print(f"[Scikit-Learn SVM] Training time (s):  {sklearn_time_svm:.2f}")

    # Train an SVM model using Snap ML
    snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
    t0 = time.time()
    snapml_svm.fit(X_train, y_train)
    snapml_time_svm = time.time() - t0
    print(f"[Snap ML SVM] Training time (s):  {snapml_time_svm:.2f}")

    return sklearn_svm, snapml_svm, sklearn_time_svm, snapml_time_svm


def plot_roc_curve(y_score: np.ndarray, y_test: np.ndarray, label: str) -> None:
    """
    Plot ROC curve for a given model.

    Args:
    - y_score(np.ndarray): Predicted scores or probabilities.
    - y_test(np.ndarray): True labels.
    - label(str): Label for the curve.
    """
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')


def plot_training_time_comparison(sklearn_times: list, snapml_times: list, models: list) -> None:
    """
    Plot a comparison of training times between Scikit-Learn and Snap ML.

    Args:
    - sklearn_times(list): List of training times for Scikit-Learn.
    - snapml_times(list): List of training times for Snap ML.
    - models: List of model names.
    """
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, sklearn_times, width, label='Scikit-Learn')
    rects2 = ax.bar(x + width / 2, snapml_times, width, label='Snap ML')

    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time by Model and Library')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.show()


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> None:
    """
    Plot confusion matrix for the given model and test data.

    Args:
    - model: The trained model
    - X_test(np.ndarray): Features of the test set
    - y_test(np.ndarray): True labels of the test set
    - model_name(str): The name of the model (for the plot title)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot()
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


if __name__ == "__main__":
    data_file = "creditcard.csv"
    data = load_data(data_file)
    
    # Data preprocessing
    data = inflate_dataset(data)
    plot_target_distribution(data)
    plot_transaction_amounts(data)

    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    w_train = compute_class_weights(y_train)

    # Train models
    sklearn_dt, snapml_dt, sklearn_time_dtc, snapml_time_dtc = train_decision_tree(X_train, y_train, w_train)
    sklearn_svm, snapml_svm, sklearn_time_svm, snapml_time_svm = train_svm(X_train, y_train)

    # Evaluate models
    print(f"[Scikit-Learn DTC] ROC AUC score: {evaluate_model(sklearn_dt, X_test, y_test):.5f}")
    print(f"[Snap ML DTC] ROC AUC score: {evaluate_model(snapml_dt, X_test, y_test):.5f}")
    print(f"[Scikit-Learn SVM] ROC AUC score: {evaluate_model(sklearn_svm, X_test, y_test):.5f}")
    print(f"[Snap ML SVM] ROC AUC score: {evaluate_model(snapml_svm, X_test, y_test):.5f}")

    # Plot ROC curves
    plt.figure()
    plot_roc_curve(sklearn_dt.predict_proba(X_test)[:, 1], y_test, 'Scikit-Learn Decision Tree')
    plot_roc_curve(snapml_dt.predict_proba(X_test)[:, 1], y_test, 'Snap ML Decision Tree')
    plot_roc_curve(sklearn_svm.decision_function(X_test), y_test, 'Scikit-Learn SVM')
    plot_roc_curve(snapml_svm.decision_function(X_test), y_test, 'Snap ML SVM')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Plot confusion matrices
    plot_confusion_matrix(sklearn_dt, X_test, y_test, "Scikit-Learn Decision Tree")
    plot_confusion_matrix(snapml_dt, X_test, y_test, "Snap ML Decision Tree")
    plot_confusion_matrix(sklearn_svm, X_test, y_test, "Scikit-Learn SVM")
    plot_confusion_matrix(snapml_svm, X_test, y_test, "Snap ML SVM")

    # Compare training times
    plot_training_time_comparison(
        [sklearn_time_dtc, sklearn_time_svm],
        [snapml_time_dtc, snapml_time_svm],
        ["Decision Tree", "SVM"]
    )
