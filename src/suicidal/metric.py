from typing import List, Tuple, Callable, Dict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score)

def calculate_evaluation_scores(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float, float, str]:
    '''
    Evaluate the model using accuracy, precision, recall, and F1 score. NOTE -->  (Macro averaging)
    :param y_true: The true labels.
    :type y_true: list
    :param y_pred: The predicted labels.
    :type y_pred: list
    :return: A tuple containing the accuracy, precision, recall, F1 score , and a string describing the metrics.
    :rtype: tuple
    '''
    accuracy = accuracy_score(y_true, y_pred)
    percision = precision_score(
        y_true, y_pred, average="macro", zero_division=1)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)
    report = classification_report(y_true, y_pred, zero_division=1)
    matrix = confusion_matrix(y_true, y_pred)
    # format confusion matrix for printing
    matrix_str = "Confusion Matrix:\n"
    matrix_str += "      Predicted\n"
    matrix_str += "       " + " ".join([str(i)
                                       for i in range(len(matrix))]) + "\n"
    matrix_str += "True\n"
    matrix_str += "\n".join([f"{i:<6} {row}" for i, row in enumerate(matrix)])

    # Append confusion matrix to the report
    report_with_matrix = report + "\n\n" + matrix_str

    return (accuracy, percision, recall, f1, report_with_matrix)
