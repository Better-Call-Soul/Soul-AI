import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  learning_curve
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import plotly.graph_objects as go


def map_tag_pattern(df, tag_col, text_col, res_col):
    '''
    Map the dataset to the required format for training
    Args:
        df (pd.DataFrame): The dataset
        tag_col (str): The column containing the tags
        text_col (str): The column containing the text
        res_col (str): The column containing the responses
    Returns:
        pd.DataFrame: The mapped dataset
    '''
    dic = {tag_col:[], text_col:[], res_col:[]}

    for _, item in df.iterrows():
        ptrns = item[text_col]
        rspns = item[res_col]
        tag = item[tag_col]
        for j in range(len(ptrns)):
            dic[tag_col].append(tag)
            dic[text_col].append(ptrns[j])
            dic[res_col].append(rspns)

    return pd.DataFrame.from_dict(dic)


def plot_learning_curve(estimator, title, X, y, filename=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    '''
    Generate a simple plot of the test and training learning curve
    Args:
        estimator: The object to use to fit the data
        title: The title of the chart
        X: The training data
        y: The target labels
        ylim: Tuple with the minimum and maximum y values
        cv: Cross-validation generator
        n_jobs: Number of jobs to run in parallel
        train_sizes: The training sizes
    Returns:
        plt: The plot
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    
    plt.savefig(filename)

    plt.show()
    return plt


# Confusion Matrix
def confusion_matrix_sklearn(predictions, target, filename=None):
    '''
    Generate a confusion matrix
    :param predictions: The predicted labels
    :type predictions: list
    :param target: The true labels
    :type target: list
    :param filename: The filename to save the plot
    :type filename: str
    '''
    class_names = np.unique(np.concatenate((predictions, target)))
    cm = confusion_matrix(target, predictions)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(filename)
    

def report_classification(y_test, y_pred, filename=None):
    '''
    Generate a classification report
    :param y_test: The true labels
    :type y_test: list
    :param y_pred: The predicted labels
    :type y_pred: list
    :param filename: The filename to save the plot
    :type filename: str
    '''
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    report = {label: {metric: report[label][metric] for metric in report[label]} for label in report if isinstance(report[label], dict)}
    labels = list(report.keys())

    evaluation_metrics = ['precision', 'recall', 'f1-score']
    metric_scores = {metric: [report[label][metric] for label in labels] for metric in evaluation_metrics}

    fig = go.Figure()
    for metric in evaluation_metrics:
        fig.add_trace(go.Bar(name=metric, x=labels, y=metric_scores[metric]))

    fig.update_layout(title='Intent Prediction Model Performance',
                      xaxis_title='Intent',
                      yaxis_title='Score',
                      barmode='group')

    fig.show()
    fig.write_image(filename)
    