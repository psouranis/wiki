import matplotlib.pyplot as plt
import seaborn as sns
from costcla.metrics import cost_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, brier_score_loss


def _cs_report(true, predicted, label_names, cost_matrix) -> None:
    """
    Shows a full cost sensitive classification report.

    :param cost_matrix: the cost matrix.
    :param label_names: the class names.
    :param true: the true labels.
    :param predicted: the predicted labels.
    """
    # Show a classification report.
    print(classification_report(true, predicted, target_names=label_names))

    # Create a confusion matrix with the metrics.
    matrix = confusion_matrix(true, predicted)

    # Create a heatmap of the confusion matrix.
    plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, annot=True, fmt='d', linewidths=.1, cmap='YlGnBu',
                cbar=False, xticklabels=label_names, yticklabels=label_names)
    plt.title('Total Classification Cost -> {}'.format(cost_loss(true, predicted, cost_matrix)), fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.xlabel('True output', fontsize='x-large')
    plt.ylabel('Predicted output', fontsize='x-large')
    plt.savefig(fname='confusion_matrix.png')
    plt.show()


def full_cs_report(y_test, y_forest, y_svm, y_bayes, label_names, cost_matrix) -> None:
    """
    Make a report for all the cost sensitive classifiers.

    :param y_test: the test labels.
    :param y_forest: the random forest predicted labels.
    :param y_svm: the svm predicted labels.
    :param y_bayes: the bayes predicted labels.
    :param cost_matrix: the cost matrix.
    :param label_names: the class names.
    """
    print('Random Forest: \n')
    _cs_report(y_test, y_forest, label_names, cost_matrix)
    print('\n---------------------------------------------------------------\n')
    print('SVM: \n')
    _cs_report(y_test, y_svm, label_names, cost_matrix)
    print('\n---------------------------------------------------------------\n')
    print('Bayes: \n')
    _cs_report(y_test, y_bayes, label_names, cost_matrix)


def cost_loss_func(y_true, y_pred) -> int:
    """
    Define a cost loss function.

    :param y_true: the true labels.
    :param y_pred: the predicted labels.
    :return: the total cost.
    """
    if y_true.shape[0] is not y_pred.shape[0]:
        raise ValueError('True labels and predicted labels shapes do not match!')

    total_cost = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            total_cost += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            total_cost += 5

    return total_cost


# Define cost loss scorer.
cost = make_scorer(cost_loss_func, greater_is_better=False)


def plot_calibration_curve(est, name, X_train, y_train, X_test, y_test) -> None:
    """
    Plot calibration curve for an estimator without and with calibration.

    :param est: the estimator.
    :param name: the estimator's name.
    :param X_train: the train data.
    :param y_train: the train labels.
    :param X_test: the test data.
    :param y_test: the test labels.
    """
    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=10, method='sigmoid')

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name), (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_train.max())

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('{}\nCalibration plots  (reliability curve)'.format(name))

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()
