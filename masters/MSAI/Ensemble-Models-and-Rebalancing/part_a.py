import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve, cross_val_score


def estimators_vs_acc(classifier, x_data, y_data, estimators_array) -> None:
    """
    Plot number of estimators vs accuracy +- std.

    :param classifier: the classifier to be used.
    :param x_data: the features.
    :param y_data: the classes.
    :param estimators_array: an array containing the number of estimators to be used for each CV fit.
    """
    # Arrays to store mean and std.
    bg_clf_cv_mean = []
    bg_clf_cv_std = []

    # For each number of estimators run a 10 fold CV and store the results.
    for n_est in estimators_array:
        bagging_clf = BaggingClassifier(base_estimator=classifier,
                                        n_estimators=n_est, random_state=0)
        scores = cross_val_score(bagging_clf, x_data, y_data, cv=10,
                                 scoring='accuracy', verbose=2, n_jobs=-1)
        bg_clf_cv_mean.append(scores.mean())
        bg_clf_cv_std.append(scores.std())

    # Bound upper and lower bounds of the error bar to [0, 1].
    y_min = np.asarray([max(mean - std, 0) for mean, std in zip(bg_clf_cv_mean, bg_clf_cv_std)])
    y_max = np.asarray([min(mean + std, 1) for mean, std in zip(bg_clf_cv_mean, bg_clf_cv_std)])
    y_bot = bg_clf_cv_mean - y_min
    y_top = y_max - bg_clf_cv_mean

    # Plot the accuracy+-std vs number of estimators.
    plt.figure(figsize=(12, 8))
    (_, caps, _) = plt.errorbar(estimators_array, bg_clf_cv_mean,
                                yerr=(y_bot, y_top), c='blue', fmt='-o', capsize=5)

    # Configure the plot.
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.ylabel('Accuracy')
    plt.xlabel('Ensemble Size')
    plt.title('Bagging Tree Ensemble')
    plt.show()


def plot_accuracy_stacking(label_list, clfs, x_data, y_data) -> None:
    """
    Plot accuracy +- std for each classifier used in the stacking model
    and for the stacking classifier.

    :param label_list: a list containing the names of the classifiers for the plot.
    :param clfs: the classifiers.
    :param x_data: the features.
    :param y_data: the classes.
    """
    # Arrays to store mean and std.
    clf_cv_mean = []
    clf_cv_std = []

    # For each classifier run a 10 fold CV and store the results.
    for classifier, label in zip(clfs, label_list):
        scores = cross_val_score(classifier, x_data, y_data, cv=10, scoring='accuracy', n_jobs=-1)
        print("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
        clf_cv_mean.append(scores.mean())
        clf_cv_std.append(scores.std())
        classifier.fit(x_data, y_data)

    # Bound upper and lower bounds of the error bar to [0, 1].
    y_min = np.asarray([max(mean - std, 0) for mean, std in zip(clf_cv_mean, clf_cv_std)])
    y_max = np.asarray([min(mean + std, 1) for mean, std in zip(clf_cv_mean, clf_cv_std)])
    y_bot = clf_cv_mean - y_min
    y_top = y_max - clf_cv_mean

    # Plot the accuracy+-std for each classifier.
    plt.figure(figsize=(12, 8))
    (_, caps, _) = plt.errorbar(range(len(clfs)), clf_cv_mean,
                                yerr=(y_bot, y_top), c='blue', fmt='-o', capsize=5)

    # Configure the plot.
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.xticks(range(len(clfs)), label_list)
    plt.ylabel('Accuracy')
    plt.xlabel('Classifier')
    plt.title('Stacking Ensemble')
    plt.show()


def plot_learning_curve(estimators, names, x_data, y_data, train_sizes=np.linspace(0.2, 1.0, 5)) -> None:
    """
    Function to plot the learning curve of the stacking model and all its submodels,

    given some training data.

    :param estimators: an array containing all the 6 estimators.
    :param names: an array containing the estimators names.
    :param x_data: the features.
    :param y_data: the labels.
    :param train_sizes: the train size percentages for the x ax.
    """
    f, axes = plt.subplots(2, 3, figsize=(18, 12), sharey='all', squeeze=False)

    for counter, (estimator, ax) in enumerate(zip(estimators, axes.reshape(-1))):
        train_sizes, train_scores, test_scores = learning_curve(estimator, x_data, y_data,
                                                                train_sizes=train_sizes, cv=10, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title(names[counter] + " Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")


def full_report(true, predicted, averaging='macro') -> None:
    """
    Shows a full classification report.

    :param true: the true labels.
    :param predicted: the predicted labels.
    :param averaging: the averaging method to be used.
    """
    print('Final Results')
    print('---------------------')
    print('Accuracy       {:.4f}'
          .format(accuracy_score(true, predicted)))
    print('Precision      {:.4f}'
          .format(precision_score(true, predicted, average=averaging)))
    print('Recall         {:.4f}'
          .format(recall_score(true, predicted, average=averaging)))
    print('F1             {:.4f}'
          .format(f1_score(true, predicted, average=averaging)))
