import operator

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def results(classifiers, scoring, X, y):
    for key, classifier in classifiers.items():
        classifier.fit(X, y)
        scores = cross_validate(classifier, X, y, scoring=scoring, cv=5)
        print("\nClassifier: ", key, "\n")
        for s in scoring:
            print("Training score of",
                  "%s: %.2f (+/- %.2f)" % (s, scores["train_" + s].mean(), scores["train_" + s].std()))


def clf_imb_report(classifiers, X, y, X_test, y_test):
    for key, classifier in classifiers.items():
        classifier.fit(X, y)
        preds = classifier.predict(X_test)
        print("Classifiers: ", classifier.__class__.__name__,
              '\n', classification_report_imbalanced(preds, y_test))


def resample_cross_val(clf, X, y, sampler, cv):
    # We are preserving 10 percent of our original data in order to cross validate
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=0.1, random_state=0)

    acc = []
    b_acc = []
    a_p_c = []
    roc = []
    gm = []

    for train_index, val_index in sss.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        rund = sampler
        X_resampled, y_resampled = rund.fit_resample(X_train, y_train)

        clf.fit(X_resampled, y_resampled)
        preds = clf.predict(X_val)

        # Compute metrics
        acc.append(accuracy_score(preds, y_val))
        b_acc.append(balanced_accuracy_score(preds, y_val))
        a_p_c.append(average_precision_score(preds, y_val))
        roc.append(roc_auc_score(preds, y_val))
        gm.append(geometric_mean_score(preds, y_val))

    scores = {'Accuracy Score = ': np.round(np.mean(acc), 3), 'Accuracy Std = ': np.round(np.std(acc), 3),
              'Balanced Accuracy Score = ': np.round(np.mean(b_acc), 3),
              'Balanced Accuracy Std = ': np.round(np.std(b_acc), 3),
              'Average Precision Recall Score = ': np.round(np.mean(a_p_c), 3),
              'Average Precision Recall Std = ': np.round(np.std(a_p_c), 3),
              'Roc Auc Score = ': np.round(np.mean(roc), 3), 'Roc Auc Std = ': np.round(np.std(roc), 3),
              'G Mean Score = ': np.round(np.mean(gm), 3), 'G Mean Std = ': np.round(np.std(gm), 3)}

    return scores


def hyperparameters(classifier, params, X, y, scoring, clf_name, random_search=True):
    best_params = []
    cv_results = []
    best_scores = []
    print("\nΕstimator : " + clf_name)
    if random_search:
        searcher = RandomizedSearchCV(classifier, params, cv=4, n_jobs=-1,
                                      verbose=1, scoring=scoring, refit='G-Mean')
    else:
        searcher = GridSearchCV(classifier, params, cv=4, n_jobs=-1,
                                verbose=1, scoring=scoring, refit='G-Mean')

    # Finding the best parameters in the original set in order to generalize better
    searcher.fit(X, y)
    cv_results.append(searcher.cv_results_)
    best_params.append(searcher.best_params_)
    best_scores.append(searcher.best_score_)

    final_results = [best_params, cv_results, best_scores]

    print('Best parameters found for Estimator : %s' % clf_name)
    print(searcher.best_params_)
    print("\nBest score found for G-Mean Score metric : %.3f" % searcher.best_score_)

    return final_results


def graph_roc_curve_multiple(est1_fpr, est1_tpr, est1_roc, est2_fpr, est2_tpr, est2_roc, est3_fpr, est3_tpr, est3_roc,
                             name):
    plt.figure(figsize=(16, 8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(est1_fpr, est1_tpr, label=name[0] + 'Score: {:.4f}'.format(est1_roc))
    plt.plot(est2_fpr, est2_tpr, label=name[1] + 'Score: {:.4f}'.format(est2_roc))
    plt.plot(est3_fpr, est3_tpr, label=name[2] + 'Score: {:.4f}'.format(est3_roc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                 arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                 )
    plt.legend()


def roc_curve_(est_fpr, est_tpr, name):
    plt.figure(figsize=(12, 8))
    plt.title(name + 'ROC Curve', fontsize=16)
    plt.plot(est_fpr, est_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01, 1, 0, 1])
    plt.show()


def prec_rec_curve(prec, rec, avg_prec):
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(rec, prec, color='b', alpha=0.2,
             where='post')
    plt.fill_between(rec, prec, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(avg_prec))


class VotingClassifier:

    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights
        self.classes_ = None
        self.probas_ = None

    def fit(self, X, y):
        for clf, X_set, y_set in zip(self.clfs, X, y):
            clf.fit(X_set, y_set)

    def predict(self, X):
        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])

        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:, c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg


def plot_confusion_matrix(y_test, y_pred):
    cfn_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()


def validate_easy_ensemble(estimator, X, y):
    acc = []
    b_acc = []
    a_p_c = []
    roc = []
    gm = []
    for key, x_val in zip(X.keys(), X.values()):
        preds = estimator.predict(x_val)

        acc.append(accuracy_score(preds, y[key]))
        b_acc.append(balanced_accuracy_score(preds, y[key]))
        a_p_c.append(average_precision_score(preds, y[key]))
        roc.append(roc_auc_score(preds, y[key]))
        gm.append(geometric_mean_score(preds, y[key]))

    scores = {'Accuracy Score = ': np.round(np.mean(acc), 3), 'Accuracy Std = ': np.round(np.std(acc), 3),
              'Balanced Accuracy Score = ': np.round(np.mean(b_acc), 3),
              'Balanced Accuracy Std = ': np.round(np.std(b_acc), 3),
              'Average Precision Recall Score = ': np.round(np.mean(a_p_c), 3),
              'Average Precision Recall Std = ': np.round(np.std(a_p_c), 3),
              'Roc Auc Score = ': np.round(np.mean(roc), 3), 'Roc Auc Std = ': np.round(np.std(roc), 3),
              'G Mean Score = ': np.round(np.mean(gm), 3), 'G Mean Std = ': np.round(np.std(gm), 3)}

    return scores
