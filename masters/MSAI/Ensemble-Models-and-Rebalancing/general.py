import numpy as np


def report(results, n_top=3) -> None:
    """
    Utility function to report best scores of Grid/Random Search CV.

    :param results: the search results.
    :param n_top: the number of top scores to show.
    """
    for i in range(1, n_top + 1):
        scores = np.flatnonzero(results['rank_test_score'] == i)
        for score in scores:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                results['mean_test_score'][score],
                results['std_test_score'][score]))
            print('Parameters: {0}'.format(results['params'][score]))
            print('')
