###### Spectral Embedding & Spectral Clustering ######
###### Artificial Intelligence ######
###### Statistical Learning  - Computational Intelligence #######


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor

save_pictures_to = 'C:/Users/User/Desktop/tefas/shapes/pictures directory/'

def clustering_print_scores(labels,X,n):
    cla = KMeans(n_clusters = n,random_state=0).fit(X)
    pred_classes = cla.predict(X)
    print('--Number of clusters is %.f --\n' %n)
    print('--Homogenity of clusters is %.3f --' %metrics.homogeneity_score(labels,pred_classes),'\n')
    print('--Silhouette measure of clusters is %.3f--' %metrics.silhouette_score(X,pred_classes),'\n')
    print('--Completness of clusters is %.3f --' %metrics.completeness_score(labels,pred_classes),'\n')
    print('--V measure score of clusters is %.3f--' %metrics.v_measure_score(labels,pred_classes),'\n')
    print('--Adjusted Mutual Information score of clusters is %.3f--'\
          %metrics.adjusted_mutual_info_score(labels,pred_classes),'\n')
    print('--Calinski Harabaz Index score is %.3f--' %metrics.calinski_harabaz_score(X,pred_classes),'\n')
    print('--Purity Score is %.3f--' %purity_score(labels,pred_classes))
    vm[n-2] = metrics.v_measure_score(labels,pred_classes)
    pur[n-2] = purity_score(labels,pred_classes)
    sil[n-2] = metrics.silhouette_score(X,pred_classes)
    cal[n-2] = metrics.calinski_harabaz_score(X,pred_classes)
    c_m = contingency_matrix(labels,pred_classes)
    df_cm=pd.DataFrame(c_m).rename(index = {i:'Class : %.f' %i for i in range(4)}\
    ,columns={i:'Cluster : %.f' %i for i in range(n)})
    plt.figure(figsize=(5,5))
    sns.heatmap(df_cm,annot=True,linewidths=.5)
    plt.title('Contigency Matrix C_ij True class i and Predicted j')
    plt.show()

def plot(X,y,title):
    plt.figure(figsize = (8,8))
    for color,marker,i in zip(colorlist,markers,n_r):
        plt.scatter(X [y == i ,0] ,X [y == i,1],
                    marker=marker , color=color ,label='shapes'+str(i))
    plt.title(title+ 'scatter plot') 
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()    
    plt.show() 

#Using Bayesian Ridge because its suited for datasets with large features
def regression(X_before,X_After,X_test):
    regressor = linear_model.BayesianRidge()
    multi_reg = MultiOutputRegressor(regressor).fit(X_before,X_After)
    X_new = multi_reg.predict(X_test)
    return(X_new)


def plot_metrics(X, title):
    plt.figure(figsize=(7, 7))
    plt.style.use('bmh')
    plt.plot(range(2, 8), X, marker='o')
    plt.xlabel('Number of clusters, k')
    plt.ylabel(title)
    plt.show()


def scores_for_test(labels,pred_classes,X):
    print('--Number of clusters is %.f --\n' %4)
    print('--Homogenity of clusters is %.3f --' %metrics.homogeneity_score(labels,pred_classes),'\n')
    print('--Silhouette measure of clusters is %.3f--' %metrics.silhouette_score(X,pred_classes),'\n')
    print('--Completness of clusters is %.3f --' %metrics.completeness_score(labels,pred_classes),'\n')
    print('--V measure score of clusters is %.3f--' %metrics.v_measure_score(labels,pred_classes),'\n')
    print('--Adjusted Mutual Information score of clusters is %.3f--'\
          %metrics.adjusted_mutual_info_score(labels,pred_classes),'\n')
    print('--Calinski Harabaz Index score is %.3f--' %metrics.calinski_harabaz_score(X,pred_classes),'\n')
    print('--Purity Score is %.3f--' %purity_score(labels,pred_classes))
    c_m = contingency_matrix(labels,pred_classes)
    df_cm=pd.DataFrame(c_m).rename(index = {i:'Class : %.f' %i for i in range(4)}\
    ,columns={i:'Cluster : %.f' %i for i in range(4)})
    plt.figure(figsize=(5,5))
    sns.heatmap(df_cm,annot=True,linewidths=.5)
    plt.title('Contigency Matrix C_ij True class i and Predicted j')
    plt.show()

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
