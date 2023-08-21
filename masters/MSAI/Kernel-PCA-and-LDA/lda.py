##LDA feature extraction ##

# References
#[1] Nonlinear Component Analysis as a Kernel Eigenvalue Problem Bernhard Scholkopf Alexander Smola and KlausRobert Muller.
#[2] KPCA Plus LDA: A Complete Kernel Fisher Discriminant Framework for Feature Extraction and Recognition Jian Yang, Alejandro F. Frangi, Jing-yu Yang, David Zhang, Senior Member, IEEE, and Zhong Jin.


from scipy.linalg import eigh
import numpy as np
from numpy import linalg
from numpy.linalg import inv
from sklearn import datasets

import numpy as np

class LDA:
    def __init__(self,X,n_components=None):
        self.n_components=n_components
        
        
    def compute_mean_vectors(self,X,y):
                
        classes = np.unique(y)
        means = []
        self.n_features=X.shape[1]
        for group in classes:
            X_classes = X[y == group, :]
            print(X_classes)
            means.append(np.mean(X_classes,axis=0))
        self.meanvectors=np.asarray(means)
        return np.asarray(means)



    def compute_S_k(self,X,meanvectors):
        
        n_features=X.shape[1]
        meanvectors_=meanvectors
        S_k=np.zeros((n_features,n_features))
        Z=X-meanvectors_
        S_k=np.dot(Z.T,Z)
        return(S_k)



    def compute_swithin(self,X,y):

        classes=np.unique(y)
        n_features=X.shape[1]
        S_w=np.zeros((n_features,n_features))
        
        for i in range(len(classes)):
            X_covs_k=(X[ y==classes[i] , :])
            means=np.transpose(self.meanvectors[i])
            S_w+= self.compute_S_k(X_covs_k,means)
        self.S_w=S_w
        return(S_w)
        
        
    def compute_sbetween(self,X,y):
        
        classes,counts=np.unique(y,return_counts=True)
        n_features=X.shape[1]
        
        S_B=np.zeros((n_features,n_features))
        overall_mean=np.mean(X,axis=0)
        
        for i in range(len(classes)):
            N=counts[i]
            means=self.meanvectors[i]-overall_mean
            means=means.reshape(n_features,1)
            S_B+= N * np.dot(means,means.T)
        self.S_B=S_B
        return(S_B)
        
    def transform_matrix(self):
        self.eig_vals, self.eig_vecs = np.linalg.eig(np.linalg.inv(self.S_w).dot(self.S_B))
        print(np.round(self.eig_vals,4),'\n',self.eig_vecs)
        
    def get_components(self):
        self.eig_pairs=[(np.abs(self.eig_vals[i]),self.eig_vecs[:,i]) for i in range(self.n_features)]
        
        return(self.eig_pairs)
    def variance_explained(self):
        sum_eigvals=np.sum(self.eig_vals)
        explained_ratio=np.round(self.eig_vals/sum_eigvals,3)
        return(explained_ratio)

    def get_W(self):
        self.keepcomponents=sum(self.eig_vals>1e-3) #count the trues in our array to keep only the eigvals >1e-3
        self.W=np.column_stack(self.eig_vecs[:,i] for i in range(self.keepcomponents))
        return(self.W)
        
    def transform(self,X):
        return(X.dot(self.W))
