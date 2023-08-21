from scipy.linalg import eigh
import numpy as np
from numpy import linalg

# References
#[1] Nonlinear Component Analysis as a Kernel Eigenvalue Problem Bernhard Scholkopf Alexander Smola and KlausRobert Muller.
#[2] KPCA Plus LDA: A Complete Kernel Fisher Discriminant Framework for Feature Extraction and Recognition Jian Yang, Alejandro F. Frangi, Jing-yu Yang, David Zhang, Senior Member, IEEE, and Zhong Jin.


def linear_kernel(x1, x2): #our linear kernel
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3): #our polynomial kernel
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=1): #our gaussian (rbf) kernel
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class KPCA:

    def __init__(self, kernel=gaussian_kernel, n_components=None,percentage=None):
        self.kernel = kernel
        self.n_components= n_components #how much components we wish to keep
        self.percentage=percentage #if we want to keep a certain percentage of our information
        if self.n_components is not None: self.n_components=int(self.n_components)
        if self.percentage is not None:self.percentage=float(self.percentage) 
    

    def fit(self,X):

        n_samples, n_features = X.shape
        self.x_fit=X #we keep the primary train dataset
        self.array=n_samples
        self.features=n_features

        K = np.zeros((n_samples, n_samples)) #construct our Gram Matrix dimension NXN
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        self.K_train=K #we keep the gram matrix of train from projections of test

        One_ = np.ones((n_samples,n_samples))/n_samples #construct our 1_N matrices
        self.K = K -  One_.dot(K) -K.dot(One_) + One_.dot(K).dot(One_) #Centerizing our Gram Matrix

        eigenvalues , eigenvectors = eigh(self.K) #Find Eigenvalues and Eigenvectors
        self.eig = eigenvalues #we keep the eigenvalues to examine how much information each eigenvalue contains

        self.projections=eigenvectors.dot(np.diag(np.sqrt(eigenvalues) / eigenvalues)) #we construct our Λ matrix
        #Its a diagonal matrix with elements in the diagonal sqrt(λ)/λ its the same as 1/sqrt(λ) for more efficient
        #arithmetic accuracy

        eigenvalues=np.sqrt(eigenvalues) #our eigenvalues will become sqrt in order to project our X_train
        self.eigenvectors=eigenvectors.dot(np.diag(eigenvalues)) #multiply each eigenvector with squared eigenvalues
        
        if self.n_components is None: #calculate how many components needed for certain percentage
            self.explain_variance_ratio() #first lets call our explain_variance function which returns the variance of non trivial eigenvalues
            self.n_components=self.get_percentage()
        
        #We keep our components that correspond to the higher eigenvectors
        #because our eigenvectors are in descenting order we need to sort in reverse
        self.pcomponents=np.column_stack((self.eigenvectors[:,-i] for i in range(1,self.n_components+1))) 

        return(self.pcomponents)

    def explain_variance_ratio(self):
        sum_eigenvalues=np.sum(self.eig) #sum of all eigenvalues
        self.eig=self.eig/sum_eigenvalues #explain variance ratio
        non_trivial=[] #keep the non trivial eigenvalues with threshold 1e-5
        for i in self.eig:
            if i>1e-5:
                non_trivial.append(i)
        self.non_trivial=np.array(sorted(non_trivial,reverse=True)) #sort them in descenting order
        return(np.round(self.non_trivial,3))
        
    def get_percentage(self):
        count=0 
        sum_percent=0 
        for i in self.non_trivial:
            if sum_percent<self.percentage and count<self.features:
                sum_percent+=i
                count+=1
            else:
                break
        return(count) #those will be our minimum n_components we need to keep in order to have certain percentage

    def transform_(self,X): #transform our test data

        n_samples,n_features=X.shape #our dimensions of the test data
        K=np.zeros((n_samples,self.array)) #construct our Gram matrix K(Φ(X_train),Φ(X_test))
        for i in range(n_samples):
            for j in range(self.array):
                K[i,j]=self.kernel(X[i],self.x_fit[j]) #Gram matrix dimenions LXN
                
        
        One_ = np.ones((n_samples,self.array)) / self.array #1'M will have dimension LXN and each element 1/N
        Ones=np.ones((self.array,self.array))/self.array #1M will have dimension NXN and each element 1/N


        K = K -  One_.dot(self.K_train) -K.dot(Ones) + One_.dot(self.K_train).dot(Ones) #centerizing the matrix

        pc_new=K.dot(self.projections) #construct our projected matrix KΛ and keep the projections that correspond
        #to the higher eigenvalues that are in descenting order in the Λ matrix
        self.pc_new=np.column_stack((pc_new[:,-i] for i in range(1,self.n_components+1)))
        return(self.pc_new)

    
    
