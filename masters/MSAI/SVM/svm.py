import numpy as np
from numpy import linalg
import cvxopt #### <- Necessary for our SVM
import cvxopt.solvers #### <- Necessary for our SVM


def linear_kernel(x1, x2): #our linear kernel
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3): #our polynomial kernel
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=3): #our gaussian (rbf) kernel
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM:

    def __init__(self, kernel=gaussian_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        
        n_samples, n_features = X.shape # .shape returns the dimensios of the matrix X

        
        K = np.zeros((n_samples, n_samples)) 
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j]) #K is our matrix that contains the values of (Φ(Χ[i],Φ(X[j])))

        P = cvxopt.matrix(np.outer(y,y) * K) # P is our matrix for our elements YkYjΦ(X[k])Φ(X[j]) we used in our kernel trick
        q = cvxopt.matrix(np.ones(n_samples) * -1) #we want q to be an array of [-1,-1,....,-1] so that  we will we have -Aj for every j
        A = cvxopt.matrix(y, (1,n_samples)) #We define A so we will have a sum of Yjaj and that sum must eqaul as b
        b = cvxopt.matrix(0.0) #b must equals zero according to our theory

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1)) #We want our Aj>=0 ,but because in cvxopt Gx<=h so we need  
            h = cvxopt.matrix(np.zeros(n_samples)) # -1 so that we have -Aj<=0 which means Aj>=0
        else: #if we have a real value C
            tmp1 = np.diag(np.ones(n_samples) * -1) # that part will ensure us Aj>=0 
            tmp2 = np.identity(n_samples) # that part will ensure us Aj<=C
            G = cvxopt.matrix(np.vstack((tmp1, tmp2))) #G must be a stack of both parts
            tmp1 = np.zeros(n_samples) #for our first condition Aj>=0
            tmp2 = np.ones(n_samples) * self.C #for our second condition Aj<=C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2))) #h will be a stack of both conditions

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b) #we find our solution next

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5 #we only need the non zero lagrange multipliers
        ind = np.arange(len(a))[sv] #and their indexes in the array
        self.a = a[sv] #we keep the non zero lagrange multipliers
        self.sv = X[sv] #those will be our support vectors that correspond to non zero lagrange multipliers
        self.sv_y = y[sv] #those will be our targets for our support vectors
        print ("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0 #next we want to find our b 
        for n in range(len(self.a)): #We will find it by finding the mean for all support vectors
            self.b += self.sv_y[n] #instead of finding the value for only argmax of support vectors
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv]) #its a more stable way
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel: #if we have a linear kernel we dont need to project our w because we
            self.w = np.zeros(n_features) #didn't increase our dimensionality
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else: #else we have to project our w something that is very hard so we will find its similarity with Φ(X[k])
            self.w = None #and we will use our kernel trick

    def project(self, X):
        if self.w is not None: 
            return np.dot(X, self.w) + self.b
        else:  #because it is very difficult to compute w with Χk projected with a kernel
            y_predict = np.zeros(len(X)) #we will find the similarity w and Φ(x[k]) has 
            for i in range(len(X)): 
                s = 0 #s will be the sum of AjYjΦ(X[j])Φ(Χ) which is the only thing we need in order to find 
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv): #our solution according to our kernel trick
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s 
            return y_predict + self.b

    def predict(self, X): #returns our sign {-1,1}
        return np.sign(self.project(X))
