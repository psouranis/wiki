###### SVM Classification Algorithm for Breast Cancer dataset #######
###### Artificial Intelligence ######
###### Statistical Learning  - Computational Intelligence #######

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import metrics
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


from svm import *


if __name__ =='__main__':
    #Load our data
    cancerdata=pd.read_csv('../dataset/breast-cancer-wisconsin-data/data.csv') #we need to put our full path of our data

    #Lets take a glimpse on our data
    cancerdata.head()

    # feature names as a list
    col = cancerdata.columns
    print(col)


    #Features of our data that dont help us with the classification
    #diagnosis is our target and we need it in y value
    list=['Unnamed: 32','id','diagnosis']
    x=cancerdata.drop(list,axis=1) #we drop out features that we dont need them
    x.head()

    #lets see some statistical measures about our dataset like mean,std,min,max etc.
    x.describe()

    #We have 2 classes malignant and beningn and we need to make them numbers in order to proceed
    #lets binary our data
    ##################
    label_quality = LabelEncoder()
    cancerdata['diagnosis'] = label_quality.fit_transform(cancerdata['diagnosis'])
    ##################

    cancerdata['diagnosis'].value_counts() #compute the count of both our classes

    sns.countplot(cancerdata['diagnosis']) #lets see a graph of our labeled data
    ##################


    y = cancerdata.diagnosis
    ax = sns.countplot(y,label="Count")       # M = 212, B = 357
    B, M = y.value_counts()
    print('Number of Benign: ',B)
    print('Number of Malignant : ',M)


    #Before we proceed we need our data to have target values -1 , 1 but instead we have 0 and 1
    #lets make a modification first
    #map 0 to -1
    ##################
    for i in range(len(y)):
        if y.values[i]==0:
            y.values[i]=-1


    #lets draw a boxplot in order to examinate min and maxes and Q1,Q2
    ##################
    data_dia = y
    data = x
    data_n_2 = (data - data.mean()) / (data.std()) #Normalization ~N(0,1) Distribution
    data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1) #for the first 10 features
    data = pd.melt(data,id_vars="diagnosis", #we can use 10:20 for our next 10 features
                        var_name="features",
                        value_name='value')
    plt.figure(figsize=(10,10))
    sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
    plt.xticks(rotation=90)
    ##################

    #jointplot about our 2 variablesthat shows us the corellation they have
    sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="red")


    #Now we are ready to target our data values
    ##################
    y = y.values
    ##################
    #Also our target values need to be float dtype
    ##################
    y=y.astype(float)
    ##################


    #Visualize our correlation Matrix
    ##################

    f,ax = plt.subplots(figsize=(14,14))
    sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

    ##################


    x=x.values
    #Lets split our data to training and test
    ##################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    ##################



    #lets visualize how much information is preserved for each component we keep

    #Normalization First
    scaler=StandardScaler()
    x_train_N = scaler.fit_transform(x_train)
    x_test_N = scaler.transform(x_test)
    #####
    pca = PCA()
    pca.fit(x_train_N)

    #Lets see how the information the components have decreases according
    plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')

    #Lets see how much information we have in total for every component we keep
    sumplot=np.array(pca.explained_variance_ratio_)
    print(np.sum(sumplot))
    for i in range(1,len(sumplot)):
        sumplot[i]=sumplot[i]+sumplot[i-1]
    plt.figure(1,figsize=(10,10))
    plt.clf()
    plt.axes([.2,.2,.7,.7])
    plt.plot(sumplot,linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('Information Preserved')

    #Lets keep our minimum features in order to have at least 90% information preserved
    ##################

    #Normalization First! Independent of PCA
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test) #carefull TRANSFORM and not fit again


    #Applying PCA
    pca = PCA(0.90)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    #Plotting our PCA
    ##################
    plt.figure(figsize = (15,15))
    plt.subplot(131)
    plt.scatter(x_train[:,0],x_train[:,1],c = y_train,
                edgecolor = "None", alpha=0.2,cmap="coolwarm")
    plt.title('PCA Scatter Plot')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();
    plt.subplot(132)
    plt.scatter(x_train[:,2],x_train[:,3],c = y_train,
                cmap="coolwarm", edgecolor = "None", alpha=0.2)
    plt.title('PCA Scatter Plot')
    plt.xlabel('component 3')
    plt.ylabel('component 4')
    plt.colorbar();
    plt.subplot(133)
    plt.scatter(x_train[:,4],x_train[:,5],c = y_train,
                cmap="coolwarm", edgecolor = "None", alpha=0.2)
    plt.title('PCA Scatter Plot')
    plt.xlabel('component 5')
    plt.ylabel('component 6')
    plt.colorbar();
    plt.show()



    #Lets see how much information each feature contains
    ##################
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    pcaratio=pd.DataFrame(pca.explained_variance_ratio_)
    for i in range(len(pcaratio)):
        pcaratio=pcaratio.rename(index={i:"Component " +str(i+1)},columns={0:"PCA Explained Variance Ratio"})
    print(pcaratio)

    ##################



    #Lets make a Grid for GridSearch in our algorithm
    #Not an exhaustive Grid search for finding the best parameters for the kernels,for rbf the best sigma
    #and for polynomial the best degree of the polynomial
    #There was held only a grid search for finding the best C
    ##################
    GridSearch=[0.125,0.25,0.5,1,2,3,4,5,6,8,16]
    GridSearch=pd.DataFrame(GridSearch).T
    GridSearch=GridSearch.rename(index=str,columns={0:"Values for C"})
    print(GridSearch)
    #################
    #Remake float datatype again
    GridSearch=GridSearch.T.values

    ##################

    #Set precision,recall,f1 lists to compair our results
    ##################

    precision=np.zeros((len(GridSearch)),dtype=float)
    recall=np.zeros((len(GridSearch)),dtype=float)
    f1=np.zeros((len(GridSearch)),dtype=float)


    #We will compute the time needed to finish the classification in each repetition
    ##################

    times=np.zeros((len(GridSearch)),dtype=float)


    #Lets apply our classifier
    ##################
    for i in range(len(GridSearch)):
        start_time=time.time()

        clf = SVM(kernel=linear_kernel,C=GridSearch[i])
        clf.fit(x_train,y_train)

        results=clf.predict(x_test)
        #A macro-average will compute the metric independently for each class and then take the average
        #A micro-average will aggregate the contributions of all classes to compute the average metric

        precision[i]=metrics.precision_score(y_test,results,average='macro')
        recall[i]=metrics.recall_score(y_test,results,average='macro')
        f1[i]=metrics.f1_score(y_test,results,average='macro')
        times[i]=time.time()-start_time

    ##################

    #Will use the following code for our best parameter after the grid search to see our best results#
    ###### Only after Grid Search ######
    clf = SVM(kernel=linear_kernel,C=0.125)
    clf.fit(x_train,y_train)
    results=clf.predict(x_test)
    print(classification_report(y_test,results))

    ##Confusion Matrix##
    cm = confusion_matrix(y_test,results)
    df_cm=pd.DataFrame(cm)
    df_cm=df_cm.rename(index={0:"Malignant",1:"Benign"},columns={0:"Malignant",1:"Benign"})
    plt.figure(figsize=(5,5))
    sns.heatmap(df_cm,annot=True)
    ######                       ######
    print(metrics.f1_score(y_test,results,average='macro'),metrics.precision_score(y_test,results,average='macro'),metrics.recall_score(y_test,results,average='macro'))


    #Time needed
    times=pd.DataFrame(times)
    times=times.rename(index=str,columns={0:"Time Needed to finish"})
    times.head()
    ##################

    #Plot our precision for C values
    ##################

    plt.figure(figsize=(12,6))
    plt.plot(GridSearch,precision,color='orange',marker='o')
    plt.title('Precision C Value')
    plt.xlabel('C value')
    plt.ylabel('Score')


    #Plot our Recall for C values
    ##################

    plt.figure(figsize=(12,6))
    plt.plot(GridSearch,recall,color='red',marker='o')
    plt.title('Recall C Value')
    plt.xlabel('C value')
    plt.ylabel('Score')

    #Plot our F1 for C Values
    ##################

    plt.figure(figsize=(12,6))
    plt.plot(GridSearch,f1,color='blue',marker='o')
    plt.title('F1 C Value')
    plt.xlabel('C value')
    plt.ylabel('Score')

    #Best C for Precision
    ##################
    print("Best C for maximizing precision after Grid Search is: ",GridSearch[np.argmax(precision)])
    print("Precision after applying best C",precision[np.argmax(precision)])
    ##################

    #Best C for Recall
    ##################
    print("Best C for maximizing Recall after Grid Search is: ",GridSearch[np.argmax(recall)])
    print("Recall after applying best C",recall[np.argmax(recall)])
    ##################

    #Best C for F1
    ##################
    print("Best C for maximizing F1 after Grid Search is: ",GridSearch[np.argmax(f1)])
    print("F1 after applying best C",f1[np.argmax(f1)])
    ##################

    #Lets plot our best time
    ##################

    print(times)
    print("fastest algorithm was:",np.argmin(times.values))
