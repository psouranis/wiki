###### KPCA+LDA Algorithm for Gene Rna sequence  dataset #######
###### Souranis Panagiotis AEM:17 ######
###### Artificial Intelligence ######
###### Statistical Learning  - Computational Intelligence #######
#====================================================================#

import os
os.chdir('C:/Users/User/Desktop/MNIST KPCA+LDA') #insert path of modules
import numpy as np
import random
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from kpca import *
from lda import *
plotly.tools.set_credentials_file(username='panossouras', api_key='xmG7L2gcWVGoGqgQaZR9')



'Our Primary Kernels'
def linear_kernel(x1, x2): #our linear kernel
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3): #our polynomial kernel
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, gamma=0.005): #our gaussian (rbf) kernel
    return np.exp(-linalg.norm(x-y)**2 *gamma)


if __name__ =='__main__'
    ##### Lets load our dataset #####
    rnadata=pd.read_csv('C:/Users/User/Desktop/RNAData/data.csv')
    #Information about our Dataset
    rnadata.head()
    rnadata.info()
    print(rnadata.shape)

    X=rnadata.drop(['Unnamed: 0'],axis=1).values



    labels=pd.read_csv('C:/Users/User/Desktop/RNAData/labels.csv')
    labels.info()
    labels=labels['Class']

    labelenc=preprocessing.LabelEncoder()
    labelenc.fit(labels)
    originalclasses=pd.DataFrame(labelenc.classes_)
    y=labelenc.transform(labels)
    y=pd.Series(y).values

    classes,counts=np.unique(y,return_counts=True)

    #Lets plot some sequences
    k=random.randint(1,801)
    trace0 = go.Scatter(x = np.arange(1000),y = X[k,:1000],mode = 'markers',text=y[k],
       name = y[k], marker = dict(size = 10,color = 'rgba(152, 0, 0, .8)',
            line = dict(width = 2,color = 'rgb(0, 0, 0)')))

    layout = go.Layout(title= 'RNA Gene Sequence',
        showlegend= True)
    data = [trace0]
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='scatter-mode')



    #Make a pie chart about our data
    labels = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
    values = [300,78,146,141,136]
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#FFA64D',' #66CCFF']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=20),
                   marker=dict(colors=colors,
                               line=dict(color='#000000', width=2)))
    layout = go.Layout(title= 'GENES Pie Chart',showlegend= True)
    data=[trace]
    fig=dict(data=data , layout=layout)
    py.plot(fig, filename='styled_pie_chart')




    #Split our Dataset in train and test

    x_train, x_test,y_train,y_test = train_test_split(X,y, test_size=0.3, stratify=y)


    scaler=StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #Implement our KPCA sklearn
    kpca = KPCA(kernel=linear_kernel,percentage=0.93)

    x_train=kpca.fit_transform(x_train)
    x_test=kpca.transform(x_test)


    #Lets see how much information each eigenvalue keeps

    explainedvariance=kpca.explain_variance_ratio


    plt.figure(1,figsize=(10,10))
    plt.clf()
    plt.axes([.2,.2,.7,.7])
    plt.plot(explainedvariance ,linewidth=4)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('Information Preserved')


    cummulative_percentage=kpca.cummulative_percentage()*100
    print('we kept a total of : %f' %np.max(cummulative_percentage) +'% information')


    plt.figure(1,figsize=(8,8))
    plt.clf()
    plt.axes([.2,.2,.7,.7])
    plt.plot(cummulative_percentage,linewidth=4)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('Information Preserved')



    #Lets create some interactive visualisations for KPCA representation

    #For 2 Dimensions
    #the picture below is interractive, so we can see the labels of each point

    trace0 = go.Scatter(
        x = x_train[:,0],y = x_train[:,1],name = 'Target',
        mode = 'markers',text = y_train,showlegend = False,marker = dict(
        size = 10,color = y_train,colorscale ='Portland',showscale = True,
        line = dict(width = 2,color = 'rgb(255, 255, 255)'),
        opacity = 1)
    )
    data = [trace0]

    layout = go.Layout(title= 'Poly KernelPCA n=350 degree=3 plus LDA n=3',
        xaxis= dict(title= 'First Principal Component',
        gridwidth= 2,),yaxis=dict(
        title= 'Second Principal Component'),
        showlegend= True
    )
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='styled-scatter')

    #3D visualization of KernelPCA
    trace1 = go.Scatter3d(
        x=x_train[:,0],y=x_train[:,1],z=x_train[:,2],
        mode='markers',marker=dict(size=12,color=y_train,colorscale='Portland',opacity=0.8))

    data = [trace1]
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='3d-scatter-colorscale')


    #Implementation of Linear Discriminant Analysis

    lda_=LDA(n_components=3)
    lda_.fit(x_train,y_train)
    x_train=lda_.transform(x_train)
    x_test=lda_.transform(x_test)


    #Lets implement our kNN algorithm

    k=8
    prec=np.zeros(5)
    rec=np.zeros(5)
    f1=np.zeros(5)

    for i in range(3,k):
        clf=KNeighborsClassifier(n_neighbors=k,weights='uniform',metric='minkowski',p=2)
        clf.fit(x_train,y_train)
        predictions=clf.predict(x_test)

        prec[i-3]=metrics.precision_score(predictions,y_test,average='micro')
        rec[i-3]=metrics.recall_score(predictions,y_test,average='micro')
        f1[i-3]=metrics.f1_score(predictions,y_test,average='micro')


    # Get best value of precision,recall, f1 score, and their indexes.
    print('best precision is :' ,np.max(prec))
    print('best precision count is :',np.argmax(prec) +1)

    print('best recall is :' ,np.max(rec))
    print('best recall count is :',np.argmax(rec) +1)

    print('best_F1 is :' ,np.max(f1))
    print('best F1 count is :',np.argmax(f1)+1)


    # Plot stored results
    plt.figure(figsize=(12,6))
    plt.plot(range(1,k),f1,color='blue',marker='o')
    plt.title('f1 Rate K Value')
    plt.xlabel('K value')
    plt.ylabel('f1 score')

    plt.figure(figsize=(12,6))
    plt.plot(range(1,k),rec,color='red',marker='o')
    plt.title('rec Rate K Value')
    plt.xlabel('K value')
    plt.ylabel('rec score')

    plt.figure(figsize=(12,6))
    plt.plot(range(1,k),prec,color='orange',marker='o')
    plt.title('prec Rate K Value')
    plt.xlabel('K value')
    plt.ylabel('prec score')


    #Lets do a confussion matrix and a classification report for certain nearest neighbor
    clf=KNeighborsClassifier(n_neighbors=3,weights='distance',metric='minkowski',p=2)
    clf.fit(x_train,y_train)
    predictions=clf.predict(x_test)
    print(metrics.classification_report(predictions,y_test))

    #crossvalidation
    predictions_train=clf.predict(x_train)
    print(metrics.classification_report(predictions_train,y_train))

    classes=['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
    ##Confusion Matrix##
    cm = confusion_matrix(y_test,predictions)
    df_cm=pd.DataFrame(cm)
    df_cm = df_cm.rename(index={0:'BRCA', 1:'COAD', 2:'KIRC', 3:'LUAD', 4:'PRAD'} ,
                         columns={0:'BRCA', 1:'COAD', 2:'KIRC', 3:'LUAD', 4:'PRAD'})
    plt.figure(figsize=(10,10))
    sns.heatmap(df_cm,annot=True)



    #Logistic Regression Classifier
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf.fit(x_train,y_train)
    predict = clf.predict(x_test)
    print(classification_report(y_test,predict))

    predicts_train = clf.predict(x_train)
    print(classification_report(y_train,predicts_train))
