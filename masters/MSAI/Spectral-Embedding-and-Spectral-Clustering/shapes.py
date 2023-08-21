###### Spectral Embedding & Spectral Clustering ######
###### Artificial Intelligence ######
###### Statistical Learning  - Computational Intelligence #######

from utils import *


#Basic things we are going to need for plotting#
colorlist = ['#00e600','#0066ff','#ff9900','r']
markers = ['>','+','^','<']
n_r = range(4)
times=[]

if __name__ ==  '__main__':
    data = pd.read_csv('C:/Users/User/Desktop/tefas/shapes/shapes.csv').values
    labels = pd.read_csv('C:/Users/User/Desktop/tefas/shapes/labels.csv')



    'Encode our values with integers'
    enc = LabelEncoder()
    enc.fit(labels)
    enc.classes_

    y = enc.transform(labels) #our new target values


    'Because our dataset is too large we have to work in a limited dataset'
    'We will keep the X_new, y_new'
    X_test , X_new ,y_test, y_new= train_test_split(data,y,test_size=0.3 , stratify=y)


    plt.figure(figsize=(8,8))
    for i in range(0,20):
        plt.subplot(4,5,i+1)
        plt.imshow(X_new[i].reshape(64,64),
                   interpolation = "none", cmap = "gist_gray")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


    print(np.unique(y_new,return_counts=True)[1]) #make sure our new dataset is balanced

    'Scaling our Data'
    scaler = MinMaxScaler()

    X_new = scaler.fit_transform(X_new)
    X_test = scaler.transform(X_test)


    Neighbors=[5,10,15,20]

    perplexity= [10,30,50,100] #Its equivalent with the neighbors in other algorithms



    #Implement t-SNE Manifold Learning after the PCA
    for n,k in zip(perplexity,n_r):
        tsne = TSNE(n_components=2,perplexity=n,n_iter=1000)
        start_time = time.time()
        X_embedded_tsne = tsne.fit_transform(X_new)
        times.append(time.time()-start_time)
        plt.figure(figsize = (8,8))
        'plot'
        plot(X_embedded_tsne,y_new,title='TSNE')


    #Implement Isomap Manifold Learning directly after scaling
    for n,k in zip(Neighbors,n_r):
        embedding = Isomap(n_components=2,n_neighbors=n)
        start_time = time.time()
        x_embedded_isomap = embedding.fit_transform(X_new)
        times.append(time.time()-start_time)
        plt.figure(figsize = (8,8))
        'plot'
        plot(x_embedded_isomap,y_new,title='ISOMAP')


    #Implement LLE Manifold Learning for visualize

    for n,k in zip(Neighbors,n_r):
        embedding = LocallyLinearEmbedding(n_neighbors = n,n_components = 2)
        start_time = time.time()
        x_embedded_lle = embedding.fit_transform(X_new)
        times.append(time.time()-start_time)
        plt.figure(figsize = (8,8))
        'plot'
        plot(x_embedded_lle,y_new,title='LLE',neighbors = n)

    #Implement MDS Manifold Learning for visualize and clustering
    embedding_mds = MDS(n_components = 2 ,metric=True,dissimilarity='euclidean')
    start_time = time.time()
    x_embedded_mds = embedding_mds.fit_transform(X_new)
    times.append(time.time()-start_time)
    'Lets plot'
    plot(x_embedded_mds,y_new,title='MDS')

    #tSNE implementation for best perplexity after visualization
    embedding_tsne = TSNE(n_components = 2, perplexity = 50, metric= 'euclidean')
    x_embedded_tsne = embedding_tsne.fit_transform(X_new)


    #LLE implementation for best parameter of neighbors after visualization
    embedding_lle = LocallyLinearEmbedding(n_neighbors = 20,n_components = 2)
    x_embedded_lle = embedding_lle.fit_transform(X_new)

    #IsoMap implementation for best parameter of neighbors after visualization
    embedding_iso = Isomap(n_components = 2 , n_neighbors = 15)
    x_embedded_isomap = embedding_iso.fit_transform(X_new)


    'Now Lets implement our clustering algorithm'


    vm,pur,sil,cal = [np.zeros(6) for _ in range(4)]

    evaluation = [vm,pur,sil,cal]
    titles = ['V-measure','Purity','Silhouette Coefficient','Calinski-Harabaz index']


    clusters = range(2,8)


    #TSNE Evaluation#
    for n in clusters :
        clustering_print_scores(y_new,x_embedded_tsne,n)

    #LLE Evaluation#
    for n in clusters :
        clustering_print_scores(y_new,x_embedded_lle,n)

    #Isomap Evaluation#
    for n in clusters :
        clustering_print_scores(y_new,x_embedded_isomap,n)

    #MDS Evaluation#
    for n in clusters :
        clustering_print_scores(y_new,x_embedded_mds,n)

    'Lets see how our metrics vary depending in the number of our clusters'
    for i,k in zip(evaluation,titles):
        plot_metrics(i,k)



    'Lets import a test in order to classify new data based on our centers we found from KMeans Algorithm'

    'Because we cant use our tSNE algorithm in new data what we are gonna do is fit a multioutput regressor'
    'So that we can project our test data in 2 dimensions as we did with tSNE'
    'The same thing we are going to do with MDS algorithm'



    'TSNE implementation on test'
    x_tsne_test = regression(X_new,x_embedded_tsne,X_test)

    #Evaluation of test#
    model = KMeans(n_clusters = 4,random_state=0).fit(x_embedded_tsne)
    pred_classes = model.predict(x_tsne_test)
    scores_for_test(y_test,pred_classes,x_tsne_test)

    #Plotting our test#
    plot(x_tsne_test,y_test,title='TSNE test ',neighbors=50)


    'MDS implementation on test'
    x_mds_test = regression(X_new,x_embedded_mds,X_test)

    #Evaluation of test#
    model = KMeans(n_clusters = 4,random_state=0).fit(x_embedded_mds)
    pred_classes = model.predict(x_mds_test)
    scores_for_test(y_test,pred_classes,x_mds_test)

    #Plotting our test#
    plot(x_mds_test,y_test,title='MDS on test ')


    'LLE implementation on test'
    x_lle_test = embedding_lle.transform(X_test)
    #Plotting our test#
    plot(x_lle_test,y_test,title='LLE',neighbors = 20)
    #Evaluation of test#

    model = KMeans(n_clusters = 4,random_state=0).fit(x_embedded_lle)
    pred_classes = model.predict(x_lle_test)
    scores_for_test(y_test,pred_classes,x_lle_test)


    'Isomap implementation test'
    x_iso_test = embedding_iso.transform(X_test)

    #Evaluation of test#
    model = KMeans(n_clusters = 4,random_state=0).fit(x_embedded_isomap)
    pred_classes = model.predict(x_iso_test)
    scores_for_test(y_test,pred_classes,x_iso_test)

    #Plotting our test#
    plot(x_iso_test,y_test,title='ISOMAP ',neighbors=20)
