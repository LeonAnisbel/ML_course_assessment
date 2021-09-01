from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from Code import Plot_func as plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns


def var_declaration(n_K):
    names = []
    performance ={}
    neighbors = range(1, n_K)
    [names.append('neigh_'+str(i)) for i in neighbors]
    results ={}
    return names,performance,neighbors,results

def train_test_approach(X,Y,n_K):
    names,performance,neighbors,_ = var_declaration(n_K)
    performance = [{l:{}} for l in names]
    for i in range(len(neighbors)):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        KNN_Classifier = KNeighborsClassifier(n_neighbors = neighbors[i], p = 2, metric='minkowski')
        #Train the model
    #     %time
        KNN_Classifier.fit(x_train, y_train)

        #Let's predict the classes for test data
        pred_test = KNN_Classifier.predict(x_test)
        performance[i] = {'Train_score':
                         accuracy_score(y_train,KNN_Classifier.predict(x_train)),
                         'Test_score':
                         accuracy_score(y_test,KNN_Classifier.predict(x_test))}

    plot.accu_score(performance,neighbors)
    #     print(f"Performance(n_neighbors = {i}): ", perfomance,'\n')

    
def Kfold_approach(X,Y,n_K):
    # Evaluate using Cross Validation
    # kfold: Split a dataset into training and test sets
    names,performance,neighbors,results = var_declaration(n_K)
    k = 15
    for i in range(len(neighbors)):
        kfold = KFold(n_splits=k,random_state=7, shuffle=True)
        perf = []
        model =  KNeighborsClassifier(n_neighbors = neighbors[i], p = 2, metric='minkowski')
        for j,(ix_train, ix_test) in enumerate(kfold.split(Y)):
            model.fit(X[ix_train],Y[ix_train])
            perf.append(accuracy_score(Y[ix_test],model.predict(X[ix_test])))
        results[names[i]] = cross_val_score(model, X, Y, cv=kfold)
        performance[names[i]]= perf


    df = pd.DataFrame(performance.values(),index=performance.keys()).stack().reset_index()
    df.columns = ['model','k','performance']
    ax = sns.barplot(y="model", x="performance", data=df)
    plot.accu_score_CV(results,neighbors,names)
    

    
def mult_methods(X,Y,n_k):
    models = {
    "K-nearest neighbors n=8" :  KNeighborsClassifier(n_neighbors=n_k, p = 2, metric='minkowski'),
    "Logistic Regression" :LogisticRegression(solver='liblinear'),
    }


    k = 15
    perfomance = {}
    for i,(title,mdl) in enumerate(models.items()):
        kfold = KFold(n_splits=k,random_state=7, shuffle=True)
        _perf = []

#         %time
        for j,(ix_train, ix_test) in enumerate(kfold.split(Y)):
            mdl.fit(X[ix_train],Y[ix_train])
            _perf.append(accuracy_score(Y[ix_test],mdl.predict(X[ix_test])))
        results = cross_val_score(mdl, X, Y, cv=kfold)
        perfomance[title] = _perf

        print(mdl,'\n',"Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0),'\n')



    df = pd.DataFrame(perfomance.values(),index=perfomance.keys()).stack().reset_index()
    df.columns = ['model','k','perfomance']
    ax = sns.barplot(y="model", x="perfomance", data=df)
