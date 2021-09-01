import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform,pdist
import seaborn as sns
import pandas as pd


def plot_pdist(X,annot=True):
    fig,ax = plt.subplots(1,3,figsize=(25,4))
    for i,dist in zip(range(4),['cosine','euclidean', 'cityblock']):
        sns.heatmap(squareform(pdist(X.T,metric=dist).round(2)),ax=ax[i], annot=annot, fmt="0.2f")
        ax[i].set_title(dist)
        

def accu_score(performance,n):
    Train_score = []
    Test_score  = []
    for i in range(len(n)):
        Train_score.append(performance[i]['Train_score'])
        Test_score.append(performance[i]['Test_score'])
   
    print(f'The number of neighbours in the KN Classifier with the highest testing accuracy ({round(max(Test_score)*100.0,3)}%) is: \n', Test_score.index(max(Test_score))+1)


    fig, ax = plt.subplots(1,1,figsize=(6, 4)) 
    plt.plot(n, Train_score,color = 'r')
    plt.plot(n, Test_score,color = 'b')
    plt.legend(['Train_score','Test_score'])
    plt.xlabel('Value of K for KN Classifier')
    plt.ylabel('Training and Testing Accuracy')
    
    
    
def accu_score_CV(performance,n,names):
    Test_score, Scores  = [],[]
    
    for i in performance:
        score = performance[i]
        Scores.append([score.mean()*100.0, score.std()*100.0])
        Test_score.append(score.mean())
        
        
        
    fig, ax = plt.subplots(1,2,figsize=(12, 4)) 
    ax[0].plot(n, Test_score,color = 'b')
    ax[0].set_xlabel('Value of K for KN Classifier')
    ax[0].set_ylabel('Testing Accuracy')
    
    column_labels=["Accuracy", "Std"]
    
    df=pd.DataFrame(Scores,index = names, columns=column_labels)
    ax[1].table(cellText=df.round(2).values,colLabels=df.columns,
                rowLabels=df.index,bbox=(0.4, 0, 0.65, 1.1))
    ax[1].axis('off')

    maxi = max(Test_score)
    print(f'The number of neighbours in the KN Classifier with the highest accuracy  ({round((maxi)*100.0,3)}%) is: \n', \
          Test_score.index(maxi)+1)