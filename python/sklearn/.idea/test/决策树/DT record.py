from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
import pydot

wine=load_wine()
print(pd.DataFrame(wine.data))
print(pd.DataFrame(wine.data).columns.values)


df=pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3)

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(Xtrain,Ytrain)
score=clf.score(Xtest,Ytest)
print(score)

feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
dot_data=tree.export_graphviz(
    clf,feature_names=feature_name,class_names=['琴酒','雪梨','贝尔摩德'],filled=True,rounded=True,out_file=None
)
graph=graphviz.Source(dot_data,encoding='UTF-8')
graph.render()