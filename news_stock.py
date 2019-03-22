#到处需要用到的库
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

#导入数据
data = pd.read_csv('d:/input/Combined_News_DJIA.csv')
print(data.head())  #查看下数据什么样子

#整合headlines
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

#分割测试/训练集
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

#提取features
feature_extraction = TfidfVectorizer()
X_train = feature_extraction.fit_transform(train["combined_news"].values)
X_test = feature_extraction.transform(test["combined_news"].values)
y_train = train["Label"].values
y_test = test["Label"].values

#训练模型
clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)

#预测
predictions = clf.predict_proba(X_test)

#验证准确度
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))
#ROC-AUC yields 0.574260752688