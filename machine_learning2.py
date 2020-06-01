# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:25:58 2020

@author: lilliloo
"""

# ------------------------------ #
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn import metrics
# ------------------------------ #
#
#混同行列
#                     予測
#                positive        negative
#実　　positive   True positive   False negative 
#際   negative  False positive   True negative
#予測結果がtrue false
#                       TP + TN
#正答率(Accuracy)　＝　----------------
#                      TP+TN+FP+FN
#
#                TP
#適合率 =  -----------------
#(Precision)   TP + FP
#
#              TP
#再現率　＝　------------
#(Recall)    TP + FN
#
#F値　　　　　　　　　　　　　　　　　2
#適合率と再現率の =     ---------
#調和平均             1/precision + 1/recall

# ----------------------------------------------- #
#画像の読み込み
digits = datasets.load_digits()

flag_3_8 = (digits.target == 3) + (digits.target == 8)
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]
#1次元化
images = images.reshape(images.shape[0],-1)
#分類機の生成
n_sample = len(flag_3_8[flag_3_8])
train_size = int(n_sample * 6 / 10)
classifier = tree.DecisionTreeClassifier(max_depth = 3)
classifier.fit(images[:train_size], labels[:train_size])
#評価
expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])
print("Accuracy:\n",metrics.accuracy_score(expected, predicted))
print("Confusion Matrix:\n", metrics.confusion_matrix(expected, predicted))
print("Precision:\n",metrics.precision_score(expected, predicted,pos_label = 3))
print("Recall:\n",metrics.recall_score(expected, predicted,pos_label = 3))
print("F-measure:\n",metrics.f1_score(expected, predicted,pos_label = 3))



