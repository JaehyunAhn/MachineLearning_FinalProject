"""
    Machine Learning Final Project
    - Author : Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
    - Due Date : 15/06/16
"""

from scipy.io import loadmat
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from DataModule import *

"""
    Load Dataset
    - objectA : A label, dataset # 16
    - objectB : B label, dataset # 16
    - test_set : test set, which is unlabeled # 48
"""
objectA = loadmat('./datang/objectA.mat')
objectB = loadmat('./datang/objectB.mat')
test_set = loadmat('./datang/test_set.mat')

# print(objectA['objectA'])
# print(objectB['objectB'])
# print(test_set['test_set'])

image_arr = []
image_arr2 = []
label_index = []
label_index2 = []
data_list = {'label': label_index, 'data': image_arr}
test_list = {'label': label_index2, 'data': image_arr2}

train = data_list
train = collect_data(object_dataset=objectA['objectA'],
                     label_name=0,
                     target=data_list)
train = collect_data(object_dataset=objectB['objectB'],
                     label_name=1,
                     target=data_list)
test = test_list
test = collect_data(object_dataset=test_set['test_set'],
                    label_name=2,
                    target=test_list)

train_data = np.vstack(train['data'])
train_label = np.hstack(train['label'])
test_data = np.vstack(test['data'])

# Random Forest
clf_rf = RandomForestClassifier()
clf_rf.fit(train_data, train_label)
y_pred_rf = clf_rf.predict(test_data)
print("Random Forest Array: ", y_pred_rf)

# Linear SVM
clf_svm = LinearSVC()
clf_svm.fit(train_data, train_label)
y_pred_svm = clf_svm.predict(test_data)
print("LSVM Array: ", y_pred_rf)

# KNN Classifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(train_data, train_label)
y_pred_knn = clf_knn.predict(test_data)
print("KNN Array: ", y_pred_knn)

"""
RF, LSVM, KNN
[0 1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 1 0 0 1 1 1 0 1 0 1 1 1 0 0]
[0 1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 1 0 0 1 1 1 0 1 0 1 1 1 0 0]
[0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 1 1 1 0 0]
PREDICTION
[0 1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 1 0 0 1 1 1 0 1 0 1 1 1 0 0]
"""
