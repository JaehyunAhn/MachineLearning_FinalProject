"""
    Machine Learning Final Project
    - Author : Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
    - Due Date : 15/06/16
"""
from scipy.io import loadmat
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from DataModule import *

"""
    Load Dataset
    - objectA : A label, dataset # 16
    - objectB : B label, dataset # 16
    - test_set : test set, which is unlabeled # 48
"""
print("Reading dataset from matlab...")
objectA = loadmat('./datang/objectA.mat')
objectB = loadmat('./datang/objectB.mat')
test_set = loadmat('./datang/test_set.mat')

# print(objectA['objectA'])
# print(objectB['objectB'])
# print(test_set['test_set'])
print("Data loaded.")

image_arr = []
image_arr2 = []
label_index = []
label_index2 = []
data_list = {'label': label_index, 'data': image_arr}
test_list = {'label': label_index2, 'data': image_arr2}

print("Collect training/testing dataset..\n")
train = data_list
train = collect_data(object_dataset=objectA['objectA'],
                     label_name='A',
                     target=data_list)
train = collect_data(object_dataset=objectB['objectB'],
                     label_name='B',
                     target=data_list)
test = test_list
test = collect_data(object_dataset=test_set['test_set'],
                    label_name='B',
                    target=test_list)

train_data = np.vstack(train['data'])
train_label = np.hstack(train['label'])
test_data = np.vstack(test['data'])

print("-TEST # DATASET: %d" % len(test_set['test_set']))

# Random Forest
print("RANDOM FOREST CLASSIFIER")
clf_rf = RandomForestClassifier()
clf_rf.fit(train_data, train_label)
y_pred_rf = clf_rf.predict(test_data)
print("-Random Forest Pred:\t", y_pred_rf)
scores = cross_val_score(clf_rf, train_data, train_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() / 2))

# Linear SVM
print("LINEAR SUPPORT VECTOR MACHINE")
clf_svm = LinearSVC()
clf_svm.fit(train_data, train_label)
y_pred_svm = clf_svm.predict(test_data)
print("-LSVM Pred:\t\t\t", y_pred_svm)
scores = cross_val_score(clf_svm, train_data, train_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() / 2))

# Support Vector Classifier
print("SUPPORT VECTOR CLASSIFIER")
clf_svc = SVC()
clf_svc.fit(train_data, train_label)
y_pred_svc = clf_svc.predict(test_data)
print("-SVC Pred:\t\t\t", y_pred_svc)
scores = cross_val_score(clf_svc, train_data, train_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() / 2))

# KNN Classifier
print("K-NEAREST NEIGHBOR CLASSIFIER")
clf_knn = KNeighborsClassifier()
clf_knn.fit(train_data, train_label)
y_pred_knn = clf_knn.predict(test_data)
print("-KNN Pred:\t\t\t", y_pred_knn)
scores = cross_val_score(clf_knn, train_data, train_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() / 2))

# LIBSVM is not supported in OS X (Windows Only)

print("Learning is done.\n")

print("<RF >= LSVC > SVC > K-NN>")
word = ''
for i in range(len(y_pred_rf)):
    word = word + y_pred_rf[i]
print(word)
word = ''
for i in range(len(y_pred_svc)):
    word = word + y_pred_svc[i]
print(word)
word = ''
for i in range(len(y_pred_svm)):
    word = word + y_pred_svm[i]
print(word)
word = ''
for i in range(len(y_pred_knn)):
    word = word + y_pred_knn[i]
print(word)
"""
Unscaled Priority

RF > SVC > LSVC > KNN

"""
