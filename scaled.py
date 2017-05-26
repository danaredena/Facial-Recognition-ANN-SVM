import numpy as np
import os
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from scipy import ndimage
from scipy import misc

#=========================== READING ===========================
train_label_file = open('train.label', 'r')
train_label_all = train_label_file.read()
train_label = train_label_all.split()
train_label_file.close()

train_path_file = open('train.list', 'r')
train_path_all = train_path_file.read()
train_path = train_path_all.split()
train_path_file.close()

train_list = []
script_dir = os.path.dirname(__file__)
for x in range(len(train_path)):
    dir_path = os.path.join(script_dir, train_path[x])
    train_face = misc.imread(dir_path)
    array = np.array(train_face)
    arr_view = array.ravel()
    scaled = []
    for y in range(len(arr_view)):
        scaled.append(arr_view[y]/255)
    train_list.append(scaled)

test1_label_file = open('test1.label', 'r')
test1_label_all = test1_label_file.read()
test1_label = test1_label_all.split()
test1_label_file.close()

test1_path_file = open('test1.list', 'r')
test1_path_all = test1_path_file.read()
test1_path = test1_path_all.split()
test1_path_file.close()

test1_list = []
for x in range(len(test1_path)):
    dir_path = os.path.join(script_dir, test1_path[x])
    test1_face = misc.imread(dir_path)
    array = np.array(test1_face)
    arr_view = array.ravel()
    scaled = []
    for y in range(len(arr_view)):
        scaled.append(arr_view[y]/255)
    test1_list.append(scaled)

test2_label_file = open('test2.label', 'r')
test2_label_all = test2_label_file.read()
test2_label = test2_label_all.split()
test2_label_file.close()

test2_path_file = open('test2.list', 'r')
test2_path_all = test2_path_file.read()
test2_path = test2_path_all.split()
test2_path_file.close()

test2_list = []
for x in range(len(test2_path)):
    dir_path = os.path.join(script_dir, test2_path[x])
    test2_face = misc.imread(dir_path)
    array = np.array(test2_face)
    arr_view = array.ravel()
    scaled = []
    for y in range(len(arr_view)):
        scaled.append(arr_view[y]/255)
    test2_list.append(scaled)

features_size = len(test2_list[0])
class_size = 20

#========================== ANN =======================
print('trainstart')
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,),max_iter=200000,alpha=1e-5)
ann = ann.fit(train_list, train_label)

#classification: test1
test1_class = ann.predict(test1_list)
filename = 'test1_scaled_ann.txt'
ann_test1 = open(filename,'w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
filename = 'test2_scaled_ann.txt'
ann_test2 = open(filename,'w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()

#========================== SVM =======================
'''
svc_l = svm.SVC(kernel='linear')
svc_l = svc_l.fit(train_list, train_label)

#classification
test1_class = svc_l.predict(test1_list)
filename = 'test1_scaled_svm.txt'
svc_l_test1 = open(filename,'w')
for y in range(len(test1_class)):
    svc_l_test1.write(str(test1_class[y])+'\n')
svc_l_test1.close()

#classification
test2_class = svc_l.predict(test2_list)
filename = 'test2_scaled_svm.txt'
svc_l_test2 = open(filename,'w')
for y in range(len(test2_class)):
    svc_l_test2.write(str(test2_class[y])+'\n')
svc_l_test2.close()
'''
