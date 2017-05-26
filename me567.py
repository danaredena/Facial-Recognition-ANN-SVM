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
    train_list.append(arr_view)

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
    test1_list.append(arr_view)

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
    test2_list.append(arr_view)

features_size = len(test2_list[0])
class_size = 20
#=========================== DECISION TREE ===========================
'''
# training
dec_tree = tree.DecisionTreeClassifier()
dec_tree = dec_tree.fit(train_list, train_label)
tree.export_graphviz(dec_tree,out_file='tree.dot')

#import pydot
#(graph,) = pydot.graph_from_dot_file('tree.dot')
#graph.write_png('tree.png')

#classification: test1
test1_class = dec_tree.predict(test1_list)
dec_tree_test1 = open('test1_dec_tree.txt','w')
for y in range(len(test1_class)):
    dec_tree_test1.write(str(test1_class[y])+'\n')
dec_tree_test1.close()

#classification: test2
test2_class = dec_tree.predict(test2_list)
dec_tree_test2 = open('test2_dec_tree.txt','w')
for y in range(len(test2_class)):
    dec_tree_test2.write(str(test2_class[y])+'\n')
dec_tree_test2.close()
'''
#=========================== ARTIFICIAL NEURAL NETWORK ===========================
'''
# training
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(class_size,),max_iter=200000)
ann = ann.fit(train_list, train_label)

#classification: test1
test1_class = ann.predict(test1_list)
ann_test1 = open('test1_ann_def.txt','w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
ann_test2 = open('test2_ann_def.txt','w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()
'''
'''
# hidden nodes
for x in range(class_size, features_size, 100):
    print(x)
    # training
    ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(x,),max_iter=200000)
    ann = ann.fit(train_list, train_label)

    #classification: test1
    test1_class = ann.predict(test1_list)
    filename = 'test1_ann' + str(x) + '.txt'
    ann_test1 = open(filename,'w')
    for y in range(len(test1_class)):
        ann_test1.write(str(test1_class[y])+'\n')
    ann_test1.close()

    #classification: test2
    test2_class = ann.predict(test2_list)
    filename = 'test2_ann' + str(x) + '.txt'
    ann_test2 = open(filename,'w')
    for y in range(len(test2_class)):
        ann_test2.write(str(test2_class[y])+'\n')
    ann_test2.close()
'''
'''
#chosen hidden nodes = 420
lrate = [1e-5,1e-4,1e-3,1e-2,1e-1]

for x in range(len(lrate)):
    print(lrate[x])
    # training
    ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,),max_iter=200000,alpha=lrate[x])
    ann = ann.fit(train_list, train_label)

    #classification: test1
    test1_class = ann.predict(test1_list)
    filename = 'test1_ann_lr:' + str(x) + '.txt'
    ann_test1 = open(filename,'w')
    for y in range(len(test1_class)):
        ann_test1.write(str(test1_class[y])+'\n')
    ann_test1.close()

    #classification: test2
    test2_class = ann.predict(test2_list)
    filename = 'test2_ann_lr:' + str(x) + '.txt'
    ann_test2 = open(filename,'w')
    for y in range(len(test2_class)):
        ann_test2.write(str(test2_class[y])+'\n')
    ann_test2.close()
'''
'''
x = 1
print(x)
# training
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,),max_iter=200000)
ann = ann.fit(train_list, train_label)
print("training done")
#classification: test1
test1_class = ann.predict(test1_list)
print("t1 done")
filename = 'test1_ann_layer:' + str(x) + '.txt'
ann_test1 = open(filename,'w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
print("t2 done")
filename = 'test2_ann_layer:' + str(x) + '.txt'
ann_test2 = open(filename,'w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()

x = 2
print(x)
# training
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,420),max_iter=200000)
ann = ann.fit(train_list, train_label)
print("training done")
#classification: test1
test1_class = ann.predict(test1_list)
print("t1 done")
filename = 'test1_ann_layer:' + str(x) + '.txt'
ann_test1 = open(filename,'w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
print("t2 done")
filename = 'test2_ann_layer:' + str(x) + '.txt'
ann_test2 = open(filename,'w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()

x = 3
print(x)
# training
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,420,420),max_iter=200000)
ann = ann.fit(train_list, train_label)
print("training done")
#classification: test1
test1_class = ann.predict(test1_list)
print("t1 done")
filename = 'test1_ann_layer:' + str(x) + '.txt'
ann_test1 = open(filename,'w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
print("t2 done")
filename = 'test2_ann_layer:' + str(x) + '.txt'
ann_test2 = open(filename,'w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()

x = 4
print(x)
# training
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,420,420,420),max_iter=200000)
ann = ann.fit(train_list, train_label)
print("training done")
#classification: test1
test1_class = ann.predict(test1_list)
print("t1 done")
filename = 'test1_ann_layer:' + str(x) + '.txt'
ann_test1 = open(filename,'w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
print("t2 done")
filename = 'test2_ann_layer:' + str(x) + '.txt'
ann_test2 = open(filename,'w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()

x = 5
print(x)
# training
ann = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(420,420,420,420,420),max_iter=200000)
ann = ann.fit(train_list, train_label)
print("training done")
#classification: test1
test1_class = ann.predict(test1_list)
print("t1 done")
filename = 'test1_ann_layer:' + str(x) + '.txt'
ann_test1 = open(filename,'w')
for y in range(len(test1_class)):
    ann_test1.write(str(test1_class[y])+'\n')
ann_test1.close()

#classification: test2
test2_class = ann.predict(test2_list)
print("t2 done")
filename = 'test2_ann_layer:' + str(x) + '.txt'
ann_test2 = open(filename,'w')
for y in range(len(test2_class)):
    ann_test2.write(str(test2_class[y])+'\n')
ann_test2.close()
'''
#=========================== SUPPORT VECTOR MACHINES ===========================]
'''
####### SVC with Linear Kernel
print('linear')
svc_l = svm.SVC(kernel='linear')
svc_l = svc_l.fit(train_list, train_label)

#classification
test1_class = svc_l.predict(test1_list)
filename = 'test1_svc_l.txt'
svc_l_test1 = open(filename,'w')
for y in range(len(test1_class)):
    svc_l_test1.write(str(test1_class[y])+'\n')
svc_l_test1.close()

#classification
test2_class = svc_l.predict(test2_list)
filename = 'test2_svc_l.txt'
svc_l_test2 = open(filename,'w')
for y in range(len(test2_class)):
    svc_l_test2.write(str(test2_class[y])+'\n')
svc_l_test2.close()

###### SVC with RBF Kernel
print('rbf')
svc_rbf = svm.SVC(kernel='rbf')
svc_rbf = svc_rbf.fit(train_list, train_label)

#classification
test1_class = svc_rbf.predict(test1_list)
filename = 'test1_svc_rbf.txt'
svc_rbf_test1 = open(filename,'w')
for y in range(len(test1_class)):
    svc_rbf_test1.write(str(test1_class[y])+'\n')
svc_rbf_test1.close()

#classification
test2_class = svc_rbf.predict(test2_list)
filename = 'test2_svc_rbf.txt'
svc_rbf_test2 = open(filename,'w')
for y in range(len(test2_class)):
    svc_rbf_test2.write(str(test2_class[y])+'\n')
svc_rbf_test2.close()

###### SVC with Polynomial Kernel
print('poly')
svc_poly = svm.SVC(kernel='poly')
svc_poly = svc_poly.fit(train_list, train_label)

#classification
test1_class = svc_poly.predict(test1_list)
filename = 'test1_svc_poly.txt'
svc_poly_test1 = open(filename,'w')
for y in range(len(test1_class)):
    svc_poly_test1.write(str(test1_class[y])+'\n')
svc_poly_test1.close()

#classification
test2_class = svc_poly.predict(test2_list)
filename = 'test2_svc_poly.txt'
svc_poly_test2 = open(filename,'w')
for y in range(len(test2_class)):
    svc_poly_test2.write(str(test2_class[y])+'\n')
svc_poly_test2.close()

###### SVC with sigmoid Kernel
print('sig')
svc_sig = svm.SVC(kernel='sigmoid')
svc_sig = svc_sig.fit(train_list, train_label)

#classification
test1_class = svc_sig.predict(test1_list)
filename = 'test1_svc_sig.txt'
svc_sig_test1 = open(filename,'w')
for y in range(len(test1_class)):
    svc_sig_test1.write(str(test1_class[y])+'\n')
svc_sig_test1.close()

#classification
test2_class = svc_sig.predict(test2_list)
filename = 'test2_svc_sig.txt'
svc_sig_test2 = open(filename,'w')
for y in range(len(test2_class)):
    svc_sig_test2.write(str(test2_class[y])+'\n')
svc_sig_test2.close()


###### SVC with Polynomial Kernel - VARYING
for x in range(1,11):
    print(x)
    svc_poly = svm.SVC(kernel='poly', degree=x)
    svc_poly = svc_poly.fit(train_list, train_label)

    #classification
    test1_class = svc_poly.predict(test1_list)
    filename = 'test1_svc_poly' + str(x) + '.txt'
    svc_poly_test1 = open(filename,'w')
    for y in range(len(test1_class)):
        svc_poly_test1.write(str(test1_class[y])+'\n')
    svc_poly_test1.close()

    #classification
    test2_class = svc_poly.predict(test2_list)
    filename = 'test2_svc_poly' + str(x) + '.txt'
    svc_poly_test2 = open(filename,'w')
    for y in range(len(test2_class)):
        svc_poly_test2.write(str(test2_class[y])+'\n')
    svc_poly_test2.close()
'''
