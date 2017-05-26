import sys

model = sys.argv[1]

test1_label_file = open('test1.label', 'r')
test1_label_all = test1_label_file.read()
test1_label = test1_label_all.split()
test1_label_file.close()
t1_size = len(test1_label)

test2_label_file = open('test2.label', 'r')
test2_label_all = test2_label_file.read()
test2_label = test2_label_all.split()
test2_label_file.close()
t2_size = len(test2_label)
tot = t1_size+t2_size

test1_file = 'test1_' + model + '.txt'
test1_class_file = open(test1_file, 'r')
test1_class_all = test1_class_file.read()
test1_class = test1_class_all.split()
test1_class_file.close()

test2_file = 'test2_' + model + '.txt'
test2_class_file = open(test2_file, 'r')
test2_class_all = test2_class_file.read()
test2_class = test2_class_all.split()
test2_class_file.close()

error_t1 = 0
for x in range(t1_size):
    if (test1_label[x] != test1_class[x]):
        error_t1+=1
print("Correct:", t1_size - error_t1, "Error:", error_t1, "Total:", t1_size)
print("CorrectRate:", (t1_size-error_t1)/t1_size)
print()
error_t2 = 0
for x in range(t2_size):
    if (test2_label[x] != test2_class[x]):
        error_t2+=1
print("Correct:", t2_size - error_t2, "Error:", error_t2, "Total:", t2_size)
print("CorrectRate:", (t2_size-error_t2)/t2_size)
print()
error_tot = error_t2+error_t1
print("TotalCorrect:", tot - error_tot, "Error:", error_tot, "Total:", tot)
print("TotCorrectRate:", (tot - error_tot)/tot)
