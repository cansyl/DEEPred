from random import shuffle
import os
import sys

category = sys.argv[1]
path = "../TrainTestDatasets/%s" % (category)
fold_path = "%s/TrainingFolds" % (path)
n_fold = 10
count = 0
# test_GO:0051287_positive.ids  train_GO:0051287_positive.ids
for fl in os.listdir(path):
    if fl.startswith("train"):
        count += 1
        pos_fl = open("%s/%s" % (path, fl), "r")
        lst_pos_fl = pos_fl.read().split("\n")
        pos_fl.close()
        second_part = fl.split("train")[1]
        tst_fl_name = "test%s" % (second_part)

        #test_fl = open("%s/%s" % (path, tst_fl_name), "r")
        #lst_test_fl = test_fl.read().split("\n")
        #test_fl.close()

        #if "" in lst_test_fl:
        #    lst_test_fl.remove("")
        #if "" in lst_pos_fl:
        #    lst_pos_fl.remove("")
        #print(fl, len(lst_pos_fl), len(lst_test_fl))
        #lst_pos_fl.extend(lst_test_fl)
        #print(len(lst_pos_fl))
        #shuffle(lst_pos_fl)

        pos_fold_size = int(len(lst_pos_fl) / n_fold)
        print(len(lst_pos_fl), pos_fold_size)
        for i in range(n_fold):
            #print("%s/fold_%s_%s" % (fold_path, str(i + 1), fl))
            #print("%s/fold_%s_%s" % (fold_path, str(i + 1), tst_fl_name))
            """
            pos_train_fold_fl = open("%s/fold_%s_%s" % (fold_path, str(i + 1), fl), "w")
            pos_test_fold_fl = open("%s/fold_%s_%s" % (fold_path, str(i + 1), tst_fl_name), "w")
            for j in range(n_fold):
                # print(i,j)
                # test
                if i == j:
                    if (j + 1) == n_fold:
                        for c in lst_pos_fl[i * pos_fold_size:]:
                            pos_test_fold_fl.write(c + "\n")
                    else:
                        for c in lst_pos_fl[i * pos_fold_size:(i + 1) * pos_fold_size]:
                            pos_test_fold_fl.write(c + "\n")
                # train
                else:
                    if (j + 1) == n_fold:
                        for c in lst_pos_fl[j * pos_fold_size:]:
                            pos_train_fold_fl.write(c + "\n")
                    else:
                        for c in lst_pos_fl[j * pos_fold_size:(j + 1) * pos_fold_size]:
                            pos_train_fold_fl.write(c + "\n")
            pos_train_fold_fl.close()
            pos_test_fold_fl.close()
            """
print(count)
