import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import sys
import os

dim_count_dict = dict()
# 1 more than original fea
dim_count_dict["MFSPMAP_MF"] = 1894
dim_count_dict["BPSPMAP_BP"] = 1862
dim_count_dict["CCSPMAP_CC"] = 1901

dim_count_dict["PAAC_MF"] = 51
dim_count_dict["PAAC_BP"] = 51
dim_count_dict["PAAC_CC"] = 51

dim_count_dict["APAAC_MF"] = 81
dim_count_dict["APAAC_BP"] = 81
dim_count_dict["APAAC_CC"] = 81

dim_count_dict["TPC_MF"] = 8001
dim_count_dict["TPC_BP"] = 8001
dim_count_dict["TPC_CC"] = 8001

dim_count_dict["QSO_MF"] = 101
dim_count_dict["QSO_BP"] = 101
dim_count_dict["QSO_CC"] = 101

dim_count_dict["CTriad_MF"] = 344
dim_count_dict["CTriad_BP"] = 344
dim_count_dict["CTriad_CC"] = 344


#dim_count_dict["sample_MF"] = 7

#feature_type = "sample"



path = "./.."
feature_path = "../FeatureVectors"
#category = "MF"
#go_term_file = "XXX"
#fold = 1

n_of_neurons_h1 = int(sys.argv[1])
n_of_neurons_h2 = int(sys.argv[2])
hm_epocs = int(sys.argv[3])
go_term_file = sys.argv[4]
category = sys.argv[5]
feature_type = sys.argv[6]
learn_rate = float(sys.argv[7])
mini_batch_size = int(sys.argv[8])
#adam,rms,moment
optimizer_type = sys.argv[9]
#input normalization yes or no
normalize_inputs = sys.argv[10]
#batch normalization
batch_normalization = sys.argv[11]
learning_rate_decay = sys.argv[12]
drop_out_rate = float(sys.argv[13])


#go_path = "%s/GOTermFiles/%s/subgroups" % (path, category)
go_path = "%s/GOTermFiles/%s/5" % (path,category)
#go_path = "%s/GOTermFiles/FirstRuns" % (path)

go_annots_path = "%s/TrainTestDatasets/%s" % (path, category)
# MFGOTerms30_2_201_300.txt
go_terms = open("%s/%s" % (go_path, go_term_file), "r")
#go_terms = open("../SampleFiles/sample_gos.txt", "r")
lst_go_terms = go_terms.read().split("\n")
go_terms.close()

if "" in lst_go_terms:
    lst_go_terms.remove("")

go_term_index_dict = dict()
go_term_list = []
for ind in range(len(lst_go_terms)):
    if lst_go_terms[ind] != "":
        #print(lst_go_terms[ind].split("\t")[0])
        # print(lst_go_terms[ind].split("\t")[0], "SADSA",ind)
        go_term_index_dict[lst_go_terms[ind].split("\t")[0]] = ind
        go_term_list.append(lst_go_terms[ind].split("\t")[0])

n_g_terms = len(go_term_list)

lst_all_test_prot_ids = set()
for test_fl in os.listdir(go_annots_path):
    if test_fl.startswith("test"):
        all_test_ids = open("%s/%s" % (go_annots_path, test_fl), "r")
        lst_all_test_ids = all_test_ids.read().split("\n")
        all_test_ids.close()
        while "" in lst_all_test_ids:
            lst_all_test_ids.remove("")
        for ln in lst_all_test_ids:
            lst_all_test_prot_ids.add(ln)

lst_all_test_prot_ids = list(lst_all_test_prot_ids)


#print(go_term_list)
# 1902936_positive.ids
# in the same order as test
lst_test_prot_ids = []
p_test_ids_dict = dict()
p_all_annots_dict = dict()
p_ids_dict = dict()
for go in go_term_index_dict.keys():

    pos_ids_file = open("%s/train_%s.ids" % (go_annots_path, go), "r")
    lst_pos_ids_file = pos_ids_file.read().split("\n")
    pos_ids_file.close()

    if "" in lst_pos_ids_file:
        lst_pos_ids_file.remove("")

    for line in lst_pos_ids_file:
        p_ids_dict[line] = 0
        try:
            p_all_annots_dict[line].add(go)
        except:
            p_all_annots_dict[line] = set()
            p_all_annots_dict[line].add(go)

    #tst_ids_file = open("../SampleFiles/sample_test_%s_positive.txt" % (go), "r")
    tst_ids_file = open("%s/test_%s.ids" % (go_annots_path, go), "r")
    lst_tst_ids_file = tst_ids_file.read().split("\n")
    tst_ids_file.close()

    if "" in lst_tst_ids_file:
        lst_tst_ids_file.remove("")

    for line in lst_tst_ids_file:
        #print(go,line)
        p_ids_dict[line] = 0
        try:
            p_all_annots_dict[line].add(go)
        except:
            p_all_annots_dict[line] = set()
            p_all_annots_dict[line].add(go)

problematic_prots = set()


# features each column is a feature vector for a sample
def normalizeFeatures(features):
    # Find means each row
    mean = features.mean(axis = 1,keepdims=True)
    column_mean = np.repeat(mean, features.shape[1], axis=1)
    mean_subtracted = features - column_mean
    variance = features.var(axis=1, keepdims = True)
    normalizedFeatures = mean_subtracted / variance
    return normalizedFeatures, mean, variance

# features each column is a feature vector for a sample
def normalizeFeatureswithMeanVariance(features,mean,variance):
    # Find means of columns
    column_mean = np.repeat(mean, features.shape[1], axis=1)
    mean_subtracted = features - column_mean
    normalizedFeatures = mean_subtracted / variance
    return normalizedFeatures

problematic_prots = set()


def createFeatureVectorsAndLabels():
    prot_feature_vector_dict = dict()
    count = 0

    prot_feature_vector_dict = dict()
    count = 0
    with open("%s/Parsed_%sFeatures_uniprot_training_test_set.txt" % (feature_path, feature_type)) as f:
    #with open("../SampleFiles/sample_features.txt") as f:
        for line in f:
            count += 1
            line = line.split("\n")[0].split("\t")
            if "" in line:
                line.remove("")
            prot_id = line[0]

            if len(line) != dim_count_dict[feature_type + "_" + category]:
                problematic_prots.add(prot_id)
            # print(prot_id)
            else:
                try:
                    p_all_annots_dict[prot_id]
                    feature_vector = [float(x) for x in line[1:]]
                    prot_feature_vector_dict[prot_id] = feature_vector
                except:
                    pass

    train_protids_feature_vectors_labels = dict()
    for go in go_term_index_dict.keys():
        pos_ids_file = open("%s/train_%s.ids" % (go_annots_path, go), "r")
        #pos_ids_file = open("../SampleFiles/sample_train_%s_positive.txt" %(go))
        lst_pos_ids_file = pos_ids_file.read().split("\n")
        pos_ids_file.close()

        if "" in lst_pos_ids_file:
            lst_pos_ids_file.remove("")
        for p_id in lst_pos_ids_file:
            try:
                prot_feature_vector_dict[p_id]
            except:
                #print("NO Prot", p_id)
                problematic_prots.add(p_id)

        for pp in problematic_prots:
            if pp in lst_pos_ids_file:
                lst_pos_ids_file.remove(pp)
        for pos_prot in lst_pos_ids_file:
            try:
                train_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[go]] = 1
            except:
                train_protids_feature_vectors_labels[pos_prot] = [prot_feature_vector_dict[pos_prot], [0] * len(go_term_index_dict.keys())]
                train_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[go]] = 1

    test_protids_feature_vectors_labels = dict()
    for go in go_term_index_dict.keys():
        #pos_ids_file = open("../SampleFiles/sample_test_%s_positive.txt" %(go))
        pos_ids_file = open("%s/test_%s.ids" % (go_annots_path, go), "r")
        lst_pos_ids_file = pos_ids_file.read().split("\n")
        pos_ids_file.close()

        if "" in lst_pos_ids_file:
            lst_pos_ids_file.remove("")

        for p_id in lst_pos_ids_file:
            try:
                prot_feature_vector_dict[p_id]
            except:
                #print("NO Prot",p_id)
                problematic_prots.add(p_id)

        for pp in problematic_prots:
            if pp in lst_pos_ids_file:
                # print(pp)
                lst_pos_ids_file.remove(pp)

        for pos_prot in lst_pos_ids_file:
            try:
                test_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[go]] = 1
            except:
                test_protids_feature_vectors_labels[pos_prot] = [prot_feature_vector_dict[pos_prot],[0] * len(go_term_index_dict.keys())]
                test_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[go]] = 1

    cafa_protids_feature_vectors_labels = dict()
    lst_cafa_ids = []
    with open("%s/Parsed_%sFeatures_CAFA2.txt" % (feature_path, feature_type)) as f:
        for line in f:
            count += 1
            line = line.split("\n")[0].split("\t")
            if "" in line:
                line.remove("")
            prot_id = line[0]
            lst_cafa_ids.append(prot_id)
            if len(line) != dim_count_dict[feature_type + "_" + category]:
                problematic_prots.add(prot_id)
            # print(prot_id)
            else:

                feature_vector = [float(x) for x in line[1:]]
                prot_feature_vector_dict[prot_id] = feature_vector

    if "" in lst_cafa_ids:
        lst_cafa_ids.remove("")

    for p_id in lst_cafa_ids:
        try:
            prot_feature_vector_dict[p_id]
        except:
            # print("NO Prot",p_id)
            problematic_prots.add(p_id)

    for pp in problematic_prots:
        if pp in lst_cafa_ids:
            # print(pp)
            lst_cafa_ids.remove(pp)

    for pos_prot in lst_cafa_ids:
        try:
            cafa_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[0]] = 1
        except:
            cafa_protids_feature_vectors_labels[pos_prot] = [prot_feature_vector_dict[pos_prot],
                                                             [0] * len(go_term_index_dict.keys())]
            #cafa_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[0]] = 1



    #-------------------
    all_test_protids_feature_vectors_labels = dict()
    lst_all_test_ids_in_order = []
    with open("%s/Parsed_%sFeatures_uniprot_training_test_set.txt" % (feature_path, feature_type)) as f:
        for line in f:
            count += 1
            line = line.split("\n")[0].split("\t")
            if "" in line:
                line.remove("")
            prot_id = line[0]
            if prot_id in lst_all_test_prot_ids:
                lst_all_test_ids_in_order.append(prot_id)
                if len(line) != dim_count_dict[feature_type + "_" + category]:
                    problematic_prots.add(prot_id)
                # print(prot_id)
                else:

                    feature_vector = [float(x) for x in line[1:]]
                    prot_feature_vector_dict[prot_id] = feature_vector

    while "" in lst_all_test_ids_in_order:
        lst_all_test_ids_in_order.remove("")

    for p_id in lst_all_test_ids_in_order:
        try:
            prot_feature_vector_dict[p_id]
        except:
            # print("NO Prot",p_id)
            problematic_prots.add(p_id)

    for pp in problematic_prots:
        if pp in lst_all_test_ids_in_order:
            # print(pp)
            lst_all_test_ids_in_order.remove(pp)

    for pos_prot in lst_all_test_ids_in_order:
        try:
            all_test_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[0]] = 1
        except:
            all_test_protids_feature_vectors_labels[pos_prot] = [prot_feature_vector_dict[pos_prot],
                                                             [0] * len(go_term_index_dict.keys())]
            # cafa_protids_feature_vectors_labels[pos_prot][1][go_term_index_dict[0]] = 1


    #------------------------



    train_feature_vectors = []
    train_labels = []
    train_prot_ids = []

    for prot_id in train_protids_feature_vectors_labels.keys():
        train_prot_ids.append(prot_id)
        train_feature_vectors.append(train_protids_feature_vectors_labels[prot_id][0])
        train_labels.append(train_protids_feature_vectors_labels[prot_id][1])

    test_feature_vectors = []
    test_labels = []
    test_prot_ids = []

    for prot_id in test_protids_feature_vectors_labels.keys():
        test_prot_ids.append(prot_id)
        test_feature_vectors.append(test_protids_feature_vectors_labels[prot_id][0])
        test_labels.append(test_protids_feature_vectors_labels[prot_id][1])


    cafa_feature_vectors = []
    cafa_labels = []
    cafa_prot_ids = []

    for prot_id in cafa_protids_feature_vectors_labels.keys():
        cafa_prot_ids.append(prot_id)
        cafa_feature_vectors.append(cafa_protids_feature_vectors_labels[prot_id][0])
        cafa_labels.append(cafa_protids_feature_vectors_labels[prot_id][1])

    test_all_feature_vectors = []
    test_all_labels = []
    test_all_prot_ids = []

    for prot_id in all_test_protids_feature_vectors_labels.keys():
        test_all_prot_ids.append(prot_id)
        test_all_feature_vectors.append(all_test_protids_feature_vectors_labels[prot_id][0])
        test_all_labels.append(all_test_protids_feature_vectors_labels[prot_id][1])



    train_feature_vectors = np.array(train_feature_vectors).T
    #print(train_feature_vectors.shape)
    train_labels = np.array(train_labels).T
    train_prot_ids = np.array(train_prot_ids)
    train_feature_vectors, mean, variance = normalizeFeatures(train_feature_vectors)

    test_feature_vectors = np.array(test_feature_vectors).T
    test_labels = np.array(test_labels).T
    test_prot_ids = np.array(test_prot_ids)
    test_feature_vectors = normalizeFeatureswithMeanVariance(test_feature_vectors,mean,variance)

    cafa_feature_vectors = np.array(cafa_feature_vectors).T
    cafa_labels = np.array(cafa_labels).T
    cafa_prot_ids = np.array(cafa_prot_ids)
    cafa_feature_vectors = normalizeFeatureswithMeanVariance(cafa_feature_vectors, mean, variance)


    test_all_feature_vectors = np.array(test_all_feature_vectors).T
    test_all_labels = np.array(test_all_labels).T
    test_all_prot_ids = np.array(test_all_prot_ids)
    test_all_feature_vectors = normalizeFeatureswithMeanVariance(test_all_feature_vectors, mean, variance)


    # maybe normalize first
    #normalized_features = normalizeFeatures(features)

    return train_feature_vectors, train_labels, train_prot_ids, test_feature_vectors, test_labels, test_prot_ids, cafa_feature_vectors, cafa_labels, cafa_prot_ids, test_all_feature_vectors, test_all_labels, test_all_prot_ids




X_train, Y_train, train_prot_ids, X_test, Y_test, test_prot_ids, X_cafa, Y_cafa, cafa_prot_ids, X_all_test, Y_all_test, test_all_prot_ids  = createFeatureVectorsAndLabels()

#print(Y_train)
#print(train_prot_ids)
#print(X_cafa)
#print(cafa_prot_ids)

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))


    return X, Y

def initialize_parameters(n_x,n_h1,n_h2,n_y):

    W1 = tf.get_variable("W1", [n_h1, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [n_h1, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [n_h2, n_h1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [n_h2, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y, n_h2], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def compute_cost(A3, Y, parameters):
    # Computes the cost
    # Returns:
    # cost - Tensor of the cost function
    
    logits = A3
    labels = Y

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) + 0.001 * tf.nn.l2_loss(parameters['W1']) + 0.001 * tf.nn.l2_loss(parameters['W2']))


    return cost



def forward_propagation(X, parameters,p_keep_hidden=drop_out_rate):

    # Retrieve the parameters from the dictionary "parameters"
    # Small epsilon value for the BN transform
    epsilon = 1e-3

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.matmul(W1, X) + b1
    A1 = None

    if batch_normalization == "yes":
        #print("Batch Normalization")
        batch_mean1, batch_var1 = tf.nn.moments(Z1, [0])
        scale1 = tf.Variable(tf.ones(n_of_neurons_h2))
        beta1 = tf.Variable(tf.zeros(n_of_neurons_h2))
        BN1 = tf.nn.batch_normalization(Z1, batch_mean1, batch_var1, None, None, epsilon)
        A1 = tf.nn.relu(BN1)
    else:
        A1 = tf.nn.relu(Z1)

    A1 = tf.nn.dropout(A1, p_keep_hidden)


    Z2 = tf.matmul(W2, A1) + b2
    A2 = None

    if batch_normalization == "yes":
        batch_mean2, batch_var2 = tf.nn.moments(Z2, [0])
        scale2 = tf.Variable(tf.ones(n_of_neurons_h2))
        beta2 = tf.Variable(tf.zeros(n_of_neurons_h2))
        BN2 = tf.nn.batch_normalization(Z2, batch_mean2, batch_var2, None, None, epsilon)
        A2 = tf.nn.relu(BN2)
    else:
        A2 = tf.nn.relu(Z2)

    A2 = tf.nn.dropout(A2, p_keep_hidden)

    A3 = tf.matmul(W3, A2) + b3

    return A3

def calculateFscore(tp,fp,fn):
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fscore = (2*precision*recall)/(precision+recall+1e-08)
    return fscore



def model(X_train, Y_train, X_test, Y_test, learning_rate=learn_rate, num_epochs=hm_epocs, minibatch_size=mini_batch_size, print_cost=True):


    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    p_keep_hidden = tf.placeholder(tf.float32)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost
    #print(n_x,n_y,m)
    # Create Placeholders of shape (n_x, n_y)

    X, Y = create_placeholders(n_x, n_y)


    # Initialize parameters

    parameters = initialize_parameters(n_x,n_of_neurons_h1,n_of_neurons_h2,n_y)


    # Forward propagation: Build the forward propagation in the tensorflow graph

    A3 = forward_propagation(X, parameters, p_keep_hidden)


    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(A3, Y,parameters)

    # whether learning rate deacay will be used or not
    if  learning_rate_decay=="yes":
        global_step = tf.Variable(0, trainable=True)
        learning_rate = tf.train.exponential_decay(learn_rate, global_step,
                                               100, 0.96, staircase=True)
    else:
        learning_rate = learn_rate

    #the tensorflow optimizer.
    optimizer = None
    if optimizer_type=="adam":
        #print("adam")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    elif optimizer_type == "moment":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=.9).minimize(cost)
    elif optimizer_type == "rms":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            #print(m,minibatch_size,num_minibatches)
            i = 0
            while i < m:
                start = i
                end = i + minibatch_size
                batch_x = X_train[:,start:end]
                batch_y = Y_train[:,start:end]
                #print(batch_x)
                #print(batch_y)
                #print(drop_out_rate)
                _, c = sess.run([optimizer, cost],
                                feed_dict={X: batch_x, Y: batch_y, p_keep_hidden:drop_out_rate})
                epoch_loss += c
                i += minibatch_size
                epoch_cost += c
                costs.append(epoch_cost)
            if epoch % 100 == 0:
                print('Epoch', epoch, 'completed out of ', num_epochs, 'loss:', epoch_cost)

        threshold = 1.00
        train_preds_arr = A3.eval({X: X_train, Y: Y_train, p_keep_hidden: 1.0}, session=sess)
        test_preds_arr = A3.eval({X: X_test, Y: Y_test, p_keep_hidden: 1.0}, session=sess)
        cafa_preds_arr = A3.eval({X: X_cafa, Y: Y_cafa, p_keep_hidden: 1.0}, session=sess)
        all_test_preds_arr = A3.eval({X: X_all_test, Y: Y_all_test, p_keep_hidden: 1.0}, session=sess)
        print("--")

        cafa_id_str = ""
        for id in cafa_prot_ids:
            cafa_id_str += id +"\t"
        print(cafa_id_str)
        for line in cafa_preds_arr:
            result_str = ""
            for item in line:
                result_str +=str(item) +"\t"

            print(result_str)

        print("--")

        test_id_str = ""
        for id in test_all_prot_ids:
            test_id_str += id + "\t"
        print(test_id_str)
        for line in all_test_preds_arr:
            result_str = ""
            for item in line:
                result_str += str(item) + "\t"

            print(result_str)

        print("--")
        # threshold, train, test, average
        train_best = [0.0, 0.0, 0.0, 0.0]
        test_best = [0.0, 0.0, 0.0, 0.0]
        average_best = [0.0, 0.0, 0.0, 0.0]

        #print(train_preds_arr)
        while threshold>0.00:
            pos_train_preds = np.where(train_preds_arr > threshold, 1, 0)
            pos_test_preds = np.where(test_preds_arr > threshold, 1, 0)

            train_tps = np.sum(np.bitwise_and(pos_train_preds, Y_train))
            train_fps = np.sum(pos_train_preds) - train_tps
            train_fns = np.sum(Y_train) - train_tps
            train_fscore = calculateFscore(train_tps,train_fps,train_fns)
            #print(threshold, train_tps, train_fps, train_fns)

            test_tps = np.sum(np.bitwise_and(pos_test_preds, Y_test))
            test_fps = np.sum(pos_test_preds) - test_tps
            test_fns = np.sum(Y_test) - test_tps
            test_fscore = calculateFscore(test_tps, test_fps, test_fns)
            #print(threshold, test_tps, test_fps, test_fns)

            temp_average = (train_fscore+test_fscore)/2

            if train_fscore> train_best[1]:
                train_best = [threshold, train_fscore, test_fscore, temp_average]

            if test_fscore > test_best[2]:
                test_best = [threshold, train_fscore, test_fscore, temp_average]

            if temp_average> average_best[3]:
                average_best = [threshold, train_fscore, test_fscore, temp_average]

            threshold -= 0.01

        print("Threshold with the best training score:")
        print("%f\t%f\t%f\t%f" %(train_best[0],train_best[1],train_best[2], train_best[3]))

        print("Threshold with the best test score:")
        print("%f\t%f\t%s\t%f" % (test_best[0], test_best[1], test_best[2], test_best[3]))

        print("Threshold with the best average score:")
        print("%f\t%f\t%s\t%f" % (average_best[0], average_best[1], average_best[2], average_best[3]))



        """
        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(A3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        """
    return parameters

parameters = model(X_train, Y_train, X_test, Y_test)
