import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics

random.seed(123)

def get_class(sample):
    if sample[0]**2 + sample[1]**2 > 2:
        if sample[0] < 0:
            return 0
        else:
            return 1
    else:
        if sample[0] < 0:
            return 2
        else:
            return 3
# def get_class(sample):
#     if sample[0] > 0:
#         if sample[1] > 0:
#             return 0
#         else:
#             return 1
#     else:
#         if sample[1] > 0:
#             return 2
#         else:
#             return 3

def generate_samples(length):
    samples = np.ndarray(shape=(length, 2), dtype=np.float32)
    labels = np.ndarray(shape=(length), dtype=np.int32)
    for i in range(0, length):
        samples[i][0] = random.gauss(0, 2)
        samples[i][1] = random.gauss(0, 2)
        labels[i] = get_class(samples[i])
    return samples, labels

def report_failures(prd_val, val_labels):
    for i in np.where(prd_val != val_labels)[0]:
        sample = val_samples[i]
        print("Sample %s, x1^2 + x2^2=%d, classification=%s should have been %s" % (
        sample, sample[0] ** 2 + sample[1] ** 2, prd_val[i], get_class(sample)))


tr_samples, tr_labels = generate_samples(10000)
val_samples, val_labels = generate_samples(1000)

#plt.scatter(tr_samples[:, 0], tr_samples[:, 1], marker='o', c=tr_labels)
#matplotlib.pyplot.show()


clf = svm.SVC()

# This is the most important line. I just feed the model training samples and it learns
clf.fit(tr_samples, tr_labels)

# This is prediction on training set
prd_tr = clf.predict(tr_samples)
print(float(sum(prd_tr == tr_labels))/prd_tr.shape[0])

prd_val = clf.predict(val_samples)
print(metrics.classification_report(val_labels, prd_val))

# report_failures(prd_val, val_labels)
# print(clf.support_vectors_.shape)

# ***************************** Neural network ************************************************************
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adagrad

no_classes = 4

def reformat_lables(labels):
    return (np.arange(no_classes) == labels[:,None]).astype(np.float32)

model = Sequential()
model.add(Dense(output_dim=16, input_dim=2, init="glorot_uniform", activation="relu"))
model.add(Dense(output_dim=16, init="glorot_uniform", activation="relu"))
model.add(Dense(output_dim=no_classes, init="glorot_uniform", activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer=Adagrad()) # wrong parameters of SDG may end-up in local minimum
model.fit(tr_samples, reformat_lables(tr_labels), batch_size=1000, verbose=1, show_accuracy=True, nb_epoch=100)

#print(model.layers[0].get_weights())

prd_val = model.predict_classes(val_samples)

print(metrics.classification_report(val_labels, prd_val))



