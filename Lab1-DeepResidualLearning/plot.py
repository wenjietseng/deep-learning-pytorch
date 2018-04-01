import matplotlib.pyplot as plt
import numpy as np
import csv

# plot training loss
resnet20_train = open('./output/resnet20-7-train.csv', 'r')
resnet56_train = open('./output/resnet56-7-train.csv', 'r')
resnet110_train = open('./output/resnet110-7-train.csv', 'r')

# plot test error rate
resnet20_test = open('./output/resnet20-7-test.csv', 'r')
resnet56_test = open('./output/resnet56-7-test.csv', 'r')
resnet110_test = open('./output/resnet110-7-test.csv', 'r')


def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta):
    loss = []
    for l in dta[:, 1]:
        loss.append(float(l))
    return loss

def get_test_err(dta):
    err = []
    for e in dta[:, 2]:
        err.append((100.0 - float(e)) / 100.0)
    return err

res20_train_loss = get_train_loss(read_table(resnet20_train))
res56_train_loss = get_train_loss(read_table(resnet56_train))
res110_train_loss = get_train_loss(read_table(resnet110_train))
res20_test_err = get_test_err(read_table(resnet20_test))
res56_test_err = get_test_err(read_table(resnet56_test))
res110_test_err = get_test_err(read_table(resnet110_test))

print(res110_test_err)
print(res110_train_loss)
print(res20_test_err)
print(res20_train_loss)
print(res56_test_err)
print(res56_train_loss)


# plt.subplot(1, 2, 1)
plt.plot(range(164), train_err, label="Train")
# plt.ylim([0, 1.])
# plt.xlabel("epoch 1 - 164")
# plt.ylabel('Error Rate')

# plt.subplot(1, 2, 2)
# plt.plot(range(164), test_err, label="Test")
# plt.ylim([0, 1.])
# plt.xlabel("epoch 1 - 164")
# # plt.ylabel('Error Rate')

# plt.show()