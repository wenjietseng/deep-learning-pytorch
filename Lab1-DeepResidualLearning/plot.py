import matplotlib.pyplot as plt
import numpy as np
import csv

# training loss
resnet20_train = open('./output/resnet20-8-train.csv', 'r')
resnet56_train = open('./output/resnet56-8-train.csv', 'r')
resnet110_train = open('./output/resnet110-8-train.csv', 'r')
vcnn20_train = open('./output/vcnn20-8-train.csv', 'r')
vcnn56_train = open('./output/vcnn56-8-train.csv', 'r')
vcnn110_train = open('./output/vcnn110-8-train.csv', 'r')

# test error rate
resnet20_test = open('./output/resnet20-8-test.csv', 'r')
resnet56_test = open('./output/resnet56-8-test.csv', 'r')
resnet110_test = open('./output/resnet110-8-test.csv', 'r')
vcnn20_test = open('./output/vcnn20-8-test.csv', 'r')
vcnn56_test = open('./output/vcnn56-8-test.csv', 'r')
vcnn110_test = open('./output/vcnn110-8-test.csv', 'r')


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

vcnn20_train_loss = get_train_loss(read_table(vcnn20_train))
vcnn56_train_loss = get_train_loss(read_table(vcnn56_train))
vcnn110_train_loss = get_train_loss(read_table(vcnn110_train))
vcnn20_test_err = get_test_err(read_table(vcnn20_test))
vcnn56_test_err = get_test_err(read_table(vcnn56_test))
vcnn110_test_err = get_test_err(read_table(vcnn110_test))

# training loss resnet 20, 56, 110
plt.subplots()
plt.plot(range(164), res20_train_loss, label="ResNet20")
plt.plot(range(164), res56_train_loss, label='ResNet56')
plt.plot(range(164), res110_train_loss, label='ResNet110')
plt.legend()
plt.ylim([0., 2.5])
plt.xlabel("Epoch 1 - 164")
plt.ylabel('Training Loss')
plt.savefig("res_traing_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# testing error resnet 20, 56, 110
plt.subplots()
plt.plot(range(164), res20_test_err, label="ResNet20")
plt.plot(range(164), res56_test_err, label='ResNet56')
plt.plot(range(164), res110_test_err, label='ResNet110')
plt.legend()
plt.ylim([0, 1])
plt.xlabel("Epoch 1 - 164")
plt.ylabel('Testing Error')
plt.savefig("res_testing_error.png", dpi=300, bbox_inches='tight')
plt.close()

# training loss vanilla cnn 20, 56, 110
plt.subplots()
plt.plot(range(164), vcnn20_train_loss, label="VanillaCNN20")
plt.plot(range(164), vcnn56_train_loss, label='VanillaCNN56')
plt.plot(range(164), vcnn110_train_loss, label='VanillaCNN110')
plt.legend()
plt.ylim([0., 2.5])
plt.xlabel("Epoch 1 - 164")
plt.ylabel('Training Loss')
plt.savefig("vcnn_traing_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# testing error vanilla cnn 20, 56, 110
plt.subplots()
plt.plot(range(164), vcnn20_test_err, label="VanillaCNN20")
plt.plot(range(164), vcnn56_test_err, label='VanillaCNN56')
plt.plot(range(164), vcnn110_test_err, label='VanillaCNN110')
plt.legend()
plt.ylim([0, 1])
plt.xlabel("Epoch 1 - 164")
plt.ylabel('Testing Error')
plt.savefig("vcnn_testing_error.png", dpi=300, bbox_inches='tight')
plt.close()