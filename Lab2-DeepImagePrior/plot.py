import matplotlib.pyplot as plt
import numpy as np
import csv

# --- requirment 1 ---
# training loss
mse_1 = open('./output/6th-out1.csv', 'r')
mse_2 = open('./output/6th-out2.csv', 'r')
mse_3 = open('./output/6th-out3.csv', 'r')
mse_4 = open('./output/6th-out4.csv', 'r')

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_mse(dta):
    mse = []
    for l in dta[:, 1]:
        mse.append(float(l))
    return mse

mse_1_dta = get_mse(read_table(mse_1))
mse_2_dta = get_mse(read_table(mse_2))
mse_3_dta = get_mse(read_table(mse_3))
mse_4_dta = get_mse(read_table(mse_4))

# training loss resnet 20, 56, 110
# plt.subplots()
plt.plot(range(2400), mse_1_dta, label="Image")
plt.plot(range(2400), mse_2_dta, label="Image + noise")
plt.plot(range(2400), mse_3_dta, label="Image shuffled")
plt.plot(range(2400), mse_4_dta, label="U(0, 1) noise")
plt.ylim([0., .1])
plt.legend()
plt.xlabel("Iteration (log scale)")
plt.xscale("log")
plt.ylabel('MSE')
plt.savefig("requirement1-final.png", dpi=300, bbox_inches='tight')
plt.close()


# --- requirment 2 ---
# --- requirment 3 ---