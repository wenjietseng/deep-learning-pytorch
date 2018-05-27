from matplotlib import pyplot as plt
import numpy as np
import csv

after_state = open('./after-state.csv', 'r')
def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta, dtype=float)

dta = read_table(after_state)
x = dta.shape[0]

plt.figure()#figsize=(3000/300, 1200/300), dpi=300)
plt.plot(range(x), dta[:, 0], label="TD after state", alpha=0.8, color="firebrick")
plt.legend()
plt.xlabel('Episodes 1 - 100000')
plt.ylabel('Reward')
plt.savefig("TDL.png", dpi=300, bbox_inches='tight')
plt.close()