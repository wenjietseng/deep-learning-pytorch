from matplotlib import pyplot as plt
import numpy as np
import csv

f = open('./trainging_loss.csv', 'r')
def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta, dtype=float)

dta = read_table(f)
x = dta.shape[0]
print(x)
plt.figure(figsize=(3000/300, 1200/300), dpi=300)
plt.plot(range(x), dta[:,0], label="Training Loss", alpha=0.8, color="forestgreen")
# plt.xlabel('Every 10 batch steps')
plt.ylabel('CVAE Training Loss')
plt.savefig("CVAE-Training-Loss.png", dpi=300, bbox_inches='tight')
plt.close()
