from matplotlib import pyplot as plt
import numpy as np
import csv

f = open('./loss_and_probs.csv', 'r')
def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta, dtype=float)

dta = read_table(f)
x = dta.shape[0]

plt.figure(figsize=(3000/300, 1200/300), dpi=300)
plt.plot(range(x), dta[:,4], label="D_loss", alpha=0.8, color="firebrick")
plt.plot(range(x), dta[:,5], label='G_loss', alpha=0.8, color="forestgreen")
plt.plot(range(x), dta[:,6], label='Q_loss', alpha=0.8, color="gold")
plt.legend()
plt.xlabel('Every 100 batch steps')
plt.ylabel('InfoGAN Loss')
plt.savefig("InfoGAN-loss.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(3000/300, 1200/300), dpi=300)
plt.plot(range(x), dta[:,7], label="Real", alpha=0.8, color="skyblue")
plt.plot(range(x), dta[:,8], label='Fake before', alpha=0.8, color="steelblue")
plt.plot(range(x), dta[:,9], label='Fake after', alpha=0.8, color="violet")
plt.legend()
plt.xlabel('Every 100 batch steps')
plt.ylabel('InfoGAN Probability')
plt.savefig("InfoGAN-prob.png", dpi=300, bbox_inches='tight')
plt.close()
