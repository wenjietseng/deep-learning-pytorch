import matplotlib.pyplot as plt
import numpy as np
import csv

# --- requirment 1 ---
# training loss
loss_show_attend_tell = open('./topdown.csv', 'r')
def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_loss(dta):
    loss = []
    for l in dta[:, 2]:
        loss.append(float(l))
    return loss

show_att_tell_dta = get_loss(read_table(loss_show_attend_tell))

# plt.subplots()
plt.plot(range(56644), show_att_tell_dta, label="Image")
# plt.ylim([0., ])
plt.legend()
plt.xlabel("Iteration (log scale)")
# plt.xscale("log")
plt.ylabel('MSE')
plt.savefig("topdown.png", dpi=300, bbox_inches='tight')
plt.close()
print(min(show_att_tell_dta))
