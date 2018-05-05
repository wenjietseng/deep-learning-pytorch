import matplotlib.pyplot as plt
import numpy as np
import csv

# --- requirment 1 ---
# training loss
loss_show_attend_tell = open('./showAttendTell.csv', 'r')
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

plt.subplots()
plt.plot(range(56644), show_att_tell_dta, label="Show, Attend and Tell")
# plt.ylim([0., 10])
# plt.legend()
plt.xlabel("Iteration")
# plt.xscale("log")
plt.ylabel('Training Loss')
plt.savefig("showAttendTell.png", dpi=300, bbox_inches='tight')
plt.close()
print(min(show_att_tell_dta))
