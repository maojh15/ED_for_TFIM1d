filename = "fout_energy.txt"

klist = []
energy = []
kindex = -1
with open(filename, 'r') as fin:
    for line in fin:
        if line[0] == 'k':
            kindex += 1
            energy.append([])
            klist.append(int(line.split(' ')[-1]))
        else:
            energy[kindex].append(float(line))        

import numpy as np
klist = np.array(klist)
klist = 2.0 * np.pi * klist / len(klist)

linelen = (klist[1]-klist[0])/2
linelen /= 2
import matplotlib.pyplot as plt

ymin = np.inf
ymax = -np.inf
for i in range(len(klist)):
    x1 = klist[i] - linelen
    x2 = klist[i] + linelen
    for y in energy[i]:
        plt.plot([x1, x2], [y, y], '-', color = 'b')
        ymin = min(y, ymin)
        ymax = max(y, ymax)
    
for k in klist:
    plt.plot([k, k], [ymin, ymax], '--', color = 'red', linewidth=0.3)
plt.show()