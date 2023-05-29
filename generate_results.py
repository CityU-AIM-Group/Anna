import os
from numpy import *

root = './rerun/officehome/best_results/'

files = os.listdir(root)
files.sort()
print(files)
results = []
OS_star = []
UNK = []
HOS = []
for file in files:
    nmes = file.split('_')
    if nmes[0] == 'latex':
        with open(root+file, 'r') as f:
            _best = f.readlines()[-1].strip()
            print('{} to {} '.format(nmes[1], nmes[2].split('.')[0]) + ' '+_best)
            re = _best.split(' & ')
            OS_star.append(float(re[0]))
            UNK.append(float(re[1]))
            HOS.append(float(re[2]))
            results.append(_best)

avg = [str(round(mean(OS_star),2))+' & '+ str(round(mean(UNK), 2)) +' & '+ str(round(mean(HOS),2))]
results += avg
results = ' & '.join(results)
print(results)

