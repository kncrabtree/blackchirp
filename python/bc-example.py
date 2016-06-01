# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:07:21 2016

@author: kncrabtree
"""

import blackchirpexperiment as bc
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print('Usage: bc-example.py exp_number path_to_exp')
    exit(1)
    
number = int(sys.argv[1])
path = sys.argv[2]

exp = bc.BlackChirpExperiment(number,path)
x, y, xl, yl, il, sl, pl, el, mx, my, cov = exp.analyze_fid(f_min=6500,f_max=19000,snr=5)

plt.plot(x,y)
plt.plot(xl,yl,'ro')
plt.plot(mx,my)
plt.show()
