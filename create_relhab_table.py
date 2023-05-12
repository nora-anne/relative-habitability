import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import UnivariateSpline
import sys

"""
Loads output arrays, calculate relative hab, saves into combined output.
"""

#parameter ranges for giants
m1s = np.linspace(.1,10,num=4)
m2s = np.linspace(.1,10,num=4)
a1s = np.concatenate((np.logspace(-1,.1,num=3),np.logspace(.1,1,num=6)[1:]))
alphas = np.logspace(-.7,-.12,num=8)
e1s = np.logspace(-2,-.04,num=6)
e1s[0]=0
e2s = np.copy(e1s)
incs = np.array([[0,0],[14.1*np.pi/180,1.4*np.pi/180]])
mutincs = np.array([0,(14.1-1.4)*np.pi/180])
delta_poms = np.array([0,np.pi])

#Earth range
Earth_smas = np.linspace(.6,1.9,num=80)
ref_hab_area = 0.8621696261749525

#HZ limits
IHZ_a = 0.99 #moist greenhouse IHZ from K13
OHZ_a = 1.7 #max greenhouse OHZ from K13

IHZ_a_2 = 0.97 #runaway greenhouse IHZ from K13
IHZ_a_3 = 0.75 #early Venus IHZ from K13

OHZ_a_2 = 1.77 #early Mars IHZ from K13

loaded = 0
time0 = time.time()
rel_hab = np.load('rel_hab_all.npy')

m1m1,m2m2,a1a1,alal,e1e1,e2e2,mimi,dpdp = np.meshgrid(m1s,m2s,a1s,alphas,e1s,e2s,mutincs,delta_poms)

tabledata = np.vstack((m1m1.flatten(),m2m2.flatten(),
                       a1a1.flatten(),alal.flatten(),
                       e1e1.flatten(),e2e2.flatten(),
                       mimi.flatten(),dpdp.flatten(),
                       rel_hab.flatten()))

print(tabledata.T.shape)

np.savetxt("foo.csv", tabledata.T, delimiter=",")

##hab_eccs_all = np.array([])
##
##rand_choice = np.random.randint(0,147455,20)
##for i_m1,m1 in enumerate(m1s):
##    for i_m2,m2 in enumerate(m2s):
##        for i_a1,a1 in enumerate(a1s):
##            for i_alpha,alpha in enumerate(alphas):
##                for i_e1,e1 in enumerate(e1s):
##                    for i_e2,e2 in enumerate(e2s):
##                        for i_inc in range(2):
##                            for i_delta_pom in range(2):
##                                if loaded in rand_choice:
##                                    print(m1,m2,a1,alpha,e1,e2,*incs[i_inc],delta_poms[i_delta_pom],
##                                          rel_hab[i_m1,i_m2,i_a1,i_alpha,i_e1,i_e2,i_inc,i_delta_pom])
##                                loaded +=1
