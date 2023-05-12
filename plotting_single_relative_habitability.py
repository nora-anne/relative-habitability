import numpy as np
import matplotlib.pyplot as plt
import rebound
from astropy import units as u
import sys
from scipy.interpolate import UnivariateSpline

from cycler import cycler
newcycle = cycler(color=['#f781bf','#ff7f00','#984ea3','#377eb8']) 
plt.rc('font', size=20,family='serif')
plt.rc('axes', prop_cycle=newcycle)

"""
Varies parameter for fiducial system and calculates and saves outcomes.
"""

Earth_smas = np.linspace(.6,1.9,num=80)
ref_hab_area = 0.8621696261749525

a1s = np.concatenate((np.logspace(-1,.1,num=30),np.logspace(.1,1,num=51)[1:]))
m2s = np.linspace(.1,10,num=40) #mass of inner giant planet in Mjup
e1s = np.logspace(-2,-.04,num=60) #ecc of inner giant
e1s[0] = 0

i=39
print(m2s[i])

data = np.load('param_m2_single'+str(i)+'.npz')
outcomes = data['outcomes']
outcomes_codes = data['outcome_codes']
max_eccs_E = data['max_eccs_E']
max_eccs_E[(outcomes==0)&(outcomes_codes!='h')] = 0

plt.figure(figsize=(10,20))
plt.subplot(311)
Esma_binwidth = np.mean(np.diff(Earth_smas))
for k,oc in enumerate(outcomes_codes):
    if oc=='2':
        plt.axvspan(Earth_smas[k]-.499*Esma_binwidth,Earth_smas[k]+.499*Esma_binwidth,alpha=.3,
                    color='k',ec=None)
    if oc=='O':
        plt.axvspan(Earth_smas[k]-.499*Esma_binwidth,Earth_smas[k]+.499*Esma_binwidth,alpha=.3,
                    color='r',ec=None)
    if oc=='z':
        plt.axvspan(Earth_smas[k]-.499*Esma_binwidth,Earth_smas[k]+.499*Esma_binwidth,alpha=.3,
                    color='b',ec=None)
    if oc=='u':
        plt.axvspan(Earth_smas[k]-.499*Esma_binwidth,Earth_smas[k]+.499*Esma_binwidth,alpha=.3,
                    color='g',ec=None)
    if oc=='h':
        plt.axvspan(Earth_smas[k]-.499*Esma_binwidth,Earth_smas[k]+.499*Esma_binwidth,alpha=.3,
                    color='orange',ec=None)

##plt.plot([],[],alpha=.3,color='k',label='giants unstable')
##plt.plot([],[],alpha=.3,color='r',label='LL predicted orbit crossing')
##plt.plot([],[],alpha=.3,color='b',label='unstable in 10$^4$ orbits')
plt.hist([],range=(1,1.1),alpha=.3,color='g',label='unstable in 5x10$^6$ orbits')
plt.hist([],range=(1,1.1),alpha=.3,color='orange',label='spectral fraction >0.05')
plt.legend(fontsize='x-small')
##plt.annotate('(a)',(.6,.1))
plt.plot(Earth_smas,outcomes,'.-')
plt.ylim(0,1.1)
##plt.xlabel('$a_\oplus$ (AU)')
plt.tick_params(axis='x',direction='inout',labelbottom=False)
plt.ylabel('stability outcome')
##plt.show()


#relative habitability calculation
hab_prob = np.zeros(len(Earth_smas))

IHZ_a = 0.99 #moist greenhouse IHZ from K13
OHZ_a = 1.7 #max greenhouse OHZ from K13

IHZ_a_2 = 0.97 #runaway greenhouse IHZ from K13
IHZ_a_3 = 0.75 #early Venus IHZ from K13

OHZ_a_2 = 1.77 #early Mars IHZ from K13

rfs = np.copy(Earth_smas)
rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4)
rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)

##plt.axvline(OHZ_a_2,linestyle=':',color='k',alpha=.5)
##plt.axvline(OHZ_a,color='k',alpha=.5)
##plt.axvline(IHZ_a,color='k',alpha=.5)
##plt.axvline(IHZ_a_3,linestyle=':',color='k',alpha=.5)
##plt.axvline(IHZ_a_2,linestyle='--',color='k',alpha=.5)
##plt.axhline(OHZ_a_2,linestyle=':',color='k',alpha=.5)
##plt.axhline(OHZ_a,color='k',alpha=.5)
##plt.axhline(IHZ_a,color='k',alpha=.5)
##plt.axhline(IHZ_a_3,linestyle=':',color='k',alpha=.5)
##plt.axhline(IHZ_a_2,linestyle='--',color='k',alpha=.5)
##plt.scatter(Earth_smas,rfs,c=max_eccs_E)
##plt.colorbar()
##plt.show()

fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]

hab_prob[fully_in_ind] = 1
hab_prob[inner_edge_ind] = 0.9
#linear decrease to 0 at the least conservative boundaries
hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

unperturbed_hab_prob = np.load('unperturbed_hab_prob.npy')

##plt.figure(figsize=(10,9))
plt.subplot(312)
plt.plot(Earth_smas,unperturbed_hab_prob,color='C3',alpha=.5,
         label='unperturbed')
plt.plot(Earth_smas,hab_prob,'.-',color='C1',label='with eccentricity')
##plt.xlabel('$a_\oplus$ (AU)')
plt.tick_params(axis='x',direction='inout',top=True,labelbottom=False)
plt.ylabel('habitability probability')
plt.legend(fontsize='x-small')
plt.ylim(0,1.1)
##plt.show()


combined_hab_prob = hab_prob*outcomes

#calculate area under the curve using spline/integral
spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

print(area1)    
print(area1/ref_hab_area)

##plt.figure(figsize=(10,9))
plt.subplot(313)
plt.plot(Earth_smas,outcomes,'--',label='stability outcome')
plt.plot(Earth_smas,hab_prob,':',label='habitability probability')
plt.plot(Earth_smas,combined_hab_prob,label='combined habitability probability')
plt.xlabel('$a_\oplus$ (AU)')
plt.tick_params(axis='x',direction='inout',top=True)
plt.legend(fontsize='x-small',loc='upper right')
plt.ylim(0,1.1)

plt.subplots_adjust(hspace=0)
##plt.savefig('example_plot')
##plt.savefig('example_plot.pdf',format='pdf')
plt.show()
