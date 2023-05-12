import numpy as np
import matplotlib.pyplot as plt
import rebound
from astropy import units as u
import sys
from scipy.interpolate import UnivariateSpline

from cycler import cycler
newcycle = cycler(color=['#f781bf','#ff7f00','#984ea3','#377eb8']) 
plt.rc('font', size=18,family='serif')
plt.rc('axes', prop_cycle=newcycle)

"""
Varies parameter for fiducial system and calculates and saves outcomes.
"""

#Earth range
Earth_smas = np.linspace(.6,1.9,num=80)
ref_hab_area = 0.8621696261749525

#HZ limits
IHZ_a = 0.99 #moist greenhouse IHZ from K13
OHZ_a = 1.7 #max greenhouse OHZ from K13

IHZ_a_2 = 0.97 #runaway greenhouse IHZ from K13
IHZ_a_3 = 0.75 #early Venus IHZ from K13

OHZ_a_2 = 1.77 #early Mars IHZ from K13

#fiducial system
data = np.load('outputs_fiducial_inner.npz')
outcomes = data['outcomes']
outcomes_codes = data['outcome_codes']
max_eccs_E = data['max_eccs_E']

hab_prob = np.zeros(len(Earth_smas))

#accounts for eccentricity dependence
rfs = np.empty(len(Earth_smas))
rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
#estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                        

fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]

hab_prob[fully_in_ind] = 1
hab_prob[inner_edge_ind] = 0.9
#linear decrease to 0 at the least conservative boundaries
hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

combined_hab_prob = hab_prob*outcomes

#calculate area under the curve using spline/integral
spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

fid_rel_hab = np.array([area1/ref_hab_area])

#a1
a1s = np.concatenate((np.logspace(-1,.1,num=30),np.logspace(.1,1,num=51)[1:]))

rel_hab = np.full(len(a1s),np.nan)
for i,a1 in enumerate(a1s):
    data = np.load('outputs inner/param_a1_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            

    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

a1s = np.concatenate((a1s,[.15]))
rel_hab = np.concatenate((rel_hab,fid_rel_hab))
sortind = np.argsort(a1s)

plt.figure(figsize=(15,9))
plt.subplot(231)
plt.axvspan(.75,1.77,color='y',ec=None,alpha=.2) #a1 in HZ
plt.axvspan(.99,1.7,color='y',ec=None,alpha=.2) #a1 in HZ
plt.axvspan(.24054182,0.56542654,color='royalblue',ec=None,alpha=.2) #a2 in HZ
plt.axvspan(.31713971,0.54371045,color='royalblue',ec=None,alpha=.2) #a2 in HZ
plt.axvline(.15,linestyle='--',color='k')
plt.plot(a1s[sortind],rel_hab[sortind])
plt.ylim(0,1.1)
plt.xscale('log')
plt.xlabel('$a_1$ (AU)')

#alpha
alphas = np.logspace(-.7,-.12,num=80) #ratio of sma of giants

rel_hab = np.full(len(alphas),np.nan)
for i,alpha in enumerate(alphas):
    data = np.load('outputs inner/param_alpha_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

alphas = np.concatenate((alphas,[.32]))
rel_hab = np.concatenate((rel_hab,fid_rel_hab))
sortind = np.argsort(alphas)

sp2 = plt.subplot(232)
plt.axvline(.32,linestyle='--',color='k')
alphas_MMR = np.load('alphas_MMR.npy')
PRlabels_MMR = np.load('PRlabels_MMR.npy')[np.argsort(alphas_MMR)]
alt = -1
for aMMR,lab in zip(np.sort(alphas_MMR),PRlabels_MMR):
    if aMMR<.7:
        plt.axvline(aMMR,linestyle=':',color='g',alpha=.5)
        alt +=1
        if alt%2==0:
            plt.annotate(lab,(.95*aMMR,1.04),fontsize='xx-small')
        if alt%2!=0:
            plt.annotate(lab,(.95*aMMR,.96),fontsize='xx-small')
plt.plot(alphas[sortind],rel_hab[sortind])
plt.ylim(0,1.1)
plt.xscale('log')
plt.xlabel(r'$\alpha$')
import matplotlib.ticker as ticker
locs = [.2,.27,.36,.5,.76]
sp2.xaxis.set_minor_locator(ticker.FixedLocator(locs))
sp2.xaxis.set_minor_formatter(ticker.ScalarFormatter())

#e1
e1s = np.logspace(-2,-.04,num=60) #ecc of inner giant
e1s[0]=0

rel_hab = np.full(len(e1s),np.nan)
for i,e1 in enumerate(e1s):
    data = np.load('outputs inner/param_e1_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

e1s = np.concatenate((e1s,[.05]))
rel_hab = np.concatenate((rel_hab,fid_rel_hab))
sortind = np.argsort(e1s)

plt.subplot(233)
plt.axvline(.05,linestyle='--',color='k')
plt.plot(e1s[sortind],rel_hab[sortind],linestyle='--',lw=2,label='1')
plt.ylim(0,1.1)
plt.xscale('log')
plt.xlabel('e')

#e2
e2s = np.logspace(-2,-.04,num=60) #ecc of inner giant
e2s[0]=0

rel_hab = np.full(len(e2s),np.nan)
for i,e2 in enumerate(e2s):
    data = np.load('outputs inner/param_e2_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

e2s = np.concatenate((e2s,[.05]))
rel_hab = np.concatenate((rel_hab,fid_rel_hab))
sortind = np.argsort(e2s)

plt.subplot(233)
plt.plot(e2s[sortind],rel_hab[sortind],label='2')
plt.legend(fontsize='x-small')

#m1
m1s = np.linspace(.1,10,num=40) #mass of inner giant planet in Mjup

rel_hab = np.full(len(m1s),np.nan)
for i,m1 in enumerate(m1s):
    data = np.load('outputs inner/param_m1_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

m1s = np.concatenate((m1s,[1]))
rel_hab = np.concatenate((rel_hab,fid_rel_hab))
sortind = np.argsort(m1s)

plt.subplot(234)
plt.axvline(1,linestyle='--',color='k')
plt.plot(m1s[sortind],rel_hab[sortind],linestyle='--',lw=2,label='1')
plt.ylim(0,1.1)
plt.xlabel('$m$ ($M_J$)')

#m2
m2s = np.linspace(.1,10,num=40) #mass of inner giant planet in Mjup

rel_hab = np.full(len(m2s),np.nan)
for i,m2 in enumerate(m2s):
    data = np.load('outputs inner/param_m2_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

m2s = np.concatenate((m2s,[1]))
rel_hab = np.concatenate((rel_hab,fid_rel_hab))
sortind = np.argsort(m2s)

plt.subplot(234)
plt.plot(m2s[sortind],rel_hab[sortind],label='2')
plt.legend(fontsize='x-small')

#i1
i1s = np.linspace(0,np.pi,num=50)

rel_hab = np.full(len(i1s),np.nan)
rel_hab[0] = fid_rel_hab
for i,i1 in enumerate(i1s[1:]):
    data = np.load('outputs inner/param_i1_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i+1] = area1/ref_hab_area
plt.subplot(235)
plt.axvline(0,linestyle='--',color='k')
plt.plot(i1s,rel_hab,linestyle='--',lw=2,label='1')
plt.ylim(0,1.1)
plt.xlabel('$i$ (rad)')

#i2
i2s = np.linspace(0,np.pi,num=50)

rel_hab = np.full(len(i2s),np.nan)
rel_hab[0] = fid_rel_hab
for i,i2 in enumerate(i2s[1:]):
    data = np.load('outputs inner/param_i2_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i+1] = area1/ref_hab_area
plt.subplot(235)
plt.plot(i2s,rel_hab,label='2')

#iE
iEs = np.linspace(0,np.pi,num=50)

rel_hab = np.full(len(iEs),np.nan)
rel_hab[0] = fid_rel_hab
for i,iE in enumerate(iEs[1:]):
    data = np.load('outputs inner/param_iE_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i+1] = area1/ref_hab_area
plt.subplot(235)
plt.plot(iEs,rel_hab,linestyle='-.',lw=2,label='$\oplus$')
plt.legend(fontsize='x-small')

#deltapom
deltapoms = np.linspace(0,np.pi,20) #alignment of giant planet periastrons

rel_hab = np.full(len(deltapoms),np.nan)
for i,deltapom in enumerate(deltapoms):
    data = np.load('outputs inner/param_deltapom_single'+str(i)+'.npz')
    outcomes = data['outcomes']
    outcomes_codes = data['outcome_codes']
    max_eccs_E = data['max_eccs_E']

    hab_prob = np.zeros(len(Earth_smas))

    #accounts for eccentricity dependence
    rfs = np.empty(len(Earth_smas))
    rfs[max_eccs_E<.6] = Earth_smas[max_eccs_E<.6]*(1-max_eccs_E[max_eccs_E<.6]**2)**(1/4) #effective flux distance from mean flux approximation
    #estimate the flux dependence from Palubski et al 2020 -- it's closer to 1/2.1
    rfs[max_eccs_E>=.6] = Earth_smas[max_eccs_E>=.6]*(1-max_eccs_E[max_eccs_E>=.6]**2)**(1/2.1)
                                            
    fully_in_ind = np.where((rfs>=IHZ_a)&(rfs<=OHZ_a))[0] #most conservative boundaries
    inner_edge_ind = np.where((rfs<IHZ_a)&(rfs>=IHZ_a_2))[0]
    inner_transition_ind = np.where((rfs<IHZ_a_2)&(rfs>=IHZ_a_3))[0]
    outer_transition_ind = np.where((rfs>OHZ_a)&(rfs<=OHZ_a_2))[0]
    
    hab_prob[fully_in_ind] = 1
    hab_prob[inner_edge_ind] = 0.9
    #linear decrease to 0 at the least conservative boundaries
    hab_prob[inner_transition_ind] = .9*(rfs[inner_transition_ind]-IHZ_a_3)/(IHZ_a_2-IHZ_a_3)
    hab_prob[outer_transition_ind] = 1-(rfs[outer_transition_ind]-OHZ_a)/(OHZ_a_2-OHZ_a)

    combined_hab_prob = hab_prob*outcomes

    #calculate area under the curve using spline/integral
    spline1 = UnivariateSpline(Earth_smas,combined_hab_prob,s=0)
    area1 = spline1.integral(min(Earth_smas),max(Earth_smas))

    rel_hab[i] = area1/ref_hab_area

plt.subplot(236)
plt.axvline(np.pi,linestyle='--',color='k')
plt.plot(deltapoms,rel_hab)
plt.ylim(0,1.1)
plt.xlabel(r'$\Delta \varpi$ (rad)')

plt.gcf().supylabel('relative habitability')
plt.subplots_adjust(hspace=.23)
plt.show()
