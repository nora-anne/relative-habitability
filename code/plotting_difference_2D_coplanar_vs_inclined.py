import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rc('font',family='serif',size=18)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=12)

"""
Plots the difference in the mean of relative habitability in a 2d corner style plot.
Between coplanar case and inclined case.
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
delta_poms = np.array([0,np.pi])

rel_hab = np.load('rel_hab_all.npy')

cmap_name = 'PuOr'

#coplanar
inc_type1 = 'coplanar'
m2m1_relhab1 = np.full((len(m2s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_m2,m2 in enumerate(m2s):
        m2m1_relhab1[i_m2,i_m1] = np.mean(rel_hab[i_m1,i_m2,:,:,:,:,0,:])
a1m1_relhab1 = np.full((len(a1s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_a1,a1 in enumerate(a1s):
        a1m1_relhab1[i_a1,i_m1] = np.mean(rel_hab[i_m1,:,i_a1,:,:,:,0,:])
alpham1_relhab1 = np.full((len(alphas),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_alpha,alpha in enumerate(alphas):
        alpham1_relhab1[i_alpha,i_m1] = np.mean(rel_hab[i_m1,:,:,i_alpha,:,:,0,:])
e1m1_relhab1 = np.full((len(e1s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_e1,e1 in enumerate(e1s):
        e1m1_relhab1[i_e1,i_m1] = np.mean(rel_hab[i_m1,:,:,:,i_e1,:,0,:])
e2m1_relhab1 = np.full((len(e2s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_e2,e2 in enumerate(e2s):
        e2m1_relhab1[i_e2,i_m1] = np.mean(rel_hab[i_m1,:,:,:,:,i_e2,0,:])
a1m2_relhab1 = np.full((len(a1s),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_a1,a1 in enumerate(a1s):
        a1m2_relhab1[i_a1,i_m2] = np.mean(rel_hab[:,i_m2,i_a1,:,:,:,0,:])
alpham2_relhab1 = np.full((len(alphas),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_alpha,alpha in enumerate(alphas):
        alpham2_relhab1[i_alpha,i_m2] = np.mean(rel_hab[:,i_m2,:,i_alpha,:,:,0,:])
e1m2_relhab1 = np.full((len(e1s),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_e1,e1 in enumerate(e1s):
        e1m2_relhab1[i_e1,i_m2] = np.mean(rel_hab[:,i_m2,:,:,i_e1,:,0,:])
e2m2_relhab1 = np.full((len(e2s),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_e2,e2 in enumerate(e2s):
        e2m2_relhab1[i_e2,i_m2] = np.mean(rel_hab[:,i_m2,:,:,:,i_e2,0,:])
alphaa1_relhab1 = np.full((len(alphas),len(a1s)),np.nan)
for i_a1,a1 in enumerate(a1s):
    for i_alpha,alpha in enumerate(alphas):
        alphaa1_relhab1[i_alpha,i_a1] = np.mean(rel_hab[:,:,i_a1,i_alpha,:,:,0,:])
e1a1_relhab1 = np.full((len(e1s),len(a1s)),np.nan)
for i_a1,a1 in enumerate(a1s):
    for i_e1,e1 in enumerate(e1s):
        e1a1_relhab1[i_e1,i_a1] = np.mean(rel_hab[:,:,i_a1,:,i_e1,:,0,:])
e2a1_relhab1 = np.full((len(e2s),len(a1s)),np.nan)
for i_a1,a1 in enumerate(a1s):
    for i_e2,e2 in enumerate(e2s):
        e2a1_relhab1[i_e2,i_a1] = np.mean(rel_hab[:,:,i_a1,:,:,i_e2,0,:])
e1alpha_relhab1 = np.full((len(e1s),len(alphas)),np.nan)
for i_alpha,alpha in enumerate(alphas):
    for i_e1,e1 in enumerate(e1s):
        e1alpha_relhab1[i_e1,i_alpha] = np.mean(rel_hab[:,:,:,i_alpha,i_e1,:,0,:])
e2alpha_relhab1 = np.full((len(e2s),len(alphas)),np.nan)
for i_alpha,alpha in enumerate(alphas):
    for i_e2,e2 in enumerate(e2s):
        e2alpha_relhab1[i_e2,i_alpha] = np.mean(rel_hab[:,:,:,i_alpha,:,i_e2,0,:])
e2e1_relhab1 = np.full((len(e2s),len(e1s)),np.nan)
for i_e1,e1 in enumerate(e1s):
    for i_e2,e2 in enumerate(e2s):
        e2e1_relhab1[i_e2,i_e1] = np.mean(rel_hab[:,:,:,:,i_e1,i_e2,0,:])

#inclined
inc_type2 = 'inclined'
m2m1_relhab2 = np.full((len(m2s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_m2,m2 in enumerate(m2s):
        m2m1_relhab2[i_m2,i_m1] = np.mean(rel_hab[i_m1,i_m2,:,:,:,:,1,:])
a1m1_relhab2 = np.full((len(a1s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_a1,a1 in enumerate(a1s):
        a1m1_relhab2[i_a1,i_m1] = np.mean(rel_hab[i_m1,:,i_a1,:,:,:,1,:])
alpham1_relhab2 = np.full((len(alphas),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_alpha,alpha in enumerate(alphas):
        alpham1_relhab2[i_alpha,i_m1] = np.mean(rel_hab[i_m1,:,:,i_alpha,:,:,1,:])
e1m1_relhab2 = np.full((len(e1s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_e1,e1 in enumerate(e1s):
        e1m1_relhab2[i_e1,i_m1] = np.mean(rel_hab[i_m1,:,:,:,i_e1,:,1,:])
e2m1_relhab2 = np.full((len(e2s),len(m1s)),np.nan)
for i_m1,m1 in enumerate(m1s):
    for i_e2,e2 in enumerate(e2s):
        e2m1_relhab2[i_e2,i_m1] = np.mean(rel_hab[i_m1,:,:,:,:,i_e2,1,:])
a1m2_relhab2 = np.full((len(a1s),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_a1,a1 in enumerate(a1s):
        a1m2_relhab2[i_a1,i_m2] = np.mean(rel_hab[:,i_m2,i_a1,:,:,:,1,:])
alpham2_relhab2 = np.full((len(alphas),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_alpha,alpha in enumerate(alphas):
        alpham2_relhab2[i_alpha,i_m2] = np.mean(rel_hab[:,i_m2,:,i_alpha,:,:,1,:])
e1m2_relhab2 = np.full((len(e1s),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_e1,e1 in enumerate(e1s):
        e1m2_relhab2[i_e1,i_m2] = np.mean(rel_hab[:,i_m2,:,:,i_e1,:,1,:])
e2m2_relhab2 = np.full((len(e2s),len(m2s)),np.nan)
for i_m2,m2 in enumerate(m2s):
    for i_e2,e2 in enumerate(e2s):
        e2m2_relhab2[i_e2,i_m2] = np.mean(rel_hab[:,i_m2,:,:,:,i_e2,1,:])
alphaa1_relhab2 = np.full((len(alphas),len(a1s)),np.nan)
for i_a1,a1 in enumerate(a1s):
    for i_alpha,alpha in enumerate(alphas):
        alphaa1_relhab2[i_alpha,i_a1] = np.mean(rel_hab[:,:,i_a1,i_alpha,:,:,1,:])
e1a1_relhab2 = np.full((len(e1s),len(a1s)),np.nan)
for i_a1,a1 in enumerate(a1s):
    for i_e1,e1 in enumerate(e1s):
        e1a1_relhab2[i_e1,i_a1] = np.mean(rel_hab[:,:,i_a1,:,i_e1,:,1,:])
e2a1_relhab2 = np.full((len(e2s),len(a1s)),np.nan)
for i_a1,a1 in enumerate(a1s):
    for i_e2,e2 in enumerate(e2s):
        e2a1_relhab2[i_e2,i_a1] = np.mean(rel_hab[:,:,i_a1,:,:,i_e2,1,:])
e1alpha_relhab2 = np.full((len(e1s),len(alphas)),np.nan)
for i_alpha,alpha in enumerate(alphas):
    for i_e1,e1 in enumerate(e1s):
        e1alpha_relhab2[i_e1,i_alpha] = np.mean(rel_hab[:,:,:,i_alpha,i_e1,:,1,:])
e2alpha_relhab2 = np.full((len(e2s),len(alphas)),np.nan)
for i_alpha,alpha in enumerate(alphas):
    for i_e2,e2 in enumerate(e2s):
        e2alpha_relhab2[i_e2,i_alpha] = np.mean(rel_hab[:,:,:,i_alpha,:,i_e2,1,:])
e2e1_relhab2 = np.full((len(e2s),len(e1s)),np.nan)
for i_e1,e1 in enumerate(e1s):
    for i_e2,e2 in enumerate(e2s):
        e2e1_relhab2[i_e2,i_e1] = np.mean(rel_hab[:,:,:,:,i_e1,i_e2,1,:])

plt.figure(figsize=(10,9))
m2m1 = plt.subplot(5,5,1)
plt.imshow(m2m1_relhab1-m2m1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
m2m1.axes.get_xaxis().set_visible(False)
plt.yticks(np.arange(len(m2s),dtype=int),np.around(m2s,1))
plt.ylabel('$m_2$')

a1m1 = plt.subplot(5,5,6,sharex=m2m1)
plt.imshow(a1m1_relhab1-a1m1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
a1m1.axes.get_xaxis().set_visible(False)
plt.yticks(np.arange(len(a1s),dtype=int),np.around(a1s,1))
plt.ylabel('$a_1$')

alpham1 = plt.subplot(5,5,11,sharex=m2m1)
plt.imshow(alpham1_relhab1-alpham1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
alpham1.axes.get_xaxis().set_visible(False)
plt.yticks(np.arange(len(alphas),dtype=int),np.around(alphas,2))
plt.ylabel(r'$\alpha$')

e1m1 = plt.subplot(5,5,16,sharex=m2m1)
plt.imshow(e1m1_relhab1-e1m1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e1m1.axes.get_xaxis().set_visible(False)
plt.yticks(np.arange(len(e1s),dtype=int),np.around(e1s,2))
plt.ylabel('$e_1$')

e2m1 = plt.subplot(5,5,21,sharex=m2m1)
plt.imshow(e2m1_relhab1-e2m1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
plt.yticks(np.arange(len(e2s),dtype=int),np.around(e2s,2))
plt.xticks(np.arange(len(m1s),dtype=int),np.around(m1s,1))
plt.ylabel('$e_2$')
plt.xlabel('$m_1$')

a1m2 = plt.subplot(5,5,7,sharey=a1m1)
plt.imshow(a1m2_relhab1-a1m2_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
a1m2.axes.get_xaxis().set_visible(False)
a1m2.axes.get_yaxis().set_visible(False)

alpham2 = plt.subplot(5,5,12,sharex=a1m2,sharey=a1m1)
plt.imshow(alpham2_relhab1-alpham2_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
alpham2.axes.get_xaxis().set_visible(False)
alpham2.axes.get_yaxis().set_visible(False)

e1m2 = plt.subplot(5,5,17,sharex=a1m2,sharey=e1m1)
plt.imshow(e1m2_relhab1-e1m2_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e1m2.axes.get_xaxis().set_visible(False)
e1m2.axes.get_yaxis().set_visible(False)

e2m2 = plt.subplot(5,5,22,sharex=a1m2,sharey=e2m1)
plt.imshow(e2m2_relhab1-e2m2_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e2m2.axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(len(m2s),dtype=int),np.around(m2s,1))
plt.xlabel('$m_2$')

alphaa1 = plt.subplot(5,5,13,sharey=alpham1)
plt.imshow(alphaa1_relhab1-alphaa1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
alphaa1.axes.get_xaxis().set_visible(False)
alphaa1.axes.get_yaxis().set_visible(False)

e1a1 = plt.subplot(5,5,18,sharex=alphaa1,sharey=e1m1)
plt.imshow(e1a1_relhab1-e1a1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e1a1.axes.get_xaxis().set_visible(False)
e1a1.axes.get_yaxis().set_visible(False)

e2a1 = plt.subplot(5,5,23,sharex=alphaa1,sharey=e2m1)
plt.imshow(e2a1_relhab1-e2a1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e2a1.axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(len(a1s),dtype=int)[::2],np.around(a1s,1)[::2])
plt.xlabel('$a_1$')

e1alpha = plt.subplot(5,5,19,sharey=e1m1)
plt.imshow(e1alpha_relhab1-e1alpha_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e1alpha.axes.get_xaxis().set_visible(False)
e1alpha.axes.get_yaxis().set_visible(False)

e2alpha = plt.subplot(5,5,24,sharex=e1alpha,sharey=e2m1)
plt.imshow(e2alpha_relhab1-e2alpha_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e2alpha.axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(len(alphas),dtype=int)[::2],np.around(alphas,2)[::2])
plt.xlabel(r'$\alpha$')

e2e1 = plt.subplot(5,5,25,sharey=e2m1)
plt.imshow(e2e1_relhab1-e2e1_relhab2,origin='lower',vmin=-.12,vmax=.12,aspect='auto',cmap=cmap_name)
e2e1.axes.get_yaxis().set_visible(False)
plt.xticks(np.arange(len(e1s),dtype=int)[::2],np.around(e1s,2)[::2])
plt.xlabel('$e_1$')

plt.subplots_adjust(wspace=.05, hspace=.05)

cbax = plt.gcf().add_axes([0.91, 0.15, 0.02, 0.7])
plt.colorbar(cax=cbax)

plt.suptitle('mean relative habitability \n'+inc_type1+' - '+inc_type2)

plt.show()

