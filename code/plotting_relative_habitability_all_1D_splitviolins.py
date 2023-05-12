import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

from cycler import cycler
newcycle = cycler(color=['#f781bf','#ff7f00','#984ea3','#377eb8']) 
plt.rc('font', size=15,family='serif')
plt.rc('axes', prop_cycle=newcycle)

"""
Plots violin plots for all 4 different scenarios in 1 dimension for each parameter.
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
print(len(rel_hab.flatten()[rel_hab.flatten()==0]))
sys.exit()

#calculate all the relevant kde heights for normalizing the width
#coplanar, anti-aligned
rel_hab_m1_00 = [rel_hab[i_m1,:,:,:,:,:,0,0].flatten() for i_m1 in range(len(m1s))]
rel_hab_m2_00 = [rel_hab[:,i_m2,:,:,:,:,0,0].flatten() for i_m2 in range(len(m2s))]
rel_hab_a1_00 = [rel_hab[:,:,i_a1,:,:,:,0,0].flatten() for i_a1 in range(len(a1s))]
rel_hab_alpha_00 = [rel_hab[:,:,:,i_alpha,:,:,0,0].flatten() for i_alpha in range(len(alphas))]
rel_hab_e1_00 = [rel_hab[:,:,:,:,i_e1,:,0,0].flatten() for i_e1 in range(len(e1s))]
rel_hab_e2_00 = [rel_hab[:,:,:,:,:,i_e2,0,0].flatten() for i_e2 in range(len(e2s))]
#inclined, anti-aligned
rel_hab_m1_10 = [rel_hab[i_m1,:,:,:,:,:,1,0].flatten() for i_m1 in range(len(m1s))]
rel_hab_m2_10 = [rel_hab[:,i_m2,:,:,:,:,1,0].flatten() for i_m2 in range(len(m2s))]
rel_hab_a1_10 = [rel_hab[:,:,i_a1,:,:,:,1,0].flatten() for i_a1 in range(len(a1s))]
rel_hab_alpha_10 = [rel_hab[:,:,:,i_alpha,:,:,1,0].flatten() for i_alpha in range(len(alphas))]
rel_hab_e1_10 = [rel_hab[:,:,:,:,i_e1,:,1,0].flatten() for i_e1 in range(len(e1s))]
rel_hab_e2_10 = [rel_hab[:,:,:,:,:,i_e2,1,0].flatten() for i_e2 in range(len(e2s))]
#coplanar, anti-aligned
rel_hab_m1_01 = [rel_hab[i_m1,:,:,:,:,:,0,1].flatten() for i_m1 in range(len(m1s))]
rel_hab_m2_01 = [rel_hab[:,i_m2,:,:,:,:,0,1].flatten() for i_m2 in range(len(m2s))]
rel_hab_a1_01 = [rel_hab[:,:,i_a1,:,:,:,0,1].flatten() for i_a1 in range(len(a1s))]
rel_hab_alpha_01 = [rel_hab[:,:,:,i_alpha,:,:,0,1].flatten() for i_alpha in range(len(alphas))]
rel_hab_e1_01 = [rel_hab[:,:,:,:,i_e1,:,0,1].flatten() for i_e1 in range(len(e1s))]
rel_hab_e2_01 = [rel_hab[:,:,:,:,:,i_e2,0,1].flatten() for i_e2 in range(len(e2s))]
#inclined, anti-aligned
rel_hab_m1_11 = [rel_hab[i_m1,:,:,:,:,:,1,1].flatten() for i_m1 in range(len(m1s))]
rel_hab_m2_11 = [rel_hab[:,i_m2,:,:,:,:,1,1].flatten() for i_m2 in range(len(m2s))]
rel_hab_a1_11 = [rel_hab[:,:,i_a1,:,:,:,1,1].flatten() for i_a1 in range(len(a1s))]
rel_hab_alpha_11 = [rel_hab[:,:,:,i_alpha,:,:,1,1].flatten() for i_alpha in range(len(alphas))]
rel_hab_e1_11 = [rel_hab[:,:,:,:,i_e1,:,1,1].flatten() for i_e1 in range(len(e1s))]
rel_hab_e2_11 = [rel_hab[:,:,:,:,:,i_e2,1,1].flatten() for i_e2 in range(len(e2s))]

##max_kdevals = np.zeros((4,6,8))
##for i in range(len(m1s)):
##    rel_habi = rel_hab_m1_00[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[0,0,i] = max(kde_vals)
##
##    rel_habi = rel_hab_m1_10[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[1,0,i] = max(kde_vals)
##
##    rel_habi = rel_hab_m1_01[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[2,0,i] = max(kde_vals)
##
##    rel_habi = rel_hab_m1_11[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[3,0,i] = max(kde_vals)
##for i in range(len(m2s)):
##    rel_habi = rel_hab_m2_00[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[0,1,i] = max(kde_vals)
##
##    rel_habi = rel_hab_m2_10[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[1,1,i] = max(kde_vals)
##
##    rel_habi = rel_hab_m2_01[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[2,1,i] = max(kde_vals)
##
##    rel_habi = rel_hab_m2_11[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[3,1,i] = max(kde_vals)
##for i in range(len(a1s)):
##    rel_habi = rel_hab_a1_00[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[0,2,i] = max(kde_vals)
##
##    rel_habi = rel_hab_a1_10[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[1,2,i] = max(kde_vals)
##
##    rel_habi = rel_hab_a1_01[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[2,2,i] = max(kde_vals)
##
##    rel_habi = rel_hab_a1_11[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[3,2,i] = max(kde_vals)
##for i in range(len(alphas)):
##    rel_habi = rel_hab_alpha_00[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[0,3,i] = max(kde_vals)
##
##    rel_habi = rel_hab_alpha_10[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[1,3,i] = max(kde_vals)
##
##    rel_habi = rel_hab_alpha_01[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[2,3,i] = max(kde_vals)
##
##    rel_habi = rel_hab_alpha_11[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[3,3,i] = max(kde_vals)
##for i in range(len(e1s)):
##    rel_habi = rel_hab_e1_00[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[0,4,i] = max(kde_vals)
##
##    rel_habi = rel_hab_e1_10[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[1,4,i] = max(kde_vals)
##
##    rel_habi = rel_hab_e1_01[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[2,4,i] = max(kde_vals)
##
##    rel_habi = rel_hab_e1_11[i]
##    x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##    kde = stats.gaussian_kde(rel_habi)
##    kde_vals = kde(x)
##    max_kdevals[3,4,i] = max(kde_vals)
##for i in range(len(e2s)):
##    rel_habi = rel_hab_e2_00[i]
##    try:
##        x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##        kde = stats.gaussian_kde(rel_habi)
##        kde_vals = kde(x)
##        max_kdevals[0,5,i] = max(kde_vals)
##    except:
##        max_kdevals[0,5,i] = max_kdevals[0,4,i]
##
##    rel_habi = rel_hab_e2_10[i]
##    try:
##        x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##        kde = stats.gaussian_kde(rel_habi)
##        kde_vals = kde(x)
##        max_kdevals[1,5,i] = max(kde_vals)
##    except:
##        max_kdevals[1,5,i] = max_kdevals[1,4,i]
##
##    rel_habi = rel_hab_e2_01[i]
##    try:
##        x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##        kde = stats.gaussian_kde(rel_habi)
##        kde_vals = kde(x)
##        max_kdevals[2,5,i] = max(kde_vals)
##    except:
##        max_kdevals[2,5,i] = max_kdevals[2,4,i]
##
##    rel_habi = rel_hab_e2_11[i]
##    try:
##        x = np.linspace(min(rel_habi),max(rel_habi),num=100)
##        kde = stats.gaussian_kde(rel_habi)
##        kde_vals = kde(x)
##        max_kdevals[3,5,i] = max(kde_vals)
##    except:
##        max_kdevals[3,5,i] = max_kdevals[3,4,i]
##
##vp_width = max_kdevals/np.amax(max_kdevals) * 9.06
##vp_width[:,0,:] *= 4/8
##vp_width[:,1,:] *= 4/8
##vp_width[:,4,:] *= 6/8
##vp_width[:,5,:] *= 6/8
##np.save('violin_plot_width',vp_width)
##print(np.median(vp_width))
##sys.exit()
vp_width = np.load('violin_plot_width.npy')
print(len(vp_width[vp_width>2]))
vp_width[vp_width>2] = 2

#violin plots
plt.figure(figsize=(8.5,11))
plt.subplot(321)
v1 = plt.violinplot(rel_hab_m1_00,widths=vp_width[0,0,:len(m1s)],showextrema=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('r')
v2 = plt.violinplot(rel_hab_m1_10,widths=vp_width[1,0,:len(m1s)],showextrema=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('y')
v3 = plt.violinplot(rel_hab_m1_01,widths=vp_width[2,0,:len(m1s)],showextrema=False)
for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('b')
v4 = plt.violinplot(rel_hab_m1_11,widths=vp_width[3,0,:len(m1s)],showextrema=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('c')

plt.xticks(ticks=np.arange(1,len(m1s)+1),labels=[str(m1)[:4] for m1 in m1s])
plt.ylim(0,1.15)
plt.hist([],range=(.5,1),color='r',alpha=.6,label='coplanar aligned')
plt.hist([],range=(.5,1),color='b',alpha=.6,label='coplanar antialigned')
plt.hist([],range=(.5,1),color='y',alpha=.6,label='inclined aligned')
plt.hist([],range=(.5,1),color='c',alpha=.6,label='inclined antialigned')
plt.legend(loc='upper center',ncol=2,bbox_to_anchor=(1, 1.4))
plt.xlabel('$m_1$')

plt.subplot(322)
v1 = plt.violinplot(rel_hab_m2_00,widths=vp_width[0,1,:len(m2s)],showextrema=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('r')
v2 = plt.violinplot(rel_hab_m2_10,widths=vp_width[1,1,:len(m2s)],showextrema=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('y')
v3 = plt.violinplot(rel_hab_m2_01,widths=vp_width[2,1,:len(m2s)],showextrema=False)
for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('b')
v4 = plt.violinplot(rel_hab_m2_11,widths=vp_width[3,1,:len(m2s)],showextrema=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('c')
plt.tick_params(axis='y', labelleft=False, direction='inout')
plt.xticks(ticks=np.arange(1,len(m2s)+1),labels=[str(m2)[:4] for m2 in m2s])
plt.ylim(0,1.15)
plt.xlabel('$m_2$')

plt.subplot(323)
v1 = plt.violinplot(rel_hab_a1_00,widths=vp_width[0,2,:len(a1s)],showextrema=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('r')
v2 = plt.violinplot(rel_hab_a1_10,widths=vp_width[1,2,:len(a1s)],showextrema=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('y')
v3 = plt.violinplot(rel_hab_a1_01,widths=vp_width[2,2,:len(a1s)],showextrema=False)
for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('b')
v4 = plt.violinplot(rel_hab_a1_11,widths=vp_width[3,2,:len(a1s)],showextrema=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('c')
plt.xticks(ticks=np.arange(1,len(a1s)+1),labels=[str(a1)[:4] for a1 in a1s],
           size='x-small')
plt.ylim(0,1.15)
plt.xlabel('$a_1$')

plt.subplot(324)
v1 = plt.violinplot(rel_hab_alpha_00,widths=vp_width[0,3,:len(alphas)],showextrema=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('r')
v2 = plt.violinplot(rel_hab_alpha_10,widths=vp_width[1,3,:len(alphas)],showextrema=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('y')
v3 = plt.violinplot(rel_hab_alpha_01,widths=vp_width[2,3,:len(alphas)],showextrema=False)
for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('b')
v4 = plt.violinplot(rel_hab_alpha_11,widths=vp_width[3,3,:len(alphas)],showextrema=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('c')
plt.tick_params(axis='y', labelleft=False, direction='inout')
plt.xticks(ticks=np.arange(1,len(alphas)+1),labels=[str(alpha)[:4] for alpha in alphas],
           size='x-small')
plt.ylim(0,1.15)
plt.xlabel(r'$\alpha$')

plt.subplot(325)
v1 = plt.violinplot(rel_hab_e1_00,widths=vp_width[0,4,:len(e1s)],showextrema=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('r')
v2 = plt.violinplot(rel_hab_e1_10,widths=vp_width[1,4,:len(e1s)],showextrema=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('y')
v3 = plt.violinplot(rel_hab_e1_01,widths=vp_width[2,4,:len(e1s)],showextrema=False)
for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('b')
v4 = plt.violinplot(rel_hab_e1_11,widths=vp_width[3,4,:len(e1s)],showextrema=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('c')
plt.xticks(ticks=np.arange(1,len(e1s)+1),labels=[str(e1)[:4] for e1 in e1s],
           size='small')
plt.ylim(0,1.15)
plt.xlabel('$e_1$')

plt.subplot(326)
v1 = plt.violinplot(rel_hab_e2_00,widths=vp_width[0,5,:len(e2s)],showextrema=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('r')
v2 = plt.violinplot(rel_hab_e2_10,widths=vp_width[1,5,:len(e2s)],showextrema=False)
for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('y')
v3 = plt.violinplot(rel_hab_e2_01,widths=vp_width[2,5,:len(e2s)],showextrema=False)
for b in v3['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_alpha(.4)
    b.set_facecolor('b')
v4 = plt.violinplot(rel_hab_e2_11,widths=vp_width[3,5,:len(e2s)],showextrema=False)
for b in v4['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_alpha(.4)
    b.set_facecolor('c')
plt.tick_params(axis='y', labelleft=False, direction='inout')
plt.xticks(ticks=np.arange(1,len(e2s)+1),labels=[str(e2)[:4] for e2 in e2s],
           size='small')
plt.ylim(0,1.15)
plt.xlabel('$e_2$')

plt.gcf().text(0.03, 0.5, 'relative habitability', va='center', rotation='vertical')
plt.subplots_adjust(wspace=0,hspace=.3)
plt.savefig('1D_violin_plot')
plt.savefig('1D_violin_plot.pdf',format='pdf')
zplt.show()
