import moabb
import pyriemann
import numpy as np
import gzip, pickle
# import matplotlib.pyplot as plt

from moabb.paradigms import BaseSSVEP, MotorImagery
from moabb.datasets import SSVEPExo, AlexMI, PhysionetMI, Weibo2014, Ofner2017
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import pairwise_distance, distance

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding

moabb.set_log_level('info')

def aggregate_cov(X, n_subjects, metadata, filterbank=True,
                  s_class='rest', estimator='lwf'):
    """Compute covmat for concatenated signals"""
    subjX = np.array(metadata['subject'])
    cov_list = []
    for s in range(n_subjects):
        s_loc = subjX == (s + 1)
        X_sig = np.array(X[np.logical_and(s_loc, y == s_class)])
        if filterbank:
            sig_ext = X_sig[:, :, :, :-1].transpose((0, 3, 1, 2))
            n_trials, n_freqs, n_channels, n_times = sig_ext.shape
            X_sig = sig_ext.reshape((n_trials, n_channels * n_freqs, n_times))
        X_sig =  np.concatenate([X_sig[i, :, :]
                                  for i in range(X_sig.shape[0])], axis=1)
        cov = Covariances(estimator='lwf').transform(X_sig.reshape((1, *(X_sig.shape))))
        cov_list.append(cov)
    return np.concatenate(cov_list, axis=0)

metric = 'logdet' #'riemann' # 

##############################################################
# Load SSVEP database
paradigm = BaseSSVEP()
SSVEPExo().download(update_path=False, verbose=False)
datasets = SSVEPExo()

X, y, metadata = paradigm.get_data(dataset=datasets)
# with gzip.open('SSVEPExo.pkz', 'wb') as f:
#     o = {'X':X, 'y':y, 'metadata':metadata}
#     pickle.dump(o, f)
# with gzip.open('SSVEPExo.pkz', 'rb') as f:
#     o = pickle.load(f)
# X, y, metadata = o['X'], o['y'], o['metadata']

n_subjects = len(np.unique(np.array(metadata['subject'])))

##############################################################
# Aggregate all signals for a class from subjects
# and compute covmats. Then, compute dist between cov and
# pairwise distance
simi_vect, simi_mat = {}, {}
events = np.unique(y) # ['13', '17', '21', 'rest']
for e in events:
    ev = aggregate_cov(X, n_subjects, metadata, filterbank=True,
                       s_class=e, estimator='lwf')
    pdist = pairwise_distance(ev, metric=metric)
    norm_fact = 1. / np.triu(pdist).sum()
    simi = 1. / norm_fact * np.triu(pdist)
    simi_mat[e] = simi
    simi_vect[e] = simi[simi != 0.]
df = pd.DataFrame(simi_vect)

##############################################################
# Joint plot
for f in events[events != 'rest']:
    g = sns.jointplot(x="rest", y=f, data=df)
    plt.savefig('ssvep-jointplot-{0}-rest-{1}.png'.format(f, metric))

##############################################################
# Laplacian Eigenmaps
fig = plt.figure(figsize=(12, 3))
for i, f in enumerate(events):
    ax = fig.add_subplot(1, 4, i+1)
    lapl = SpectralEmbedding(n_components=2, affinity='precomputed',
                             n_jobs=-1)
    le = lapl.fit_transform(simi_mat[f])
    for i in range(n_subjects):
        ax.scatter(le[i, 0], le[i, 1], label='s'+str(i+1))
    if f is not 'rest': ax.set_title('{0} Hz'.format(f))
    else: ax.set_title('Resting state')
ax.legend()
plt.tight_layout(0.5)
plt.savefig('ssvep-laplacianEigenmaps.png')
plt.close()

#############################################################
# MI datasets
events = ['right_hand', 'feet', 'rest']
paradigm = MotorImagery(events=events, n_classes=3, resample=128)
db = paradigm.datasets
for d in paradigm.datasets:
    dname = d.__dict__['code'].split()[0]
    X, y, metadata = paradigm.get_data(dataset=d)
    simi_vect, simi_mat = {}, {}
    events = np.unique(y)
    n_subjects = len(np.unique(np.array(metadata['subject'])))
    for e in events:
        ev = aggregate_cov(X, n_subjects, metadata, filterbank=False,
                           s_class=e, estimator='lwf')
        pdist = pairwise_distance(ev, metric=metric)
        norm_fact = 1. / np.triu(pdist).sum()
        simi = 1. / norm_fact * np.triu(pdist)
        simi_mat[e] = simi
        simi_vect[e] = simi[simi != 0.]
    df = pd.DataFrame(simi_vect)

    for f in events[events != 'rest']:
        g = sns.jointplot(x="rest", y=f, data=df)
        plt.savefig('{0}-jointplot-{1}-rest-{2}.png'
                    .format(dname, f, metric))

    fig = plt.figure(figsize=(12, 3))
    for i, f in enumerate(events):
        ax = fig.add_subplot(1, 4, i+1)
        lapl = SpectralEmbedding(n_components=2, affinity='precomputed',
                                n_jobs=-1)
        le = lapl.fit_transform(simi_mat[f])
        for i in range(n_subjects):
            ax.scatter(le[i, 0], le[i, 1], label='s'+str(i+1))
        if f is not 'rest': ax.set_title('{0}'.format(f))
        else: ax.set_title('Resting state')
    ax.legend()
    plt.tight_layout(0.5)
    plt.savefig('{0}-laplacianEigenmaps.png'.format(dname))
    plt.close()
