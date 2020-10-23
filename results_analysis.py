from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.cross_validation import LeaveOneOut, cross_val_score, ShuffleSplit, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pyriemann.classification import MDM, TSclassifier, KNearestNeighbor
from joblib import Parallel, delayed
from pyriemann.utils.mean import mean_covariance

from scipy.io import loadmat
from scipy import stats
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
import pandas as pd
from time import time
from joblib import Parallel, delayed

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import texttable as tt

import pickle

if __name__ == '__main__':
    plt.ion()

    # Load results
    # pickle.dump({'results': accuracies_all_subjects, 'lambdas_grid': l, 'n_samples_grid': n},
    #             open('results_riemannian.p', 'wb'))
    results_riemannian = pickle.load(open('./results_riemannian.p', 'rb'))

    methods = {'riemannian': './results_riemannian.p',
               'euclidean': './results_euclidean.p',
               'no_similarity': './results_riemannian_no_similarity.p',
               'rest_similarity': './results_resting_similarity.p',
               'per_class_similarity': './results_per_class_similarity.p'}
    # Show results
    n_subjects = 12
    results = dict()
    accuracies = dict()
    method_results = dict()
    accuracy_matrix_mean = dict()
    best_accuracy = dict()
    best_lambda = dict()
    for method_idx, (method, file) in enumerate(methods.items()):
        fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw=dict(projection='3d'), figsize=(14, 8))
        method_results[method] = list()
        best_accuracy[method] = list()
        best_lambda[method] = list()
        accuracy_matrix_mean[method] = np.zeros((8,6))
        results[method] = pickle.load(open(file, 'rb'))
        l = results[method]['lambdas_grid']
        n = results[method]['n_samples_grid']
        n_samples = n[:, 0]
        lambdas = l[0, :]
        accuracies[method] = results[method]['results']
        ns_test = 3  # Take (ns_test+1) * 4 samples for example sake
        for subject in range(n_subjects):
            method_results[method].append(np.array([np.concatenate(l.T), np.concatenate(n.T), accuracies[method][subject]]).T)
            # print(method_results)
            accuracy_matrix = np.reshape(accuracies[method][subject], l.shape, 1)
            accuracy_matrix_mean[method] += accuracy_matrix
            max_idx = np.argmax(accuracy_matrix[ns_test, :])
            best_accuracy[method].append(accuracy_matrix[ns_test, max_idx])
            best_lambda[method].append(lambdas[max_idx])
            axes[int(subject/4), subject % 4].set_title('subject {}'.format(subject+1))
            surf = axes[int(subject/4), subject % 4].plot_surface(l, n, accuracy_matrix, cmap=cm.coolwarm,
                                                                  linewidth=0, antialiased=False)
            axes[int(subject/4), subject % 4].set_xlabel('lambda')
            axes[int(subject/4), subject % 4].set_xlim(0, 1)
            axes[int(subject/4), subject % 4].xaxis.set_major_locator(LinearLocator(6))
            # axes[int(subject/4), subject % 4].zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            axes[int(subject/4), subject % 4].set_ylabel('#samples')
            if (subject % 4) == 3:
                axes[int(subject/4), subject % 4].set_zlabel('accuracy')
            axes[int(subject/4), subject % 4].set_zlim(0.5, 1)
            axes[int(subject/4), subject % 4].set_yticklabels([str(x) for x in np.array(n_samples) * 4])
        fig.suptitle(method)
        accuracy_matrix_mean[method] /= n_subjects
        # axes[0, 1].annotate(method, (0.5, 1), xytext=(0, 30), textcoords='offset points', xycoords='axes fraction',
        #                     ha='center', va='bottom', size=14)

    # plot mean surf
    for method in methods.keys():
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(l, n, accuracy_matrix_mean[method], cmap=cm.coolwarm,
                               linewidth=0, antialiased=False, vmin=0.58, vmax=0.82)
        ax.set_xlabel('$\lambda$', size='x-large')
        ax.set_ylabel('$n$', size='x-large')
        ax.set_zlabel('accuracy', size='x-large')
        ax.set_yticklabels([str(x) for x in np.array(n_samples) * 4])
        ax.set_zlim(0.58, 0.82)
        plt.subplots_adjust(left=0, bottom=0.02, right=0.92, top=1)

    # T test different methods
    ### CHANGE SIZE OF INPUT TO TEST. MAKE IT MULTI VARIABLE
    ### ASLO TRY WITH Wilcoxon rank-sum: stats.ranksums
    pvalues_simi = list()
    pvalues_riem = list()
    print("P-VALUES:")
    for sub in range(n_subjects):
        statistic, pvalue_similarity = stats.ttest_rel(method_results['euclidean'][sub][:, 2],
                                                       method_results['rest_similarity'][sub][:, 2])
        statistic, pvalue_riemannian = stats.ttest_rel(method_results['euclidean'][sub][:, 2],
                                                       method_results['no_similarity'][sub][:, 2])
        print("subject {}".format(sub+1))
        print("------------")
        print("  E-MDPM -> R-MDPM p-values = {}".format(pvalue_riemannian))
        print("  E-MDPM -> MDWM p-values = {}".format(pvalue_similarity))
        pvalues_simi.append(pvalue_similarity)
        pvalues_riem.append(pvalue_riemannian)

    # 1-WAY ANOVA
    print("1-WAY ANOVA:")
    all_subjects_eucl_accuracy = list()
    all_subjects_riem_accuracy = list()
    all_subjects_simi_accuracy = list()

    for sub in range(n_subjects):
        all_subjects_eucl_accuracy = all_subjects_eucl_accuracy + list(method_results['euclidean'][sub][:, 2])
        all_subjects_riem_accuracy = all_subjects_riem_accuracy + list(method_results['no_similarity'][sub][:, 2])
        all_subjects_simi_accuracy = all_subjects_simi_accuracy + list(method_results['rest_similarity'][sub][:, 2])
    f_value_anova, p_value_anova = stats.f_oneway(all_subjects_eucl_accuracy, all_subjects_riem_accuracy,
                                                  all_subjects_simi_accuracy)  # ANOVA E-MDPM, R-MDPM, and MDWM
    f_value_anova_E_R, p_value_anova_E_R = stats.f_oneway(all_subjects_eucl_accuracy,
                                                          all_subjects_riem_accuracy)  # ANOVA E-MDPM and R-MDPM
    f_value_anova_E_W, p_value_anova_E_W = stats.f_oneway(all_subjects_eucl_accuracy,
                                                          all_subjects_simi_accuracy)  # ANOVA E-MDPM and MDWM
    f_value_anova_R_W, p_value_anova_R_W = stats.f_oneway(all_subjects_riem_accuracy,
                                                          all_subjects_simi_accuracy)  # ANOVA R-MDPM and MDWM

    print("ANOVA E-MDPM, R-MDPM, MDWM f-value: {}, and associated p-value: {}".format(f_value_anova, p_value_anova))
    print("ANOVA E-MDPM and R-MDPM    f-value: {}, and associated p-value: {}".format(f_value_anova_E_R, p_value_anova_E_R))
    print("ANOVA E-MDPM and MDWM      f-value: {}, and associated p-value: {}".format(f_value_anova_E_W, p_value_anova_E_W))
    print("ANOVA R-MDPM and MDWM      f-value: {}, and associated p-value: {}".format(f_value_anova_R_W, p_value_anova_R_W))


    best_accracy_simi = [x for x in best_accuracy['rest_similarity']]  # MDWM (uses resting class similarity as weight)
    best_accracy_riem = [x for x in best_accuracy['no_similarity']]    # Riemannian MDPM (R-MDPM)
    best_accracy_eucl = [x for x in best_accuracy['euclidean']]        # Euclidean MDPM (E-MDPM)
    best_lambda_simi = [x for x in best_lambda['rest_similarity']]
    best_lambda_riem = [x for x in best_lambda['no_similarity']]
    best_lambda_eucl = [x for x in best_lambda['euclidean']]
    pvalues_simi = [x for x in pvalues_simi]
    pvalues_riem = [x for x in pvalues_riem]

    tab = tt.Texttable()

    x = [[], ['accuracy MDWM (w/ sjk)'] + best_accracy_simi,
         ['lambda MDWM'] + best_lambda_simi,
         ['accuracy R-MDPM (w/o sjk)'] + best_accracy_riem,
         ['lambda R-MDPM'] + best_lambda_riem,
         ['accuracy E-MDPM (w/o sjk)'] + best_accracy_eucl,
         ['lambda E-MDPM'] + best_lambda_eucl,
         ['p-values MDWM vs E-MDPM'] + pvalues_riem,
         ['p-values R-MDPM vs E-MDPM'] + pvalues_riem
         ]  # The empty row will have the header
    tab.add_rows(x)
    tab.set_cols_align(['l'] + 12*['r'])
    tab.set_cols_width([19] + 12*[7])
    tab.set_precision(6)

    tab.header(['Methods'] + ['sub {}'.format(x) for x in range(1, 13)])
    print("snapshot of performance w/ vs w/o measure of similarity ")
    print(tab.draw())

    # Area
    array = method_results['euclidean'][0][:, :2]
    array = np.round(array, 1)
    my_index = pd.MultiIndex.from_arrays(array.T, names=['lambdas', 'n_samples'])
    for sub in range(n_subjects):
        df = pd.DataFrame(None, index=my_index, columns=methods.keys())
        for method in methods.keys():
            df[method] = method_results[method][sub][:, 2]
        ax = df.plot.area(stacked=False, title="subject {}".format(sub+1), grid=False, figsize=(12, 4), ylim=[0.4, 1], use_index=True)
        ax.set_ylabel("accuracy")
        # ax.xaxis.set_major_locator(LinearLocator(7))


    # # Label rows and columns
    # for ax, subject in zip(axes[0], ['subject ' + str(x) for x in range(4)]):
    #     ax.set_title(subject, size=14)
    # for ax, method in zip(axes[:, 0], methods.keys()):
    #     ax.set_zlabel(method, size=14, ha='left')
    #
    #     tmp_planes = ax.zaxis._PLANES
    #     ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
    #                         tmp_planes[0], tmp_planes[1],
    #                         tmp_planes[4], tmp_planes[5])
    #     view_1 = (25, -135)
    #     view_2 = (25, -45)
    #     init_view = view_2
    #     ax.view_init(*init_view)

