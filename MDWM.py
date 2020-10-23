from pyriemann.utils.geodesic import geodesic 
import numpy

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from joblib import Parallel, delayed

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance


class MDWM (MDM):
    def __init__(self, L=0,  **kwargs):
        """Init."""
        self.L = L
        super().__init__(**kwargs)

    def fit(self, X, y, X_domain, y_domain, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample from the domain. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.classes_ = numpy.unique(y)
        # TODO: ajouter un test pour verifier que y et y_domain
        #       ont les meme classes


        if sample_weight is None:
            sample_weight = numpy.ones(X_domain.shape[0])
             
        if self.n_jobs == 1:
            self.target_means_ = [mean_covariance(X[y == l], 
                                              metric=self.metric_mean)
                                    # sample_weight=sample_weight_target[y == l])
                                        for l in self.classes_]

            self.domain_means_ = [mean_covariance(X_domain[y_domain == l], 
                                                  metric=self.metric_mean,
                                    sample_weight=sample_weight[y_domain == l])
                                        for l in self.classes_]
        else:
            self.target_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X[y == l], metric=self.metric_mean)
                for l in self.classes_)  # sample_weight=sample_weight_target[y == l])
            self.domain_means_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(X_domain[y_domain == l],
                                metric=self.metric_mean,
                                sample_weight=sample_weight[y_domain == l])
                for l in self.classes_)

        self.class_center_ = [geodesic(self.target_means_[i], 
                                       self.domain_means_[i],
                                       self.L, self.metric) 
                for i, _ in enumerate(self.classes_)]

        return self


    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.class_center_)

        if self.n_jobs == 1:
            dist = [distance(covtest, self.class_center_[m], self.metric_dist)
                    for m in range(Nc)]
        else:
            dist = Parallel(n_jobs=self.n_jobs)(delayed(distance)(
                covtest, self.class_center_[m], self.metric_dist)
                for m in range(Nc))

        dist = numpy.concatenate(dist, axis=1)
        return dist
           
           
    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]


    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

