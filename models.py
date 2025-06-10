import copy
import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import KFold
from scipy.stats import ttest_1samp


def project(ps_target, p_transform, p_train):
    p_w_x_double_hat = (p_transform / p_train) * ps_target
    p_notw_x_double_hat = (1 - p_transform) / (1 - p_train) * (1 - ps_target)
    p_w_x = p_w_x_double_hat / (p_w_x_double_hat + p_notw_x_double_hat)
    return p_w_x
       

class EM():
    def __init__(self, trained_classifier, p0):
        self.classifier = trained_classifier
        self.p0 = p0

    def fit(self, X_Train, y):
        self.y_train = y
        self.X_train = X_Train

    def predict(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.y_train)#np.mean(self.classifier.predict_proba((self.X_train))[:,1])
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        it = 0
        while((delta_p > threshold) & (it < maxiter)):
            p_w_x_num = (p_s / p_target) * ps_target
            p_w_x_den = p_w_x_num + ((1-p_s) / (1-p_target)) * (1-ps_target)
            p_w_x = p_w_x_num / p_w_x_den
            p_s_p_1 = np.mean(p_w_x)
            delta_p = abs( p_s - p_s_p_1)
            p_s = p_s_p_1
            it += 1
            if it >= maxiter:
                print(f'reached max iter with delta = {delta_p}')

        chi = 2 * (np.sum(np.log(ps_target_proj)) - 
            np.sum(np.log(p_w_x)) + 
            X.shape[0] * np.log(p_s) - 
            X.shape[0] * np.log(self.p0))
        print(f"p_s: {p_s}, chi: {chi}")
        #if p_s < self.p0:
        #   return False
        return chi2.cdf(chi, 1) > 0.95


class EM_conditional():
    def __init__(self, trained_classifier, p0):
        self.classifier = trained_classifier
        self.p0 = p0

    def fit(self, X_Train, y):
        self.y_train = y
        self.X_train = X_Train

    def predict(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        it = 0
        while((delta_p > threshold) & (it < maxiter)):
            p_w_x_num = (p_s / p_target) * ps_target
            p_w_x_den = p_w_x_num + ((1-p_s) / (1-p_target)) * (1-ps_target)
            p_w_x = p_w_x_num / p_w_x_den
            p_s_p_1 = np.mean(p_w_x)
            delta_p = abs( p_s - p_s_p_1)
            p_s = p_s_p_1
            it += 1

        chi = 2 * (np.sum(np.log(ps_target_proj)) - 
            np.sum(np.log(p_w_x)) + 
            X.shape[0] * np.log(p_s) - 
            X.shape[0] * np.log(self.p0))
        if p_s < self.p0:
           return False
        return chi2.cdf(chi, 1) > 0.9

class T_test():
    def __init__(self, trained_classifier, p0):
        self.pretrained_classifier = trained_classifier
        self.p0 = p0

    def fit(self, X_Train, y):
        self.y_train = y
        self.X_train = X_Train
        kf = KFold(n_splits=10)
        val_res = []
        val_ys = []
        for train, test in kf.split(X_Train):
            val_res.append(
                        copy.deepcopy(
                            self.pretrained_classifier)\
                        .fit(X_Train[train, :], y[train])\
                        .predict_proba(X_Train[test,:])[:,1]
            )
            val_ys.append(y[test])
        
        val_ys = np.concatenate(val_ys)
        val_res = np.concatenate(val_res)
        self.avg_p_true = np.mean(val_res[val_ys==1])
        self.avg_p_false = np.mean(val_res[val_ys==0])

        self.mean_prob_0 = self.p0 * self.avg_p_true + (1 - self.p0) * self.avg_p_false

    def predict(self, X):
        probs = self.pretrained_classifier.predict_proba(X)[:,1]
        return ttest_1samp(probs, self.mean_prob_0, alternative='greater').pvalue < 0.05

class EM_boot():
    def __init__(self, trained_classifier, p0):
        self.classifier = trained_classifier
        self.p0 = p0

    def fit(self, X_Train, y):
        self.y_train = y
        self.X_train = X_Train

    def predict_boot(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        it = 0
        while((delta_p > threshold) & (it < maxiter)):
            p_w_x_num = (p_s / p_target) * ps_target
            p_w_x_den = p_w_x_num + ((1-p_s) / (1-p_target)) * (1-ps_target)
            p_w_x = p_w_x_num / p_w_x_den
            p_s_p_1 = np.mean(p_w_x)
            delta_p = abs( p_s - p_s_p_1)
            p_s = p_s_p_1
            it += 1

        return p_s

    def predict(self, X):
        boot_res = []
        for i_boot in range(2000):
            i_thisboot = np.random.choice(X.shape[0], X.shape[0], replace=True)
            boot_res.append(self.predict_boot(X[i_thisboot,:]))

        return np.quantile(boot_res, 0.05) > self.p0