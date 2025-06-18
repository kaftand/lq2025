import copy
from random import sample
import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import KFold
from scipy.stats import ttest_1samp
from scipy.stats import binom


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
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
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
        #print(f"p_s: {p_s}, chi: {chi}")
        #if p_s < self.p0:
        #   return False
        return chi2.cdf(chi, 1) > 0.95

    def predict_chi(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
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
        return chi

    def predict_p(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
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
        return p_s

class EM_V2():
    def __init__(self, trained_classifier, p0):
        self.classifier = trained_classifier
        self.p0 = p0

    def fit(self, X_Train, y):
        self.y_train = y
        self.X_train = X_Train

    def predict(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        p_s = self.p0#np.mean(ps_target_proj)
        ps_target = ps_target_proj
        p_target = self.p0
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
        chi = 2 * np.sum(np.log(p_w_x_den))
        #print(f"p_s: {p_s}, chi: {chi}")
        #if p_s < self.p0:
        #   return False
        return chi2.cdf(chi, 1) > 0.95

    def predict_chi(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
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
        return chi

    def predict_p(self, X, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
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
        return p_s

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
        p_target = np.mean(self.classifier.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        p_s = self.p0#np.mean(ps_target_proj)
        ps_target = ps_target_proj
        p_target = self.p0
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

        return p_s

    def predict_ci(self, X):
        boot_res = []
        for i_boot in range(1000):
            i_thisboot = np.random.choice(X.shape[0], X.shape[0], replace=True)
            boot_res.append(self.predict_boot(X[i_thisboot,:]))

        return np.quantile(boot_res, 0.025), np.quantile(boot_res, 0.975)
    def predict(self, X):
        lb, ub = self.predict_ci(X)

        return (lb > self.p0) or (ub < self.p0)



class EM_prob_sampled():
    def __init__(self, trained_classifier, p0):
        self.classifier = trained_classifier
        self.p0 = p0

    def fit(self, X, y):
        self.y_train = y
        self.X_train = X
        self.boot_models = []
        for _ in range(500):
            i_thisboot = np.random.choice(X.shape[0], X.shape[0], replace=True)
            mdl = copy.deepcopy(self.classifier).fit(X[i_thisboot,:], y[i_thisboot])
            self.boot_models.append(mdl)
            

    def predict_sample(self, X, clf, threshold = 0.00001, maxiter=10_000):
        p_target = np.mean(self.y_train)#np.mean(clf.predict_proba((self.X_train))[:,1])##
        delta_p = np.inf
        p_s = p_target
        ps_target = clf.predict_proba(X)[:,1]
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
        return p_s
        #print(f"p_s: {p_s}, chi: {chi}")
        #if p_s < self.p0:
        #   return False
        return chi2.cdf(chi, 1) > 0.95

    def predict(self, X, threshold = 0.00001, maxiter=1000):
        preds = [ ]
        for clf in self.boot_models * 2:
            i_thisboot = np.random.choice(X.shape[0], X.shape[0], replace=True)
            pred = self.predict_sample(X[i_thisboot,:], clf, threshold=threshold, maxiter=maxiter)
            preds.append(pred)
        preds = np.array(preds)
        return np.mean(preds > self.p0 ) > 0.
        

def sample_at_prev(X, y, prev, seed, n):
    n=100
    n_pos = binom.rvs(n, prev, random_state=seed)
    n_neg = n - n_pos

    i_pos = np.where(y==1)[0]
    i_neg = np.where(y==0)[0]
    
    np.random.seed(seed)
    choice_neg = np.random.choice(i_neg, n_neg, replace=True )
    choice_pos = np.random.choice(i_pos, n_pos, replace=True )
    choices = np.concatenate([choice_neg, choice_pos])
    return X[choices, :], y[choices]

class EM_empircal_lr():
    def __init__(self, trained_classifier, p0, N_test):
        self.classifier = trained_classifier
        self.p0 = p0
        self.boot_chi = []
        self.N_test = N_test

    def fit(self, X, y):
        self.y_train = y
        self.X_train = X
        all_choices = set(range(X.shape[0]))
        for _ in range(5_000):
            i_thisboot = np.random.choice(X.shape[0], X.shape[0], replace=True)
            i_oob = list(all_choices - set(i_thisboot))
            mdl = copy.deepcopy(self.classifier).fit(X[i_thisboot,:], y[i_thisboot])
            X_sample, y_sample = sample_at_prev(X[i_oob, :], y[i_oob], self.p0, None, self.N_test)
            chi = self.predict_chi(X_sample, X[i_thisboot,:], mdl)
            chi = self.predict_chi(X_sample, X[i_thisboot,:], mdl)
            self.boot_chi.append(chi)
        self.chi_hat = np.quantile(self.boot_chi, 0.95)#*.632  + 0.368 * chi2.ppf(0.95, df=1)
        print(self.chi_hat)
            
    def predict_chi(self, X, X_train, mdl, threshold = 0.00001, maxiter=10_000):
        p_target = np.mean(mdl.predict_proba((X_train))[:,1])#np.mean(self.y_train)#
        delta_p = np.inf
        p_s = p_target
        ps_target = self.classifier.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        p_s = self.p0#np.mean(ps_target_proj)
        ps_target = ps_target_proj
        p_target = self.p0
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
        chi = 2 * np.sum(np.log(p_w_x_den))#(np.sum(np.log(ps_target_proj)) - 
                  #  np.sum(np.log(p_w_x)) + 
                  #  X.shape[0] * np.log(p_s) - 
                  #  X.shape[0] * np.log(self.p0))
        return chi

    def predict_sample(self, X, clf, threshold = 0.00001, maxiter=1000):
        p_target = np.mean(clf.predict_proba((self.X_train))[:,1])#np.mean(self.y_train)#
        delta_p = np.inf
        p_s = p_target
        ps_target = clf.predict_proba(X)[:,1]
        ps_target[ps_target >= 1] = 0.9999
        ps_target[ps_target <= 0] = 0.0001
        ps_target_proj = project(ps_target, self.p0, p_target)
        p_s = self.p0
        ps_target = ps_target_proj
        p_target = self.p0
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
        return p_s
        #print(f"p_s: {p_s}, chi: {chi}")
        #if p_s < self.p0:
        #   return False
        return chi2.cdf(chi, 1) > 0.95

    def predict(self, X, threshold = 0.00001, maxiter=1000):
        chi = self.predict_chi(X, self.X_train, self.classifier)
        return  chi > self.chi_hat