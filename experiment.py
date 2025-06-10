from sklearn.model_selection import train_test_split
from Datasets import load_dataset, simulate_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import models
import numpy as np
from scipy.stats import binom


def score(y_pred, y):
    return y_pred

def train_models(X, y, p0):
    classifier = CalibratedClassifierCV(DecisionTreeClassifier(class_weight=None, min_samples_leaf=50))
    classifier = classifier.fit(X, y)
    em = models.EM(classifier, p0)
    em.fit(X,y)
    em_conditional = models.EM_conditional(classifier, p0)
    em_conditional.fit(X,y)
    t_test = models.T_test(classifier, p0)
    t_test.fit(X,y)
    return [em, em_conditional, t_test]

def sample_at_prev(X, y, prev, seed):
    n = 50
    n_pos = binom.rvs(n, prev, random_state=seed)
    n_neg = n - n_pos

    i_pos = np.where(y==1)[0]
    i_neg = np.where(y==0)[0]
    
    np.random.seed(seed)
    choice_neg = np.random.choice(i_neg, n_neg, replace=True )
    choice_pos = np.random.choice(i_pos, n_pos, replace=True )
    choices = np.concatenate([choice_neg, choice_pos])
    return X[choices, :], y[choices]
    

def run_trial(i_dataset, prev, seed):
    if i_dataset == 4:
        X, y = simulate_dataset(seed)
    X, y = load_dataset(i_dataset)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state=seed)
    models = train_models(XTrain, yTrain, prev)
    XSampled, ySampled = sample_at_prev(XTest, yTest, prev, seed)
    scores_fpr = {f"{model.__class__.__name__}_fpr" : score(model.predict(XSampled), prev) for model in models}
    models = train_models(XTrain, yTrain, prev)
    XSampled, ySampled = sample_at_prev(XTest, yTest, prev + 0.1, seed)
    scores_tpr = {f"{model.__class__.__name__}_tpr" : score(model.predict(XSampled), prev) for model in models}
    return scores_tpr | scores_fpr