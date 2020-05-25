from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, 
                             GradientBoostingClassifier,
                             RandomForestRegressor,
                             GradientBoostingRegressor)

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
#from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from functools import partial
from sklearn.model_selection import cross_val_score
from tqdm.autonotebook import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer as SI
from sklearn.model_selection  import GridSearchCV

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter, OrderedDict
pd.options.display.float_format = '{:,.4g}'.format
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from imblearn.over_sampling import SMOTE


class Sylreha:
    """
    Sylreha (partial) is a Supervised Learning API for classification 
    models listed within
    """
    
    def __init__(self):
        self.best_models = {}
        self.data_train = {}
        self.accuracy_collection = {}
        self.params = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        #           'alpha': [1e-12, 1e-10, 1e-8, 1e-4, 1e-3,0.1, 0.2,0.4, 0.75,
        #                          1, 1.5, 3, 5, 10],
                  'C': [1e-3, 1e-2, 1e-1, 1, 10, 100],
                  'alpha': [1e-4, 1e-3, 1e-2,0.1, 1, 2, 5, 10],
                  'none': [1],
                  'max_depth': list(range(3, 5))#+[None], 

        }

        self.params_gs = {
            'KNClassifier':{'n_neighbors': [1,3, 5, 7, 9, 11, 13, 15]},
            'LogisticRegression L1': {'C': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10]},
            'LogisticRegression L2': {'C': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]},
            'SVM L1': {'C': [1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]},
            'SVM L2': {'C': [1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]},
            'SVC RBF': {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 500, 1000], 
                       'gamma': [1e-2,0.1, 1, 2, 3, 5, 10,
                                15, 20, 100, 500, 1000 ]},
            'SVC Poly': {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 500, 1000], 
                         'degree': [3],
                         'coef0': [0,5,10, 100]},
            'Random Forest': {'n_estimators': [300], 
                              'max_features' : [None, 'sqrt', 'log2', 'auto'],
                              'max_depth' : list(range(6, 10))#+[None],
                             },
            'Decision Tree':{'max_depth' : range(3, 5), 
                             'max_features' : [None, 'sqrt', 'log2', 'auto']},
            'GBM': {'n_estimators': [300], 
                              'max_features' :  [None, 'sqrt', 'log2', 'auto'],
                              'learning_rate': np.arange(0.05, 0.16, 0.01).tolist(),
                              'max_depth' : list(range(6, 10))#+[None],
                             },
            'Bayes_Bernouli': {'alpha': [1e-12, 1e-10, 1e-8, 1e-4, 1e-3,0.1, 0.2,0.4, 0.75,
                                 1, 1.5, 3, 5, 10]},
            'Bayes_Multionomial': {'alpha': [1e-12, 1e-10, 1e-8, 1e-4, 1e-3,0.1, 0.2,0.4, 0.75,
                                 1, 1.5, 3, 5, 10]},
            'Bayes_Gaussian': {'var_smoothing': [1e-12, 1e-10, 1e-9, 1e-8, 1e-4, 1e-3,0.1, 0.2,0.4, 0.75,
                                 1] }
        #     'XGBoost': {'learning_rate': np.arange(0.05, 0.16, 0.01).tolist(),
        #                'max_depth' : list(range(3, 10)),
        #                'n_estimators': [500]}
        }



    def impute_this(self, df, mean=[], median=[], most_frequent=[], fv=np.nan):
        """

        """
        if mean != []:
            for col_name in mean:
                imr = SI(strategy='mean', fill_value=fv)
                imputed = imr.fit_transform(df[[col_name]])
                df[col_name] = imputed
        if median != []:
            for col_name in median:
                imr = SI(strategy='median', fill_value=fv)
                imputed = imr.fit_transform(df[[col_name]])
                df[col_name] = imputed
        if most_frequent != []:
            for col_name in most_frequent:
                imr = SI(strategy='most_frequent', fill_value=fv)
                imputed = imr.fit_transform(df[[col_name]])
                df[col_name] = imputed
        return df


    # # Visualization Functions

    def plot_target_counts(self, y, regress=False, series=True):
        """
            Input:
            -------
            y: target input
            regress: boolean True if regression target
            series: boolean True if y input is a pandas dataframe input

            Output:
            --------
            Plots either PCC or Population Distribution of target
        """
        fig, ax = plt.subplots(figsize=(10,5))

        if not regress:
            sns.countplot(y, ax=ax);
            ax.set(title='Target Distribution')
            counts = y.value_counts() if series else y
            print(counts)
            pcc = np.sum((y.value_counts()/len(y))**2)
            print(f"1.25 * Proportional Chance Criterion {pcc*1.25}")
        else: 
            ax.plot(y.cumsum()/y.sum())
            ax.set(ylabel='%', xlabel='targets', title='Population Distribution') 



    def plot_accuracy(model_name, setting_name, flag=False, gs=False):
        model_set = accuracy_collection[model_name]
        if not gs:
            testdf = pd.DataFrame(zip(np.array(model_set['train']).flatten(), 
                                       np.array(model_set['test']).flatten(), 
                                       np.array(model_set[setting_name]).flatten()))
            train_ = testdf.groupby(2)[0].mean()
            test_ = testdf.groupby(2)[1].mean()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x=2, y=0, 
                              label="Training Accuracy", data=testdf,
                              ci='sd',
                              estimator='mean', markers=True, dashes=False, ax=ax);
            sns.lineplot(x=2, y=1,
                              label="Test Accuracy", data=testdf, 
            #                       err_style="bars", 
                              ci='sd',
                              estimator='mean',
                              markers=True, dashes=False, ax=ax);
            title = f"""Accuracy vs {setting_name}
            Train Acc: {train_[test_.idxmax()]}   Test Acc: {test_[test_.idxmax()]}"""
            ax.set(xlabel=setting_name, ylabel='Accuracy', 
                       title=title);
            if flag:
                ax.set_xscale('log')
                ax.set_xlabel(f"Log of {setting_name}")
            ax.axvline(test_.idxmax(), ls='--', c='gray')
            ax.legend();



    # # Compute Functions

    def calc_accuracy_params(self, n_trials, key, X, y, model, setting_name, 
                             settings, scaler=None, balance=False, test_size=0.25):
        """
        Runs the model with their settings in for loop

        Input:
        -------
        n_trials: number of trials to run
        key: model name to run
        X: feature array
        y: target array
        model: model syntax
        setting_name: hyperparameter name
        settings: hyperparameter values
        scaler: scaler syntax i.e MinMaxScaler 

        Ouput:
        -------
        global accuracy_collections: includes train, test, paramter, 
            coefficient, and probability arrays
        returns tuple of above values
        """
        train_ = []
        test_ = []
        coefs_ = []
        settings = []
        probs_ = []
        report_ = []
#         models_
#         global accuracy_collection

        self.accuracy_collection[key] = {'train':[], 'test':[], 'params':[],
                                    'coefs':[], 'probabilities':[], 'report':[]}
        with tqdm(total=len(settings)*n_trials) as pbar:
            trial = 0
            for s in settings: 
                train_acc = []
                test_acc = []
                best_acc = 0
                for n in range(n_trials):
                    X_train, X_test,y_train, y_test = train_test_split(X, y, 
                                                                test_size=test_size,
                                                                random_state=n*8)

                    if balance:
                        sm = SMOTE(random_state=42)
                        X_train, y_train = sm.fit_resample(X_train, y_train)

                    if scaler: 
                        scaler.fit(X_train)
                        X_train = scaler.transform(X_train)
                        X_test = scaler.transform(X_test)
    #                 clf = model(**{setting_name: s, 'n_jobs':-1})
                    try:
                        if setting_name != 'none':
                            clf = model(**{setting_name: s, 'n_jobs':-1})
                        else: 
                            clf = model()
                    except TypeError:
                        clf = model(**{setting_name: s})
                    clf.fit(X_train, y_train) 
                    train_acc.append(clf.score(X_train, y_train))
                    test_acc.append(clf.score(X_test, y_test))
                    probs_ = clf.predict_proba(X_test)
                    pbar.update(1)

                if np.mean(test_acc) > best_acc and hasattr(clf, 'coef_'):
    #                 print(9999999999999)
                    best_acc = np.mean(test_acc)
                    coefs_ = clf.coef_
                    self.accuracy_collection[key]['coefs'].append(coefs_)

                elif np.mean(test_acc) > best_acc and hasattr(clf, 'feature_importances_'):
                    best_acc = np.mean(test_acc)
                    coefs_ = clf.feature_importances_
                    self.accuracy_collection[key]['coefs'].append(coefs_)

                self.accuracy_collection[key]['train'].append(train_acc)
                self.accuracy_collection[key]['test'].append(test_acc)
                self.accuracy_collection[key][setting_name].append([s]*n_trials)
                self.accuracy_collection[key]['probabilities'].append(probs_)

                train_.append(np.mean(train_acc))
                test_.append(np.mean(test_acc))

                trial += 1

        return (train_, test_, settings, coefs_, probs_)

    def calculate_accuracy_params_gs(self, model, key, X, y, scaler=None, balance=False, n_trials=1, test_size=0.25):
        """
        Runs the model with their GridSearchCV params

        Input:
        -------
        n_trials: number of trials to run
        key: model name to run
        X: feature array
        y: target array
        model: model syntax
        setting_name: hyperparameter name
        settings: hyperparameter values
        scaler: scaler syntax i.e MinMaxScaler 

        Ouput:
        -------
        global accuracy_collections: includes train, test, paramter, 
            coefficient, and probability values
        returns tuple of above values
        """
        train_ = []
        test_ = []
        coefs_ = []
        settings = []
        probs_ = []
        report_ = []
#         global accuracy_collection 
#         global best_models
#         global data_train
        self.accuracy_collection[key] = {'train':[], 'test':[], 'params':[],
                                    'coefs':[], 'probabilities':[], 'report':[]}
        with tqdm(total=n_trials) as pbar:
            for n in range(n_trials):
                X_train, X_test,y_train, y_test = train_test_split(X, y, 
                                                                test_size=test_size, random_state=n)
                if balance:
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)

                if scaler: 
                    scaler.fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)
                clf = GridSearchCV(model, param_grid= self.params_gs[key],
                                   iid=False, cv=5, verbose=5, refit=True, 
                                   scoring='recall')
                clf.fit(X_train, y_train)

                clf_in = clf.best_estimator_
                train_acc = clf_in.score(X_train, y_train)
                test_acc = clf_in.score(X_test, y_test)

                probs_ = 'NA'

                try:
                    test_max = max(test_)
                except:
                    test_max = 0

                if hasattr(clf_in, 'coef_') and test_acc > test_max:
                    coefs_ = clf_in.coef_
                if hasattr(clf_in, 'feature_importances_') and test_acc > test_max:
                    coefs_ = clf_in.feature_importances_
                if hasattr(clf, 'predict_proba') and test_acc > test_max:
                    probs_ = clf.predict_proba(X_test)    
                if test_acc > test_max:
                    model = clf_in
                    traindata = X_train
                    settings = clf.best_params_
                    y_pred = clf_in.predict(X_test)
                    report_ = classification_report(y_test, y_pred)

                train_.append(train_acc)
                test_.append(test_acc)
                pbar.update(1)

        mean_train = np.mean(train_)
        mean_test = np.mean(test_)
        idx = np.argmax(test_)

        self.best_models[key] = model
        self.data_train[key] = traindata
        self.accuracy_collection[key]['train'] = mean_train
        self.accuracy_collection[key]['test'] = mean_test
        self.accuracy_collection[key]['params'] = settings
        try:
            self.accuracy_collection[key]['coefs'] = coefs_
        except:
            self.accuracy_collection[key]['coefs'] = "NA"
        self.accuracy_collection[key]['probabilities'] = probs_
        self.accuracy_collection[key]['report'] = report_

        return (mean_train, mean_test, settings, coefs_, probs_)

    def we_classify(self, X, y, features, scaler=None, balance=False, mods='All',
                    gs=False, n_trials=1, random_state=0, test_size=0.25):    
        """
            Calls the classifier main script and preps the models for the 
            calc_accuracy_params function

            Input:
            -------
            X: feature array
            y: target array
            features: feature names
            scaler: scaler syntax i.e MinMaxScaler
            mods: list of models to run
            gs: boolean True if to use GridSearchCV 
            n_trials: 1 if gs otherwise specify integer
            random_state: random seed (default None) 
            test_size: size of test set (default 0.25)

            Output:
            --------
            Dataframe of performance results of the run models
        """
        classify_models = {
            'KNClassifier': (KNeighborsClassifier, 'n_neighbors'),
            'LogisticRegression L1': (partial(LogisticRegression, 
                                              penalty='l1',
                                              solver='liblinear', n_jobs=-1, max_iter=1000), 
                                      'C'),
            'LogisticRegression L2': (partial(LogisticRegression, 
                                              penalty='l2',
                                              solver='liblinear', n_jobs=-1, max_iter=1000), 
                                      'C'),
            'SVM L1': (partial(LinearSVC, penalty='l1', loss='squared_hinge',
                                   dual=False), 
                                      'C'),
            'SVM L2': (partial(LinearSVC, penalty='l2',loss='squared_hinge',
                                   dual=False), 
                                      'C'),
            'Decision Tree': (DecisionTreeClassifier, 'max_depth'),
            'Bayes_Bernouli': (BernoulliNB, 'alpha'), 
            'Bayes_Multionomial': (MultinomialNB, 'alpha'),
            'Bayes_Gaussian': (GaussianNB, 'none')
        }

        if not gs:
            classify_gs_models = {
                'SVC Poly': SVC(kernel='poly', gamma='scale', max_iter=1000), 
                'SVC RBF': SVC(kernel='rbf', max_iter=1000), 
                'Random Forest': RandomForestClassifier(), 
                'GBM': GradientBoostingClassifier(), 
            }

        else:
            classify_gs_models = {
                'KNClassifier': KNeighborsClassifier(),
                'LogisticRegression L1': LogisticRegression(penalty='l1',
                                                  solver='liblinear', n_jobs=-1, max_iter=1000),
                'LogisticRegression L2': LogisticRegression(penalty='l2',
                                                  solver='liblinear', n_jobs=-1, max_iter=1000),
                'SVM L1': LinearSVC(penalty='l1', loss='squared_hinge',
                                       dual=False),
                'SVM L2': LinearSVC(penalty='l2', loss='squared_hinge',
                                       dual=False),
                'Bayes_Bernouli': BernoulliNB(),
                'Bayes_Multionomial': MultinomialNB(),
                'Bayes_Gaussian': GaussianNB(),
                'SVC Poly': SVC(kernel='poly', gamma='scale', max_iter=1000), 
                'SVC RBF': SVC(kernel='rbf', max_iter=1000), 
                'Decision Tree': DecisionTreeClassifier(random_state=random_state),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_jobs=-1), 
                'GBM': GradientBoostingClassifier(random_state=random_state) 
    #             'XGBoost': XGBClassifier(random_state=random_state, n_jobs=-1)
            }

        if mods == 'All':
            classify_gs_models_ = classify_gs_models
            classify_models_ = classify_models
        else:
            classify_gs_models_ = {k:v for k,v in classify_gs_models.items()
                                  if k in mods}
            classify_models_ = {k:v for k,v in classify_models.items()
                                  if k in mods}

        res = []
        if not gs:
            for key, (model, setting) in classify_models_.items():
                print(f'Running {key}')
                train, test, param, coefs = self.calc_accuracy_params(key, X, y, 
                                                                 model, setting, 
                                                                 self.params[setting],
                                                                 scaler=scaler, 
                                                                 balance=balance,
                                                                 n_trials=n_trials, 
                                                                 test_size=test_size)
        #         print(coefs)   
                ix = np.argmax(test)
                test_acc = test[ix]
                train_acc = train[ix]
                s = param[ix]
        #         print(np.argmax(coefs))
                ix_coefs = (np.unravel_index(np.argmax(coefs), coefs.shape)
                            [int(len(coefs.shape) == 2)]
                                if len(coefs) else None)
                predictor = features[ix_coefs] if ix_coefs else None
                res.append((key, train_acc, test_acc, {setting:s}, predictor))

        #SVM Poly and Rbf, Ensemble
        for key, value in classify_gs_models_.items():
            print(f'Running {key}')

            train, test, param, weights, probs = self.calculate_accuracy_params_gs(value, key, 
                                                                              X, y, 
                                                                              scaler=scaler, balance=balance, 
                                                                              n_trials=n_trials, test_size=test_size)
            if weights != 'NA':
                ix = np.argmax(weights)
                predictor = features[ix]
            else: 
                predictor = 'NA'
            probs_flag = True if probs != 'NA' else False
            res.append((key, train, test, param, predictor, probs_flag))

    #     print(res)

        return pd.DataFrame(res, columns=['Model', 
                                          'Training accuracy', 
                                          'Test accuracy',
                                           'Parameters',
                                           'Top Predictor',
                                            'probs'])
    def we_regress(n_trials, X, y, features, flag=False, scaler=None):
    
        regress_models = {
            'KNRegressor': (KNeighborsRegressor, 'n_neighbors'),
            'Lasso': (partial(Lasso,  max_iter=10000), 'alpha'),
            'Ridge': (Ridge, 'alpha')
        }

        res = []
        #separate Lasso for KNN 
        if flag:
            train, test, param, coefs = calc_accuracy_params(n_trials, 'Lasso', X, y, 
                                                                 Lasso, 'alpha', 
                                                                 params['alpha'])
            X = np.delete(X, np.argwhere(abs(coefs) == 0), 1)
        #for the rest
        for key, (model, setting) in regress_models.items():
            train, test, param, coefs = calc_accuracy_params(n_trials, key, X, y, 
                                                             model, setting, 
                                                             params[setting],
                                                             scaler = scaler) 

            ix = np.argmax(test)
            test_acc = test[ix]
            s = param[ix]
            predictor = features[np.argmax(coefs)] if len(coefs) else None


            res.append((key, test_acc, s, predictor)) 
    #                     rf_test_acc, rf_s, rf_predictor))
        return pd.DataFrame(res, columns=['model', 'acc', setting,
                                           'predictor',
    #                                        'rf_acc', setting,
    #                                        'rf_predictor'
                                         ])
    # we_regress(2, X, y)



