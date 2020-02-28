#  from importlib import reload
# reload(rb); from  scikit_tester.data_robot import *

import numpy as np
import pandas as pd
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class DrLogisticRegression:
    """DATA ROBBOT wraper around a regularized LogisticRegression estimator"""
    def __init__(self, remove_columns=[], replace_nulls={},
                 penalty=None, dual=None, tol=None, C=None,
                 fit_intercept=None, intercept_scaling=None, class_weight=None,
                 random_state=None, solver=None, max_iter=None,
                 multi_class=None, verbose=None, warm_start=None, n_jobs=None,
                 l1_ratio=None):
        # the problem requierments called for a regularized LogisticRegression
        # which means:  C<1. The user can override this value for a specific problem
        self.C = 0.5
        self.solver='liblinear'

        # a new parameter that enables user to discard columns: for 'loans' problem, it is column 'Id'
        self.remove_columns = remove_columns

        # default values for nulls in specific columns
        self.replace_nulls = replace_nulls

        # We wish to expose all of the LogisticRegression parameters to the user
        # of the Wrapper. We use the same default values as the LogisticRegression
        # Chose not to expose the values though since it would require us to maintain
        # those defaults each time LogisticRegression changes them. Instead we
        # give the user the possibility to set those parameters and pass on the
        # values only if they are not None, otherwise we rely on the
        # LogisticRegression defaults
        args = {}
        locs = locals().copy()
        locs.pop('self', None)
        locs.pop('args', None)
        for arg in locs:
            if arg in ['remove_columns', 'replace_nulls']:
               continue
            if locs[arg] is not None:
                args[arg] = locs[arg]
        if 'C' not in args:
            args['C'] = self.C
        if 'solver' not in args:
            args['solver'] = self.solver

        self.clf = LogisticRegression(**args)

    def fit(self, X, y):
        """
        :param X:  features DataFrame
        :param y:  classification labels
        :return:   none
        """
        the_X, the_y = self._df_2_np_xy_train(X, y)
        self.clf = self.clf.fit(the_X, the_y)

    def predict(self, X):
        """
        :param X: features DataFrame
        :return:  labels vector numpy array
        """
        the_X = self._df_2_np_x(X)
        return self.clf.predict(the_X)

    def predict_proba(self, X):
        """
        :param X: features DataFrame
        :return:  numpy array. matrix where each row has the probability
                  for each class label
        """
        the_X = self._df_2_np_x(X)
        return self.clf.predict_proba(the_X)

    def evaluate(self, X, y):
        """
        :param X:  features DataFrame
        :param y:  classification labels
        :return:   dict. Example: {'f1_score': 0.3, logloss: 0.7}
        """
        the_X, the_y = self._df_2_np_xy(X, y)
        y_pred =  self.clf.predict(the_X)

        f1 = f1_score(the_y, y_pred)

        proba = self.predict_proba(X)
        lf = log_loss(the_y, proba)

        res = {'f1_score': round(f1, 2), 'logloss': round(lf, 2)}
        return res

    def tune_parameters(self, X, y):
        """
        :param X:  features DataFrame
        :param y:  classification labels
        :return:   dict.
        """
        the_X, the_y = self._df_2_np_xy_train(X, y)

        param_grid = {'C': [0.5],
                      'tol': [0.0001, 0.001, 0.01],
                      'fit_intercept': [True, False],
                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                      }
        scoring = ["f1", "neg_log_loss", "recall", "precision"]
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        grid = GridSearchCV(LogisticRegression(), param_grid=param_grid,
                            scoring=scoring, refit="f1", cv=kfold,
                            return_train_score=True)
        grid.fit(the_X, the_y)
        res = grid.best_params_
        f1 = grid.best_score_
        logloss = grid.cv_results_['mean_test_neg_log_loss'][grid.best_index_]
        score = {'f1_score': round(f1,2), 'logloss': round(logloss, 2)}
        res['scores'] = score
        return res

    def _df_2_np_xy_train(self, df, y):
        """ creates an X data features numpy arrays from a pandas dataframe
            X will be a valid input for a scikit-learn estimator
            input df : pandas dataframe, y : numpy array
            return X, y : numpy arrays
        """

        # we do not know the problem domain for which this classifier will be used.
        # so we cannot assume default values for NULL cells. Need to drop these rows.
        # But we do allow the user to pass default values for null in specific columns

        # these new column name should be unique enough
        df0 = df.copy()
        for c in self.remove_columns:
            if c in df0.columns:
                df0.drop(c, axis=1, inplace=True)
        for c in self.replace_nulls:
            if c in df0.columns:
                df0.fillna({c: self.replace_nulls[c]}, inplace=True)


        df0['data_robot_logistic_regression_label'] = y
        df1 = df0.dropna()
        the_y = df1['data_robot_logistic_regression_label'].values
        df2 = df1.drop('data_robot_logistic_regression_label', axis=1)

        # we deal with categorical columns using pandas.get_dummies
        # this has the downside of not handling categorical columns which are numbers
        # dealing with those columns with sklearn OneHotEncoder and ColumnTransformer
        # requires the user to specify those columns.
        # ! TBD - user input for numeric categorical columns in future version
        df3 = pd.get_dummies(df2)
        self._columns = df3.columns
        X = df3.values

        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        # an input with many categorical columns will generate many dummy columns,
        # which implies many 0 values. This will justify using a scipy sparse matrix
        # instead of a numpy array
        # ! TBD use sparse matrix for features when many categorical columns
        return X, the_y

    def _df_2_np_xy(self, df, y):
        """ creates an X data features numpy arrays from a pandas dataframe
            X will be a valid input for a scikit-learn estimator
            input df : pandas dataframe, y : numpy array
            return X, y : numpy arrays
        """

        # we do not know the problem domain for which this classifier will be used.
        # so we cannot assume default values for NULL cells. Need to drop these rows.
        # We also receive default values for columns with NULLs from the user

        df0 = df.copy()
        for c in self.remove_columns:
            if c in df0.columns:
                df0.drop(c, axis=1, inplace=True)
        for c in self.replace_nulls:
            if c in df0.columns:
                df0.fillna({c: self.replace_nulls[c]}, inplace=True)

        # this new column name should be unique enough
        df0['data_robot_logistic_regression_label'] = y
        df1 = df0.dropna()
        the_y = df1['data_robot_logistic_regression_label'].values
        df2 = df1.drop('data_robot_logistic_regression_label', axis=1)

        # we deal with categorical columns using pandas.get_dummies
        # this has the downside of not handling categorical columns which are numbers
        # dealing with those columns with sklearn OneHotEncoder and ColumnTransformer
        # requires the user to specify those columns.
        # ! TBD - user input for numeric categorical columns in future version
        df3 = pd.get_dummies(df2)

        # deal with discrepancies against the dummy columns of the training set
        for newc in df3.columns:
            if newc not in self._columns:
                df3.drop(newc, axis=1, inplace=True)
        for c in self._columns:
            if c not in df3.columns:
                df3[c] = [0] * df3.shape[0]

        X = df3.values

        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        # an input with many categorical columns will generate many dummy columns,
        # which implies many 0 values. This will justify using a scipy sparse matrix
        # instead of a numpy array
        # ! TBD use sparse matrix for features when many categorical columns
        return X, the_y

    def _df_2_np_x(self, df):
        """ creates an X data features numpy arrays from a pandas dataframe
            X will be a valid input for a scikit-learn estimator
            input df : pandas dataframe
            return X : numpy array
        """

        # we do not know the problem domain for which this classifier will be used.
        # so we cannot assume default values for NULL cells. Need to drop these rows.
        # We also receive default values for columns with NULLs from the user

        # warn the user that rows with null values will be dropped
        df0 = df.copy()
        a = df0.isnull().values
        b = np.sum(a, axis=1)
        c = np.where(b > 0)
        if len(c[0]) > 0:
            s = [str(n) for n in c[0]]
            txl = ','.join(s)
            msg = 'Following rows were excluded from processing because they contain NULL values:\n[{}]'.format(txl)
            warnings.warn(msg)

        for c in self.remove_columns:
            if c in df0.columns:
                df0.drop(c, axis=1, inplace=True)
        for c in self.replace_nulls:
            if c in df0.columns:
                df0.fillna({c: self.replace_nulls[c]}, inplace=True)
        df1 = df0.dropna()

        # we deal with categorical columns using pandas.get_dummies
        # this has the downside of not handling categorical columns which are numbers
        # dealing with those columns with sklearn OneHotEncoder and ColumnTransformer
        # requires the user to specify those columns.
        # ! TBD - user input for numeric categorical columns in future version
        df2 = pd.get_dummies(df1)

        # deal with discrepancies against the dummy columns of the training set
        for newc in df2.columns:
            if newc not in self._columns:
                df2.drop(newc, axis=1, inplace=True)
        for c in self._columns:
            if c not in df2.columns:
                df2[c] = [0]*df2.shape[0]

        X = df2.values

        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        # an input with many categorical columns will generate many dummy columns,
        # which implies many 0 values. This will justify using a scipy sparse matrix
        # instead of a numpy array
        # ! TBD use sparse matrix for features when many categorical columns
        return X

if __name__ != '__main__':
    INPUT_FILE = '/Users/ashacked/dev/python/ML/scikit_tester/DR_Demo_Lending_Club_reduced.csv'
    df = pd.read_csv(INPUT_FILE, na_values=['NA', 'na', 'NULL', 'null', 'nil'])
    features = df.drop('is_bad', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, df['is_bad'], random_state=0)

    dr = DrLogisticRegression(remove_columns=['Id'],
                              replace_nulls={'mths_since_last_delinq': 1200, 'mths_since_last_record': 1200})
    pars = dr.tune_parameters(features, df['is_bad'])
    dr.fit(X_train, y_train.values)
    y = dr.predict(X_test)
    pr = dr.predict_proba(X_test)
    ev = dr.evaluate(X_test, y_test.values)
    print('end')