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
        num_folds = 5
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        grid = GridSearchCV(LogisticRegression(), param_grid=param_grid,
                            scoring=scoring, refit="f1", cv=kfold,
                            return_train_score=True)
        grid.fit(the_X, the_y)
        res = grid.best_params_
        f1 = grid.best_score_
        logloss = grid.cv_results_['mean_test_neg_log_loss'][grid.best_index_]
        score = {'f1_score': round(f1,2), 'logloss': round(logloss, 2)}
        res['scores'] = score

        results = {}
        for i in range(num_folds):
            fold_prefix = 'split{}_test_'.format(i)
            f1_fold = ''.join([fold_prefix,'f1'])
            fold_results = grid.cv_results_[f1_fold]
            avg_f1 = np.mean(fold_results)
            tuple = np.where(fold_results >= avg_f1)
            avg_f1_ix = tuple[0][0] if len(tuple[0]) > 0 else 0

            max_f1 = np.max(fold_results)
            tuple = np.where(fold_results >= max_f1)
            max_f1_ix = tuple[0][0] if len(tuple[0]) > 0 else 0
            out = grid.cv_results_['params'][max_f1_ix]

            lgs_fold = ''.join([fold_prefix,'neg_log_loss'])
            lgs = grid.cv_results_[lgs_fold][avg_f1_ix]
            scores = {'f1_score': round(avg_f1, 2), 'logloss': round(lgs, 2)}
            out['scores'] = scores
            results['fold_{}'.format(i+1)] = out

        results['the_best'] = res
        return results

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

def lending_club_demo():
    import os
    data_file = 'DR_Demo_Lending_Club_reduced.csv'
    directory = os.path.dirname(__file__) # put the data_file in the same location as the script
    INPUT_FILE = '/'.join([directory, data_file])

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

def test_reproducible():
    from sklearn.datasets import load_iris
    iris = load_iris()
    cols = [ '_'.join(ft.split()[:2] ) for ft in iris.feature_names]
    df = pd.DataFrame(data=iris.data, columns=cols)
    y = (iris.target != 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0)

    dr1 = DrLogisticRegression()
    dr1.fit(X_train, y_train)
    y_pred1 = dr1.predict(X_test)
    f11 = f1_score(y_test, y_pred1)

    dr2 = DrLogisticRegression()
    dr2.fit(X_train, y_train)
    y_pred2 = dr2.predict(X_test)
    f12 = f1_score(y_test, y_pred2)

    dr3 = DrLogisticRegression()
    dr3.fit(X_train, y_train)
    y_pred3 = dr3.predict(X_test)
    f13 = f1_score(y_test, y_pred3)

    assert_array_equal(y_pred1, y_pred2)
    assert_array_equal(y_pred1, y_pred3)
    assert_almost_equal(f11, f12)
    assert_almost_equal(f11, f13)

def test_nulls():
    from numpy import nan as NA
    from sklearn.datasets import load_iris

    iris = load_iris()
    cols = ['_'.join(ft.split()[:2]) for ft in iris.feature_names]
    df = pd.DataFrame(data=iris.data, columns=cols)
    y = (iris.target != 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0)

    dr1 = DrLogisticRegression()
    dr1.fit(X_train, y_train)
    ac = dr1.clf.coef_[0]

    X_train_n = X_train.copy()
    X_train_n.iloc[0, 0] = NA
    X_train_n.iloc[1, 0] = NA
    X_train_n.iloc[1, 1] = NA
    # test we set NaN correctly
    assert (np.isnan(X_train_n.iloc[1, 1]) == True)
    assert (np.isnan(X_train_n.iloc[1, 2]) == False)

    dr2 = DrLogisticRegression()
    dr2.fit(X_train_n, y_train)
    bc = dr2.clf.coef_[0]
    # test that after removing a small amount of NULL rows
    # we still get a valid model
    assert_array_almost_equal(ac, bc, decimal=2)

    X_test_n = X_test.copy()
    X_test_n.iloc[0, 0] = NA
    X_test_n.iloc[1, 0] = NA
    X_test_n.iloc[1, 1] = NA
    y_pred_n = dr2.predict(X_test_n)
    # test that during prediction null rows are removed and prediction
    # is performed only on the valid rows
    assert y_test.shape[0] == (y_pred_n.shape[0] + 2)

    # test that during prediction a warning is issued informing that null rows
    # were removed. the message gives the indexes of the rows removed
    # and the caller can use those indexes for a correct comparation between
    # y_test and y_predict
    assert_warns(UserWarning, dr2.predict, X_test_n)

def test_output_format():
    from sklearn.datasets import load_iris
    iris = load_iris()
    cols = ['_'.join(ft.split()[:2]) for ft in iris.feature_names]
    df = pd.DataFrame(data=iris.data, columns=cols)
    y = (iris.target != 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0)

    dr = DrLogisticRegression()
    dr.fit(X_train, y_train)

    y_pred = dr.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]

    proba = dr.predict_proba(X_test)
    assert proba.shape[0] == y_test.shape[0]
    assert proba.shape[1] == 2

    eva = dr.evaluate(X_test, y_test)
    assert 'f1_score' in eva
    assert 'logloss' in eva

    tune = dr.tune_parameters(X_test, y_test)
    for k in tune.keys():
        one_fold = tune[k]
        assert 'fit_intercept' in one_fold
        assert 'solver' in one_fold
        assert 'tol' in one_fold
        assert 'scores' in one_fold
        assert 'f1_score' in one_fold['scores']
        assert 'logloss' in one_fold['scores']

if __name__ == '__main__':
    from sklearn.utils._testing import assert_almost_equal
    from sklearn.utils._testing import assert_array_almost_equal
    from sklearn.utils._testing import assert_array_equal
    from sklearn.utils._testing import assert_warns

    #lending_club_demo()
    test_reproducible()
    test_nulls()
    test_output_format()

'''
comments:

a. unitest - model can handle new category levels at prediction time
    This was one of the requirements for a unittest in the problem paper.
    As far as I know, in a supervised model like classification the 
    category labels have to be known at training time. That is when
    the model learns them. At prediction time the model can only map 
    a features vector to one of the labels learned at training.
    In order to handle a ML problem where the categories set is hard
    to close at the training time, we need to use an unsupervised 
    algorithm like clustering for example.
    
b. short answer 1
   compare logistic regression with f1=0.6 vs. neural network with f1=0.63
   for the credit risk use case - problem set in  DR_Demo_Lending_Club_reduced.csv
   
   neural networks are notoriously performance demanding and that is one of the
   reasons that they became practicle only after the GPUs (very strong CPUs) were
   introduced and deep networks with many neuron layers could be created. so, from 
   a time/resources standpoint the Logistic Regression is certainly preferable.
   But there is also the model perfomance side.
   This kind of problem requires a recall close to 1.
   recall = TP / (TP + FN)
   That means we cannot afford that a positive instance, one that is a credit risk will
   not be recognized. The financial crisis in 2008 caused by many mortgage defaults shows
   us that. We must strive for a recall as close to 1 as possible. this means FN close to 
   zero.
   The recall is hidden inside the f1 formula
   f1 = 2 * precision * recall /(precision+recall)
   so, the fact that the neural network has a higher f1 could be either because it has a 
   higher precision or because it has a higher recall. we do not know exactly from the
   problem. but we cannot take the risk. recall is much to important here.
   So I would go for the neural network with the higher f1=0.63, and use a (more expensive)
   GPU to run the model. I think that for a potential client like  bank it is definetly
   worthwhile.
'''