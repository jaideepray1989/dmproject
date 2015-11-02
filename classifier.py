#!/usr/bin/env python
import numpy
from sklearn import cross_validation, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd #this is how I usually import pandas

# please set the DIR_PATH to where the files are.
from sklearn.utils import column_or_1d

DIR_PATH = ""
TRAIN_FILE      = DIR_PATH + "training.txt"
TEST_FILE       = DIR_PATH + "testing.txt"                  # set this to the new test file name
LABEL_TRAINING_FILE = DIR_PATH + "label_training.txt"
PREDICTION_FILE = DIR_PATH + "preds.txt"                 # predictions will be written here

__train__ = TRAIN_FILE
__label_training__ = LABEL_TRAINING_FILE
__test__ = TEST_FILE
__submission__ = PREDICTION_FILE;

def read_data(path):
    data = pd.read_table(path, sep = ' ', header = None, names = ['doc_id','feature_index','tf-idf'])
    return data

def pivot_table(data_train):
    doc_feature_df_train = data_train.pivot('doc_id', columns = 'feature_index', values = 'tf-idf')
    doc_feature_df_train = doc_feature_df_train.fillna(0.0)
    return doc_feature_df_train

def label_extract(path):
    label_train = pd.read_table(path, sep = ' ', header = None, names = ['class_label'])
    return label_train


if __name__ == '__main__':
    print 'Preprocessing...'
    x_train = read_data(__train__)
    x_test = read_data(__test__)
    num_train = 1842
    doc_class_df_train = label_extract(__label_training__)
    print 'Dividing into training set and cv set...'
    num_label = len(doc_class_df_train)
    x_test['doc_id'] = x_test['doc_id'].map(lambda id : id + num_train)
    merged = x_train.copy().append(x_test.copy(), ignore_index=True)
    doc_feature_df_merged = merged.pivot('doc_id', columns = 'feature_index', values = 'tf-idf')
    doc_feature_df_merged = doc_feature_df_merged.fillna(0.0)
    doc_feature_df_train = doc_feature_df_merged[0:num_train]
    doc_feature_df_test = doc_feature_df_merged[num_train:]
    x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(
        doc_feature_df_train, doc_class_df_train, test_size=0.3, random_state=None)
    print 'Training set size: %d, cv set size: %d' % (
        y_train.shape[0], y_cv.shape[0])

    y_train = column_or_1d(y_train)
    print 'Naive Bayes model'
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    numpy.savetxt(__submission__, clf.predict(doc_feature_df_test), delimiter='\n', fmt='%d')

    print 'Bernoulli Bayes model'
    bclf = BernoulliNB()
    bclf.fit(x_train,y_train)

    print 'Linear Model model'
    lclf = LogisticRegression()
    lclf.fit(x_train,y_train)

    print 'Random Forest classifier'
    rclf = RandomForestClassifier(n_estimators=1000)
    rclf.fit(x_train,y_train)

    print 'Fitting SVM classifier'
    svm_clf = svm.SVC()
    svm_clf.fit(x_train,y_train)


    # Gaussian Naive Bayes algorithm accuracy on cross validation set
    print 'Accuracy in training set for Gaussian NB: %f' % clf.score(x_train, y_train)
    print 'Accuracy in cv set for Gaussian NB: %f' % clf.score(x_cv, y_cv)

    # Bernouilli's Naive Bayes algorithm accuracy on cross validation set
    print 'Accuracy in training set for Bernoulli NB: %f' % bclf.score(x_train, y_train)
    print 'Accuracy in cv set for Bernoulli NB: %f' % bclf.score(x_cv, y_cv)

    # Logistic regression algorithm accuracy on cross validation set
    print 'Accuracy in training set for Logistic Regression: %f' % lclf.score(x_train, y_train)
    print 'Accuracy in cv set for Logistic Regression : %f' % lclf.score(x_cv, y_cv)

    # Random Forest Classifier algorithm accuracy on cross validation set
    print 'Accuracy in training set for Random Forest : %f' % rclf.score(x_train, y_train)
    print 'Accuracy in cv set for Random Forest : %f' % rclf.score(x_cv, y_cv)

    # SVM algorithm accuracy on cross validation set
    print 'Accuracy in training set for svm : %f' % svm_clf.score(x_train, y_train)
    print 'Accuracy in cv set for svm : %f' % svm_clf.score(x_cv, y_cv)

