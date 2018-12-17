import os
import sys
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import classify
from PIL import Image


class TrainCluster(object):
    def __init__(self):
        self.metrics = classify.read_training_data()
        self.nclasses = set([item['class'] for item in self.metrics])
        self.kmeans = KMeans(n_clusters=len(self.nclasses), random_state=0)
        self.confusion_matrix = None
        self.label_keys = None
        self.coverage_estimation = None
        self.prob_labels = dict()
        self.last_pred_percent = None
        self.last_pred_count = None
        self.flat_confusion = None

    def fit_trainer(self, features=None):
        df = pd.DataFrame.from_dict(self.metrics)
        if features is None:
            vals = df[['mean', 'median', 'std']].values
        else:
            vals = df[features].values
        self.kmeans.fit(vals)
        df['pred_labels'] = self.kmeans.predict(vals[:])
        self.flat_confusion = round(df[['class', 'pred_labels']].pivot_table(
            index='class', columns='pred_labels', aggfunc=len), 2)
        self.confusion_matrix = round(df[['class', 'pred_labels']].pivot_table(
            index='class', columns='pred_labels', aggfunc=len)/len(df), 2)
        self.label_keys = dict()
        for key in self.confusion_matrix.index.tolist():
            self.label_keys[key] = self.confusion_matrix.loc[key].dropna().sort_values(ascending=False).index.tolist()
        for label in df['pred_labels'].drop_duplicates().tolist():
            self.prob_labels[label] = (self.confusion_matrix[label]/self.confusion_matrix[label].sum()).dropna().to_dict()

    def convert_label_to_center(self, label):
        return self.kmeans.cluster_centers_[label]

    def transform_image(self, filepath, name, measure='mean'):
        self.fit_trainer([measure])
        im = classify.image_to_array(filepath)
        coverage_dict = dict()
        pct_dict = dict()
        orig_shape = im.shape
        labs = self.kmeans.predict(im.flatten().reshape(-1, 1))
        total_count = 0
        for lab in labs:
            if lab not in coverage_dict:
                coverage_dict[lab] = 1
            else:
                coverage_dict[lab] += 1
            total_count += 1
        for key in coverage_dict:
            pct_dict[key] = round(coverage_dict[key]/total_count * 1.0, 5)
        self.last_pred_count = coverage_dict
        self.last_pred_percent = pct_dict
        centers = np.vectorize(self.convert_label_to_center)
        centers = centers(labs)
        nimage = centers.reshape(orig_shape[:])
        nim = Image.fromarray(nimage)
        nim = nim.convert('L')
        nim.save(os.path.join(os.path.join(os.path.dirname(__file__), 'transformed'), name))
        estimated_coverage = dict()
        for label in self.label_keys.keys():
            estimated_coverage[label] = 0
        for number_class in self.last_pred_percent.keys():
            for ordinal_class in self.prob_labels[number_class].keys():
                estimated_coverage[ordinal_class] += (self.prob_labels[number_class][ordinal_class] *
                                                      self.last_pred_percent[number_class])
        return estimated_coverage


def classify_and_transform(input_file, output_file, measure='median'):
    '''
    this is an example of how to run the classification against a random input
    :param input_file: fullpath to the file that you would like to transform
    :param output_file: output full path to the file name that you would like to have classified
    :param measure: default to mean can take 'median'
    :return: None
    '''
    try:
        training = TrainCluster()
        estimates = training.transform_image(input_file, output_file, measure)
        print(estimates)
    except Exception as e:
        print(str(e))
        pass


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    classify_and_transform(input_file, output_file, measure='mean')
