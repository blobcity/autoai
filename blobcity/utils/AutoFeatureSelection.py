# Copyright 2021 BlobCity, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This Python File Consists of Functions to perform Automatic feature Selection 

"""
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from blobcity.utils.Cleaner import data_cleaner


class AutoFeatureSelection:
    
    @staticmethod
    def drop_high_correlation_features(X, threshold=0.95):
        """
        Drops highly correlated features based on the given threshold.

        :param X: pandas DataFrame
        :param threshold: Correlation threshold for dropping features
        :return: pandas DataFrame with reduced multicollinearity
        """
        cor_matrix = X.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        return X.drop(columns=to_drop) if to_drop else X

    @staticmethod
    def drop_constant_features(X):
        """
        Drops constant and low variance features.

        :param X: pandas DataFrame
        :return: pandas DataFrame with non-constant features
        """
        selector = VarianceThreshold(threshold=0)
        X_reduced = X.loc[:, selector.fit(X).get_support()]
        return X_reduced

    @staticmethod
    def main_score(result_score, dict_class):
        """
        Aggregates feature scores by averaging categorical features.

        :param result_score: Dictionary of feature importance scores
        :param dict_class: Dictionary class object
        :return: Aggregated score dictionary
        """
        if dict_class.ObjectExist:
            for obj in dict_class.ObjectList:
                obj_scores = {key: val for key, val in result_score.items() if obj in key}
                result_score[obj] = mean(obj_scores.values()) if obj_scores else result_score.get(obj, 0)

            return {key: val for key, val in result_score.items() if not any(f"{obj}_" in key for obj in dict_class.ObjectList)}

        return result_score

    @staticmethod
    def get_feature_importance(X, Y, score_func, dict_class):
        """
        Computes feature importance using SelectKBest.

        :param X: pandas DataFrame of features
        :param Y: pandas Series or DataFrame target
        :param score_func: Scoring function (f_classif or f_regression)
        :param dict_class: Dictionary class object
        :return: Reduced pandas DataFrame with important features
        """
        if X.shape[1] < 3:
            dict_class.feature_importance = None
            return X
        
        fit = SelectKBest(score_func=score_func, k=X.shape[1]).fit(X, Y)
        scores = pd.DataFrame({'features': X.columns, 'score': fit.scores_})
        scores['score'] = MinMaxScaler().fit_transform(scores[['score']])

        main_scores = AutoFeatureSelection.main_score(dict(scores.values), dict_class)
        return AutoFeatureSelection.get_absolute_list(main_scores, X, dict(scores.values), dict_class)

    @staticmethod
    def get_absolute_list(feature_scores, X, full_scores, dict_class):
        """
        Removes features with low importance scores.

        :param feature_scores: Dictionary of processed feature scores
        :param X: pandas DataFrame
        :param full_scores: Original dictionary of feature scores
        :param dict_class: Dictionary class object
        :return: Filtered pandas DataFrame
        """
        low_importance_features = [key for key, val in feature_scores.items() if val < 0.001]
        drop_list = [key for key in full_scores.keys() if any(feature in key for feature in low_importance_features)]

        dict_class.feature_importance = {k: v for k, v in feature_scores.items() if v >= 0.001}
        return X.drop(columns=drop_list) if drop_list else X

    @staticmethod
    def feature_selection(dataframe, target, dict_class, disable_collinearity=False):
        """
        Performs automatic feature selection.

        :param dataframe: pandas DataFrame
        :param target: Target column name
        :param dict_class: Dictionary class object
        :param disable_collinearity: If True, skips correlation-based feature removal
        :return: List of selected feature names
        """
        df = data_cleaner(dataframe, dataframe.drop(columns=[target]).columns.to_list(), target, dict_class)
        score_func = f_classif if dict_class.getdict()['problem']["type"] == 'Classification' else f_regression
        
        X, Y = df.drop(columns=[target]), df[target]
        X = AutoFeatureSelection.drop_constant_features(X)
        if not disable_collinearity:
            X = AutoFeatureSelection.drop_high_correlation_features(X)

        X = AutoFeatureSelection.get_feature_importance(X, Y, score_func, dict_class)
        selected_features = AutoFeatureSelection.get_original_features(X.columns.to_list(), dict_class)

        dict_class.addKeyValue('features', {'X_values': selected_features, 'Y_values': target})
        return selected_features

    @staticmethod
    def get_original_features(feature_list, dict_class):
        """
        Extracts original feature names after processing categorical features.

        :param feature_list: List of feature names
        :param dict_class: Dictionary class object
        :return: Filtered list of feature names
        """
        if not dict_class.ObjectExist:
            return feature_list

        categorical_features = dict_class.ObjectList
        filtered_features = [feat for feat in feature_list if not any(f"{cat}_" in feat for cat in categorical_features)]
        return filtered_features + categorical_features

    # IMAGE PROCESSING FUNCTIONS
    @staticmethod
    def image_processing(data, targets, resize, dict_class):
        """
        Processes images into a structured DataFrame.

        :param data: Path to image dataset
        :param targets: List of target labels
        :param resize: Image resize dimensions
        :param dict_class: Dictionary class object
        :return: pandas DataFrame of processed images and labels
        """
        training_data, label_mapping = AutoFeatureSelection.create_training_data(data, targets, resize)
        dict_class.original_label = label_mapping
        dict_class.original_shape = [len(training_data), *training_data[0][0].shape]
        dict_class.addKeyValue('cleaning', {"resize": resize})

        return pd.DataFrame(training_data, columns=['image', 'label'])

    @staticmethod
    def create_training_data(data, targets, resize):
        """
        Reads images, resizes, and maps them to target labels.

        :param data: Path to image dataset
        :param targets: List of target labels
        :param resize: Resize dimensions
        :return: Tuple (training_data, label_mapping)
        """
        training_data, label_mapping = [], {idx: category for idx, category in enumerate(targets)}

        for category in targets:
            category_path = os.path.join(data, category)
            class_num = targets.index(category)

            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, (resize, resize))
                    training_data.append([img_resized, class_num])
                except Exception:
                    continue

        return training_data, label_mapping

    @staticmethod
    def get_reshaped_image(training_data):
        """
        Flattens image data for model training.

        :param training_data: List of processed image data
        :return: Tuple (X, y) with reshaped images and labels
        """
        X = np.array([item[0] for item in training_data]).reshape(len(training_data), -1)
        y = np.array([item[1] for item in training_data])
        return X, y
