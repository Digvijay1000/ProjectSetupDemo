import sys
from dataclasses import dataclass 


import numpy as np
import pandas as pd 

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is Responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", 
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"                   
            ]
            num_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scalar",StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ('scalar',StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical Columns :{numerical_columns}")
            logging.info(f"Categorical Columns :{categorical_columns}")

            '''we have created two seperate pipeleines one for "numerical columns" and one for "categorical columns",
              but at the end we need one sinlge pipeline so we will combine this '''
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeleine", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading The Train and Test File')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Divide Train Dataset to independent and depenedent feature

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Divide Test Dataset to independent and depedent feature

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # now we will make train array and test array 

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved Prprocessing Object")

            # we are making pkl file so for that we will make common functionality in util.py file

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )   







        except Exception as e:
            raise CustomException(e, sys)