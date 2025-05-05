import json
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import (
    PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler,
    QuantileTransformer, StandardScaler, MinMaxScaler,
    FunctionTransformer, Binarizer
)
import warnings

from datasets.base.abstract_dataset import AbstractDatasetClass
from datasets.robustvision.preprocessor import clean_data, detect_and_remove_outliers_in_features_iqr, \
    remove_outliers_in_labels, binData, createFeatures, global_normalization, subject_wise_normalization, \
    separate_features_and_targets
from src.utils.data_utils import create_lstm_tensors_dataset, create_dataloaders_dataset


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobustVisionDataset(AbstractDatasetClass):
    def __init__(self, data_dir, sequence_length, input_features, target_feature, meta_features, is_train=True):
        super().__init__(data_dir, sequence_length)
        self.features_tensor = None
        self.targets_tensor = None
        self.feature_names = []
        self.subject_list = []
        self.scalers = {}

        self.sequence_length = sequence_length
        self.input_features = input_features
        self.target_feature = target_feature
        self.meta_features = meta_features
        self.data_dir = data_dir
        self.is_train = is_train

        self.name = "robustvision"
        self.best_transformers = None
        self.minDepth = 0.35  # in meter
        self.maxDepth = 3
        self.subject_scaler = RobustScaler()  # or any other scaler
        self.feature_scaler = None
        self.target_scaler = None
        self.target_column_name = 'Gt_Depth'
        self.subject_id_column = 'SubjectID'
        self.multiplicator = 100  # to convert from cm to m
        self.transformers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler': MaxAbsScaler,
            'RobustScaler': RobustScaler,
            'QuantileTransformer-Normal': lambda: QuantileTransformer(output_distribution='normal'),
            'QuantileTransformer-Uniform': lambda: QuantileTransformer(output_distribution='uniform'),
            'PowerTransformer-YeoJohnson': lambda: PowerTransformer(method='yeo-johnson'),
            'PowerTransformer-BoxCox': lambda: PowerTransformer(method='box-cox'),
            'Normalizer': Normalizer,
            'Binarizer': lambda threshold=0.0: Binarizer(threshold=threshold),
            'FunctionTransformer-logp1p': lambda func=np.log1p: FunctionTransformer(func),
            'FunctionTransformer-rec': lambda func=np.reciprocal: FunctionTransformer(func),
            'FunctionTransformer-sqrt': lambda func=np.sqrt: FunctionTransformer(func),
        }

        self.scaler_config = {
            'use_minmax': True,
            'use_standard_scaler': False,
            'use_robust_scaler': False,
            'use_quantile_transformer': False,
            'use_power_transformer': False,
            'use_max_abs_scaler': False
        }

        self.scaler_config_features = {
            'use_minmax': False,  # avg 56
            'use_standard_scaler': False,  # avg 15
            'use_robust_scaler': False,  # avg 15.5
            'use_quantile_transformer': False,  # avg 14.4
            'use_power_transformer': False,  # avg 15.2
            'use_max_abs_scaler': False  # avg 10.79
            # none                               # avg
        }

        print(f"[Dataset Init] Using input features: {self.feature_names}")


        self.input_data = self.load_data()

        # Preprocessing pipeline
        data = self.create_features(self.input_data)
        data = self.normalize_data(data)

        if is_train:
            data = self.calculate_transformations_for_features(data)
        else:
            data = self.apply_transformations_on_features(data)

        data = self.scale_target(data, isTrain=is_train)

        # Sequenzen erzeugen
        self.features, self.targets, subject_ids = self.create_sequences(data)
        # self.features, self.targets = self.separate_features_and_targets(sequences)

        # Tensoren bauen
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.float32)

        self.subject_list = list(map(str, subject_ids))  # [N]
        self.feature_names = self.input_features

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, idx):
        x = self.features_tensor[idx]  # <-- torch.Tensor
        y = self.targets_tensor[idx]  # <-- torch.Tensor oder int/float
        return x, y

    @property
    def feature_names(self):
        return self.input_features

    def separate_features_and_targets(self, sequences):
        features = []
        targets = []
        for seq in sequences:
            features.append(seq)
            targets.append(seq[-1][self.target_feature])  #
        return np.array(features), np.array(targets)
    # 1.
    def load_data(self):
        """
        Read and aggregate data from multiple subjects.

        :param data_dir: Directory containing the subject folders.
        :return: Combined dataframe of all subjects.
        """
        print("Reading and aggregating data...")
        all_data = []
        for subj_folder in os.listdir(self.data_dir):
            subj_path = os.path.join(self.data_dir, subj_folder)
            if os.path.exists(subj_path):
                depthCalib_path = os.path.join(subj_path, "depthCalibration.csv")
                if os.path.exists(depthCalib_path):
                    df = pd.read_csv(depthCalib_path, delimiter="\t")
                    df.rename(columns=lambda x: x.strip().replace(' ', '_').title(), inplace=True)
                    # df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

                    starting_columns = [
                        'Gt_Depth', 'World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y',
                        'World_Gaze_Direction_L_Z', 'World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y',
                        'World_Gaze_Direction_R_Z', 'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z',
                        'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Z'
                    ]

                    # self.clean_column_names(df_depthEval)
                    df = df[starting_columns]
                    for col in starting_columns:
                        df[col] = df[col].astype(float)

                    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

                    df = df[starting_columns].astype(float)
                    df2 = clean_data(df, target_column_name=self.target_column_name, multiplication=self.multiplicator)
                    df3 = remove_outliers_in_labels(df2, window_size=5, threshold=10,
                                                    target_column_name=self.target_column_name)
                    df4 = detect_and_remove_outliers_in_features_iqr(df3)
                    df5 = binData(df4, False)
                    df5['SubjectID'] = subj_folder
                    all_data.append(df5)

        self.input_data = pd.concat(all_data, ignore_index=True)
        print("Finished loading input data.")
        self.subject_list = self.input_data['SubjectID'].unique()

        return self.input_data

    # 2.
    def create_features(self, data_in):
        """
        Generate features from the input data.

        :param data_in: Input dataframe.
        :return: Dataframe with additional features.
        @param data_in:
        @return:
        """
        data_in = createFeatures(data_in, input_features=self.input_features)

        return data_in

    # 3.
    def normalize_data(self, data):

        # Apply global normalization first
        data_set_in = global_normalization(data)

        # Then proceed with the existing subject-wise normalization
        unique_subjects = data_set_in['SubjectID'].unique()

        # Choose your scaler for subject-wise normalization
        subject_scaler = RobustScaler()

        # Apply subject-wise normalization
        dataset_in_normalized = subject_wise_normalization(data_set_in, unique_subjects, subject_scaler)

        return dataset_in_normalized

    # 4. A (Traininig data)
    def calculate_transformations_for_features(self, data_in):
        best_transformers = {}
        transformed_data = data_in.copy()
        ideal_skew = 0.0
        ideal_kurt = 3.0

        for column in data_in.columns:
            if column == "Gt_Depth" or column == "SubjectID":
                continue

            # print(f"Processing column: {column}")
            original_skew = skew(data_in[column])
            original_kurt = kurtosis(data_in[column], fisher=False)  # Pearson's definition

            best_transform = None
            best_transform_name = ""
            min_skew_diff = float('inf')

            for name, transformer_class in self.transformers.items():
                transformer = transformer_class()  # Create a new object for each transformer
                try:
                    data_transformed = transformer.fit_transform(data_in[[column]])
                    current_skew = skew(data_transformed)[0]
                    current_kurt = kurtosis(data_transformed, fisher=False)[0]

                    # Calculate the distance from the ideal distribution characteristics
                    dist = np.sqrt((current_skew - ideal_skew) ** 2 + (current_kurt - ideal_kurt) ** 2)

                    # If this transformer is the best so far, store it
                    if dist < min_skew_diff:
                        min_skew_diff = dist
                        best_transform = transformer
                        best_transform_name = name

                except ValueError as e:  # Handle failed transformations, e.g., Box-Cox with negative values
                    # print(f"Transformation failed for {name} on column {column}: {e}")
                    continue

            best_transformers[column] = (best_transform_name, best_transform)

            # Transform the column in the dataset
            if best_transform:
                transformed_column = best_transform.transform(data_in[[column]])
                transformed_data[column] = transformed_column.squeeze()

        self.best_transformers = best_transformers
        return transformed_data

    # 4. B ( Validation/ test data)
    def apply_transformations_on_features(self, data_in):
        transformed_validation_data = data_in.copy()

        for column, (name, transformer) in self.best_transformers.items():
            if transformer is not None:
                if column == "Gt_Depth":
                    transformed_validation_data[column] = data_in[[column]]
                elif column == "SubjectID":
                    continue
                else:
                    # Apply the transformation using the fitted transformer object
                    transformed_column = transformer.transform(data_in[[column]])
                    transformed_validation_data[column] = transformed_column.squeeze()

        return transformed_validation_data

    # 5.
    def scale_target(self, data_in, isTrain=False):

        if isTrain:
            self.target_scaler = self.select_scaler(self.scaler_config)
            # Extract GT_depth before scaling and reshape for scaler compatibility
            gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.fit_transform(gt_depth)
                # Re-attach the excluded columns
            data_in['Gt_Depth'] = gt_depth.ravel()
        else:
            gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.transform(gt_depth)
                # Re-attach the excluded columns
            data_in['Gt_Depth'] = gt_depth.ravel()

        return data_in

    # 6.
    def scale_features(self, data_in, isTrain=True):
        """
        Scale the features in the training and validation datasets using the provided scaler.

        :param data_in: Dataframe with training data.
        :return: Scaled dataframe.
        @param isTrain:
        """

        # target_column = data_in[self.target_column_name].values.reshape(-1, 1)
        # subject_id_column = data_in[self.subject_id_column]
        # data = data_in.drop(columns=[self.target_column_name, self.subject_id_column])
        #
        # # Fit the scaler on the training data
        # if isTrain:
        #     self.feature_scaler = self.select_scaler(self.scaler_config_features)
        #     # Fit the scaler only if it's not already fitted (e.g., during validation)
        #     data_scaled = self.feature_scaler.fit_transform(data)
        # else:
        #     data_scaled = self.feature_scaler.transform(data)
        #
        # data_out = pd.DataFrame(data_scaled, columns=data.columns)
        # data_out[self.target_column_name] = target_column.ravel()
        # data_out[self.subject_id_column] = subject_id_column.reset_index(drop=True)

        return data_in

    # 6.
    # def create_sequences(self, data, window_size=10, stride=1):
    #     sequences = []
    #     subject_ids = []
    #
    #     for subject_id, df_subject in data.groupby(self.subject_id_column):
    #         df_subject = df_subject.reset_index(drop=True)
    #         for i in range(0, len(df_subject) - window_size + 1, stride):
    #             window = df_subject.iloc[i:i + window_size]
    #             sequences.append(window[self.input_features].values)
    #             subject_ids.append(subject_id)
    #
    #     return sequences, subject_ids

    def create_sequences(self, df):
        """
        Create sequences of data for time-series analysis.

        :param df: Dataframe with the data to sequence.
        :return: List of sequences, where each sequence is a tuple of (features, target, subject_id).
        """
        sequences = []
        subject_ids = []
        targets = []

        grouped_data = df.groupby('SubjectID')
        for subj_id, group in grouped_data:
            for i in range(len(group) - self.sequence_length):
                seq_features = group.iloc[i:i + self.sequence_length].drop(columns=['Gt_Depth', 'SubjectID']).to_numpy()
                seq_target = group.iloc[i + self.sequence_length]['Gt_Depth']
                sequences.append(seq_features)
                subject_ids.append(subj_id)
                targets.append(seq_target)

        # Rückgabe: alles als np.array
        return np.array(sequences), np.array(targets), np.array(subject_ids)

    # Utilities
    def select_scaler(self, config):
        """Select the scaler based on the configuration provided."""
        if config['use_minmax']:
            return MinMaxScaler(feature_range=(0, 1000))
        if config['use_standard_scaler']:
            return StandardScaler()
        if config['use_robust_scaler']:
            return RobustScaler(with_scaling=True, with_centering=True, unit_variance=True)
        if config['use_quantile_transformer']:
            return QuantileTransformer(output_distribution='normal')
        if config['use_power_transformer']:
            return PowerTransformer(method='yeo-johnson')
        if config['use_max_abs_scaler']:
            return MaxAbsScaler()
        return None

    def get_data(self):
        return self.input_data

    def get_data_loader(self, train_index, val_index=None, test_index=None, batch_size=100):
        """
        Create and return data loaders for training, validation, and testing datasets.

        :param train_index: Indices for training subjects.
        :param val_index: Indices for validation subjects (optional).
        :param test_index: Indices for test subjects (optional).
        :param batch_size: Batch size for the data loaders.
        :return: Data loaders for training, validation, and testing datasets, and the input size.
        """
        train_loader = self.prepare_loader(train_index, batch_size, is_train=True)
        val_loader = self.prepare_loader(val_index, batch_size, is_train=False) if val_index is not None else None
        # test_loader = self.prepare_loader(test_index, batch_size, is_train=False) if test_index is not None else None

        input_size = train_loader.dataset[0][0].shape[1]  # Assuming the first dimension is batch_size

        return train_loader, val_loader, input_size


    def prepare_loader(self, subject_index, batch_size, is_train=False):
        subjects = subject_index if isinstance(subject_index, list) else [subject_index]
        # print(f"Preparing data for subjects: {subjects}")
        data = self.input_data[self.input_data['SubjectID'].isin(subjects)]

        # Check if the data is empty before proceeding
        if data.empty:
            raise ValueError(f"No data found for subjects: {subjects}")

        # Feature creation and normalization
        data = self.create_features(data)
        data = self.normalize_data(data)

        # Apply transformations if necessary
        if is_train:
            data = self.calculate_transformations_for_features(data)
        else:
            data = self.apply_transformations_on_features(data)

        # Scale features and target (transform only using the fitted scaler)
        data = self.scale_target(data, isTrain=is_train)
        # Generate sequences
        sequences = self.create_sequences(data)
        features, targets = separate_features_and_targets(sequences)

        # Convert to tensors and create data loader
        features_tensor, targets_tensor = create_lstm_tensors_dataset(features, targets, is_train)
        data_loader = create_dataloaders_dataset(features_tensor, targets_tensor, batch_size=batch_size)

        return data_loader

    import torch
    import json
    import pickle

    def save_processed_data(self, save_path):
        save_path.mkdir(parents=True, exist_ok=True)

        # Features und Targets als Tensors speichern
        torch.save(self.features_tensor, save_path / "features.pt")
        torch.save(self.targets_tensor, save_path / "targets.pt")  # Achtung: nicht self.targets!

        # Feature-Namen und Subject-IDs speichern
        with open(save_path / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        # Subject-IDs als Liste von Strings/Ints speichern – JSON-kompatibel
        subject_list_serializable = list(map(str, self.subject_list))  # falls z. B. ndarray
        with open(save_path / "subject_ids.json", "w") as f:
            json.dump(subject_list_serializable, f)

        # Scaler und Transformer-Objekte (für Invers-Transformation)
        with open(save_path / "transformers.pkl", "wb") as f:
            pickle.dump(self.scalers, f)

        print(f"[INFO] Daten erfolgreich gespeichert unter: {save_path}")

    # def save_processed_data(self, save_path):
    #     save_path = Path(save_path)
    #     save_path.mkdir(parents=True, exist_ok=True)
    #
    #     torch.save(self.features_tensor, save_path / "features.pt")
    #     torch.save(self.targets_tensor, save_path / "targets.pt")
    #
    #
    #
    #     with open(save_path / "subject_ids.json", "w") as f:
    #         if isinstance(self.subject_list, np.ndarray):
    #             json.dump(self.subject_list.tolist(), f)
    #         else:
    #             json.dump(self.subject_list, f)
    #
    #     with open(save_path / "feature_names.json", "w") as f:
    #         json.dump(self.feature_names, f)
    #
    #     # Nur konkret verwendete Transformer speichern (nicht das große Dictionary mit Lambdas!)
    #     to_save = {
    #         "feature_scaler": self.feature_scaler,
    #         "target_scaler": self.target_scaler,
    #         "subject_scaler": self.subject_scaler
    #     }
    #     with open(save_path / "transformers.pkl", "wb") as f:
    #         pickle.dump(to_save, f)

    # def save_processed_data(self, save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    #     sequences = self.create_sequences(self.input_data)
    #     features, targets = separate_features_and_targets(sequences)
    #
    #     print(f"type(features): {type(features)}")
    #     print(f"type(features[0]): {type(features[0])}")
    #
    #     features_np = np.array(features, dtype=np.float32)
    #     targets_np = np.array(targets, dtype=np.float32)
    #
    #     torch.save(torch.from_numpy(features_np), os.path.join(save_dir, "features.pt"))
    #     torch.save(torch.from_numpy(targets_np), os.path.join(save_dir, "targets.pt"))
    #
    #     with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
    #         json.dump(self.input_features, f)
    #
    #     with open(os.path.join(save_dir, "transformers.pkl"), "wb") as f:
    #         import pickle
    #         pickle.dump(self.best_transformers, f)


    # def load_processed_data(self, load_dir):
    #     self.features_tensor = torch.load(os.path.join(load_dir, "features.pt"))
    #     self.targets_tensor = torch.load(os.path.join(load_dir, "targets.pt"))
    #
    #     with open(os.path.join(load_dir, "feature_names.json"), "r") as f:
    #         self.input_features = json.load(f)
    #
    #     with open(os.path.join(load_dir, "transformers.pkl"), "rb") as f:
    #         import pickle
    #         self.best_transformers = pickle.load(f)

    def load_processed_data(self, load_dir):
        import torch
        import json
        import pickle

        load_dir = Path(load_dir)
        self.features_tensor = torch.load(load_dir / "features.pt")
        self.targets_tensor = torch.load(load_dir / "targets.pt")

        with open(load_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        with open(load_dir / "subject_ids.json", "r") as f:
            self.subject_list = json.load(f)

        with open(load_dir / "transformers.pkl", "rb") as f:
            scalers = pickle.load(f)
            self.feature_scaler = scalers.get("feature_scaler", None)
            self.target_scaler = scalers.get("target_scaler", None)
            self.subject_scaler = scalers.get("subject_scaler", None)

    def set_selected_features(self, selected_feature_names, mode="keep"):
        """
        Parameters:
        - selected_feature_names (list[str]): Features, die behalten oder entfernt werden sollen
        - mode (str): "keep" oder "remove"
        """
        if mode not in ["keep", "remove"]:
            raise ValueError("mode must be either 'keep' or 'remove'")

        if mode == "keep":
            selected_indices = [self.feature_names.index(f) for f in selected_feature_names if f in self.feature_names]
        else:  # remove
            selected_indices = [i for i, f in enumerate(self.feature_names) if f not in selected_feature_names]

        self.features_tensor = self.features_tensor[:, :, selected_indices]
        self.feature_names = [self.feature_names[i] for i in selected_indices]

    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = value