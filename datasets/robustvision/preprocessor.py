import math
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.utils import resample
from joblib import Parallel, delayed, Memory

minDepth = 0.35
maxDepth = 3
memory = Memory("cache_dir", verbose=0)


def detect_and_remove_outliers_in_features_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1 * IQR))

    # Filter out the rows with outliers
    new_df = df[~mask.any(axis=1)]
    return new_df


def remove_outliers_in_labels(df, window_size, threshold, target_column_name):
    # Check if 'Gt_Depth' column exists
    if 'Gt_Depth' not in df.columns:
        raise ValueError("Column 'Gt_Depth' not found in the DataFrame")

    # Iterate over the DataFrame
    outlier_indices = []
    for i in range(len(df)):
        # Define the window range
        start = max(i - window_size // 2, 0)
        end = min(i + window_size // 2 + 1, len(df))
        window = df['Gt_Depth'].iloc[start:end]

        # Calculate the median of the window
        mean = np.mean(window)
        # median = np.nanmedian(window)

        # # Check if the current value is an outlier
        if abs(df['Gt_Depth'].iloc[i] - mean) > threshold:
            outlier_indices.append(i)

        # Check if the current value is an outlier
        # if abs(df['Gt_Depth'].iloc[i] - median) > threshold:
        #     outlier_indices.append(i)

    # Check if the outlier indices are in the DataFrame index
    outlier_indices = [idx for idx in outlier_indices if idx in df.index]
    # Now drop the outliers safely
    df_cleaned = df.drop(outlier_indices)
    print(f"Removed {len(outlier_indices)} outlier from data set.")

    return df_cleaned

def clean_data(df, target_column_name='Gt_Depth', multiplication=1):
    """
    Perform basic cleaning of the dataframe, including outlier removal and binning.

    :param df: The raw dataframe.
    :return: Cleaned dataframe.
    """


    # Remove rows where all elements are NaN
    df = df.dropna(how='all')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().copy()

    # df2 = df[df['Gt_Depth'] > 0.1]
    df2 = df[df['Gt_Depth'] > 0.35]
    print("Removed rows with Gt_Depth <= 0.35")
    df2 = df2[df2['Gt_Depth'] <= 3]

    df2["Gt_Depth"] = df2["Gt_Depth"].multiply(100)

    df2 = df2.reset_index(drop=True)

    return df2


def split_data_by_subjects(scaled_data, train_size=0.9):
    # Get unique subject IDs
    unique_subjects = scaled_data['SubjectID'].unique()

    # Split subject IDs into training and validation sets
    train_subjects, validation_subjects = train_test_split(unique_subjects, train_size=train_size, shuffle=True)

    # Filter data by subject IDs for training and validation sets
    train_data = scaled_data[scaled_data['SubjectID'].isin(train_subjects)]
    validation_data = scaled_data[scaled_data['SubjectID'].isin(validation_subjects)]

    # now drop the columns as we do not need them anymore
    # Drop the 'SubjectID' column from both datasets
    # train_data = train_data.drop(columns=['SubjectID'])
    # validation_data = validation_data.drop(columns=['SubjectID'])

    return train_data, validation_data


def getIPD(row):
    # Ensure all components (X, Y, Z) for both right and left gaze origins are present
    # if 'World_Gaze_Origin_R_X' in row and 'World_Gaze_Origin_R_Y' in row and 'World_Gaze_Origin_R_Z' in row:
    posR = [row['World_Gaze_Origin_R_X'], 0.0, row['World_Gaze_Origin_R_Z']]
    # else:
    #     logger.info("Warning: Missing right eye origin data.")
    #     return np.nan
    #
    # if 'World_Gaze_Origin_L_X' in row and 'World_Gaze_Origin_L_Y' in row and 'World_Gaze_Origin_L_Z' in row:
    posL = [row['World_Gaze_Origin_L_X'], 0.0, row['World_Gaze_Origin_L_Z']]
    # else:
    #     logger.info("Warning: Missing left eye origin data.")
    #     return np.nan

    # Check if the positions are exactly the same (which shouldn't happen)
    if posR == posL:
        logger.info("Warning: Left and right eye positions are identical.")
        return 0.0

    # Calculate the Euclidean distance between the two eye origins (IPD)
    deltaX = posR[0] - posL[0]
    deltaY = posR[1] - posL[1]
    deltaZ = posR[2] - posL[2]

    ipd = math.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    # logger.info the calculated IPD for debugging
    # logger.info("IPD ", ipd)

    return ipd


def getAngle(row):
    # Extract gaze direction vectors for right and left eyes
    vecR = [row['World_Gaze_Direction_R_X'], row['World_Gaze_Direction_R_Y'], row['World_Gaze_Direction_R_Z']]
    vecL = [row['World_Gaze_Direction_L_X'], row['World_Gaze_Direction_L_Y'], row['World_Gaze_Direction_L_Z']]

    # Compute vector norms (lengths)
    vecR_n = np.linalg.norm(vecR)
    vecL_n = np.linalg.norm(vecL)

    # Check if any vector has zero or near-zero length (i.e., invalid data)
    if vecR_n < 1e-8 or vecL_n < 1e-8:
        logger.info(f"Warning: Vector norm too small (R: {vecR_n}, L: {vecL_n}).")
        return np.nan

    # Compute the dot product
    dot_product = np.dot(vecR, vecL)

    # Normalize dot product to avoid floating point precision issues
    cos_angle = dot_product / (vecR_n * vecL_n)

    # Ensure the cosine value is within the valid range [-1, 1]
    # cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angle in radians and convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    # logger.info the angle for debugging
    # logger.info(f"Angle: {angle_degrees} degrees")

    return angle_degrees

from sklearn.preprocessing import RobustScaler
import pandas as pd
#
# def global_normalization(data: pd.DataFrame) -> pd.DataFrame:
#     # 1) which columns *not* to scale?
#     not_features = ["SubjectID", "GT_Depth"]
#
#     print("Data columns are ", data.columns)
#     # # 2) our true feature columns
#     feature_cols = [c for c in data.columns if c not in not_features]
#
#     # 3) scale *only* those
#     scaler = RobustScaler()
#     scaled = scaler.fit_transform(data[feature_cols])
#
#     # 4) rebuild a DataFrame with exactly the same index
#     df_scaled = pd.DataFrame(scaled, columns=feature_cols, index=data.index)
#
#     # 5) re-attach the meta and target columns untouched
#     for col in not_features:
#         df_scaled[col] = data[col].values
#
#     return df_scaled

def global_normalization(data):
    features = data.drop(columns=['SubjectID', 'Gt_Depth'])
    scaler = RobustScaler()  # Global scaler

    normalized_features = scaler.fit_transform(features)
    data_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    data_normalized['SubjectID'] = data['SubjectID'].values
    data_normalized['Gt_Depth'] = data['Gt_Depth'].values
    return data_normalized





def calculate_ipd(row):
    """
    Calculate the IPD based on the ground truth depth and vergence angle.

    Args:
        row: A pandas Series row containing the vergence angle and the known focused depth.

    Returns:
        The calculated IPD for the subject.
    """
    vergenceAngle = row['Vergence_Angle']
    focusedDepth = row['Gt_Depth']  # Ground truth depth (in cm)

    # Ensure the vergence angle and depth are numeric
    vergenceAngle = pd.to_numeric(vergenceAngle, errors='coerce')
    focusedDepth = pd.to_numeric(focusedDepth, errors='coerce')

    if pd.isna(vergenceAngle) or pd.isna(focusedDepth):
        return np.nan  # Invalid or missing values

    try:
        # Calculate IPD using the ground truth depth and vergence angle
        ipd = 2 * focusedDepth * math.tan(math.radians(vergenceAngle) / 2)
    except ValueError as e:
        print(f"Error in calculating IPD: {e}")
        ipd = np.nan  # Return NaN in case of calculation errors

    return ipd


def getEyeVergenceAngle(row):
    vergenceAngle = getAngle(row)
    if vergenceAngle != 0:
        ipd = getIPD(row)
        # Added a check to prevent division by zero if vergenceAngle is extremely small
        depth = ipd / (2 * math.tan(math.radians(vergenceAngle) / 2)) if math.radians(vergenceAngle) != 0 else 0
    else:
        depth = 0.0
    # Convert depth to millimeters by multiplying by 1000 (from meters to mm)
    depth_fin = abs(depth) * 100  # Convert from meters to millimeters
    # logger.info("Depth fin ", depth_fin)
    return vergenceAngle, depth_fin


def subject_wise_normalization(data, unique_subjects, scaler):
    normalized_data_list = []
    for subject in unique_subjects:
        subject_data = data[data['SubjectID'] == subject]
        subject_data_normalized = normalize_subject_data(subject_data, scaler)
        normalized_data_list.append(subject_data_normalized)
    return pd.concat(normalized_data_list, ignore_index=True)


def normalize_subject_data(subject_data, scaler):

    features = subject_data.drop(columns=['SubjectID', 'Gt_Depth'])
    # logger.info(f"Subject data before normalization (shape: {features.shape}):")  # , features.head())
    normalized_features = scaler.fit_transform(features)
    subject_data_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    subject_data_normalized['SubjectID'] = subject_data['SubjectID'].values
    subject_data_normalized['Gt_Depth'] = subject_data['Gt_Depth'].values
    return subject_data_normalized


def createFeatures(data_in, input_features=None):

    data_in['Vergence_Angle'], data_in['Vergence_Depth'] = zip(*data_in.apply(getEyeVergenceAngle, axis=1))

    # 1. Depth Normalization
    max_depth = data_in['Vergence_Depth'].max()
    min_depth = data_in['Vergence_Depth'].min()

    data_in['Normalized_Depth'] = (data_in['Vergence_Depth'] - min_depth) / (max_depth - min_depth)

    # 2. Directional Magnitude for Right and Left
    data_in['Directional_Magnitude_R'] = np.linalg.norm(
        data_in[['World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z']].values,
        axis=1)
    data_in['Directional_Magnitude_L'] = np.linalg.norm(
        data_in[['World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y', 'World_Gaze_Direction_L_Z']].values,
        axis=1)

    # 3. Gaze Direction Cosine Angles (already computed as Vergence_Angle)
    data_in['Cosine_Angles'] = np.cos(np.radians(data_in['Vergence_Angle']))

    # 4. Gaze Point Distance
    data_in['Gaze_Point_Distance'] = np.sqrt(
        (data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']) ** 2 +
        (data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']) ** 2)

    # 5. Normalized Vergence Angle
    max_angle = data_in['Vergence_Angle'].max()
    min_angle = data_in['Vergence_Angle'].min()
    data_in['Normalized_Vergence_Angle'] = 2 * (
            (data_in['Vergence_Angle'] - min_angle) / (max_angle - min_angle)) - 1

    # 6. Difference in World Gaze Direction
    data_in['Delta_Gaze_X'] = data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']
    data_in['Delta_Gaze_Y'] = data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']
    data_in['Delta_Gaze_Z'] = data_in['World_Gaze_Direction_R_Z'] - data_in['World_Gaze_Direction_L_Z']

    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    # 1. Angle between Gaze Vectors
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))

    gaze_r = data_in[['World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y', 'World_Gaze_Direction_R_Z']].values
    gaze_l = data_in[['World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y', 'World_Gaze_Direction_L_Z']].values
    data_in['Gaze_Vector_Angle'] = [angle_between_vectors(gaze_r[i], gaze_l[i]) for i in range(len(gaze_r))]

    # 2. Gaze Point Depth
    data_in['Gaze_Point_Depth_Difference'] = data_in['World_Gaze_Direction_R_Z'] - data_in[
        'World_Gaze_Direction_L_Z']

    # 3. Relative Changes
    # data_in['Relative_Change_Normalized_Depth'] = data_in['Normalized_Depth'].diff().fillna(0)
    data_in['Relative_Change_Vergence_Angle'] = data_in['Vergence_Angle'].diff().fillna(0)

    # 4. Ratios
    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_Delta_Gaze_XY'] = data_in['Delta_Gaze_X'] / data_in['Delta_Gaze_Y']

    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_World_Gaze_Direction_X'] = data_in['World_Gaze_Direction_R_X'] / data_in[
        'World_Gaze_Direction_L_X']
    data_in['Ratio_World_Gaze_Direction_Y'] = data_in['World_Gaze_Direction_R_Y'] / data_in[
        'World_Gaze_Direction_L_Y']
    data_in['Ratio_World_Gaze_Direction_Z'] = data_in['World_Gaze_Direction_R_Z'] / data_in[
        'World_Gaze_Direction_L_Z']

    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    data_in['Velocity_Gaze_Direction_R_X'] = data_in['World_Gaze_Direction_R_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_R_Y'] = data_in['World_Gaze_Direction_R_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Y'] = data_in['Velocity_Gaze_Direction_R_Y'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_R_Z'] = data_in['World_Gaze_Direction_R_Z'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Z'] = data_in['Velocity_Gaze_Direction_R_Z'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_X'] = data_in['World_Gaze_Direction_L_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_X'] = data_in['Velocity_Gaze_Direction_L_X'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_Y'] = data_in['World_Gaze_Direction_L_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Y'] = data_in['Velocity_Gaze_Direction_L_Y'].diff().fillna(0)

    data_in['Velocity_Gaze_Direction_L_Z'] = data_in['World_Gaze_Direction_L_Z'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Z'] = data_in['Velocity_Gaze_Direction_L_Z'].diff().fillna(0)

    data_in['Angular_Difference_Gaze_Directions'] = data_in['Gaze_Vector_Angle']  # Already computed above

    data_in['Interaction_Normalized_Depth_Vergence_Angle'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Assuming a rolling window of size 5
    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    data_in['Lag_1_Normalized_Depth'] = data_in['Normalized_Depth'].shift(1).fillna(0)

    # Relative Changes
    data_in['Diff_Normalized_Depth'] = data_in['Normalized_Depth'].diff()

    # Ratios
    data_in['Directional_Magnitude_Ratio'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_X_Ratio'] = data_in['World_Gaze_Direction_R_X'] / data_in['World_Gaze_Direction_L_X']
    data_in['Gaze_Direction_Y_Ratio'] = data_in['World_Gaze_Direction_R_Y'] / data_in['World_Gaze_Direction_L_Y']
    data_in['Gaze_Direction_Z_Ratio'] = data_in['World_Gaze_Direction_R_Z'] / data_in['World_Gaze_Direction_L_Z']

    # Angular Differences
    data_in['Angular_Difference_X'] = data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']

    # Higher Order Interactions (example)
    data_in['Depth_Angle_Interaction'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Distance between Gaze Points
    data_in['Gaze_Point_Euclidean_Distance'] = np.sqrt(
        (data_in['World_Gaze_Direction_R_X'] - data_in['World_Gaze_Direction_L_X']) ** 2 +
        (data_in['World_Gaze_Direction_R_Y'] - data_in['World_Gaze_Direction_L_Y']) ** 2
    )

    # Angle between Gaze Directions (using dot product)
    dot_product = (
            data_in['World_Gaze_Direction_R_X'] * data_in['World_Gaze_Direction_L_X'] +
            data_in['World_Gaze_Direction_R_Y'] * data_in['World_Gaze_Direction_L_Y'] +
            data_in['World_Gaze_Direction_R_Z'] * data_in['World_Gaze_Direction_L_Z']
    )
    magnitude_R = data_in['Directional_Magnitude_R']
    magnitude_L = data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_Angle'] = np.arccos(dot_product / (magnitude_R * magnitude_L))

    # Calculate acceleration
    # Velocity and acceleration for gaze direction L
    data_in['Acceleration_Gaze_Direction_L_X'] = data_in['Velocity_Gaze_Direction_L_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Y'] = data_in['Velocity_Gaze_Direction_L_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_L_Z'] = data_in['Velocity_Gaze_Direction_L_Z'].diff().fillna(0)

    # Velocity and acceleration for gaze direction L
    data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Y'] = data_in['Velocity_Gaze_Direction_R_Y'].diff().fillna(0)
    data_in['Acceleration_Gaze_Direction_R_Z'] = data_in['Velocity_Gaze_Direction_R_Z'].diff().fillna(0)

    data_in["Velocity_X"] = np.linalg.norm(
        data_in[["Velocity_Gaze_Direction_R_X", "Velocity_Gaze_Direction_R_Y", "Velocity_Gaze_Direction_R_Z"]].values,
        axis=1
    )

    data_in["Acceleration_X"] = np.linalg.norm(
        data_in[["Acceleration_Gaze_Direction_R_X", "Acceleration_Gaze_Direction_R_Y",
                 "Acceleration_Gaze_Direction_R_Z"]].values,
        axis=1
    )


    # logger.info("NaN count per column before dropping NaNs:")
    # logger.info(data_in.isna().sum())
    data_in = data_in.dropna()

    data_in = data_in.replace([np.inf, -np.inf], np.nan)
    data_in = data_in.dropna()
    data_in = data_in[input_features]

    print("Data_in has ", data_in.shape[0], " rows and ", data_in.shape[1], " columns.")


    return data_in

# def separate_features_and_targets(sequences, target_feature):
#     features = []
#     targets = []
#     for seq in sequences:
#         features.append(seq)
#         targets.append(seq[-1][target_feature])  # z. B. letztes Target in der Sequenz
#     return np.array(features), np.array(targets)

def separate_features_and_targets(sequences):
    features = [seq[0] for seq in sequences]
    targets = [seq[1] for seq in sequences]
    return features, targets


def binData(df, isGIW=False):
    # Step 1: Bin the target variable
    num_bins = 60  # You can adjust this number
    df['Gt_Depth_bin'] = pd.cut(df['Gt_Depth'], bins=num_bins, labels=False)

    # Step 2: Calculate mean count per bin
    bin_counts = df['Gt_Depth_bin'].value_counts()
    mean_count = bin_counts.mean()

    # Step 3: Resample each bin
    resampled_data = []
    for bin in range(num_bins):
        bin_data = df[df['Gt_Depth_bin'] == bin]
        bin_count = bin_data.shape[0]

        if bin_count == 0:
            continue  # Skip empty bins

        if bin_count < mean_count:
            # Oversample if count is less than mean
            bin_data_resampled = resample(bin_data, replace=True, n_samples=int(mean_count), random_state=123)
        elif bin_count > mean_count:
            # Undersample if count is more than mean
            bin_data_resampled = resample(bin_data, replace=False, n_samples=int(mean_count), random_state=123)
        else:
            # Keep the bin as is if count is equal to mean
            bin_data_resampled = bin_data

        resampled_data.append(bin_data_resampled)

    # Step 4: Combine back into a single DataFrame
    balanced_df = pd.concat(resampled_data)

    if isGIW:
        balanced_df = sample_from_bins(balanced_df)
        logger.info(balanced_df['Gt_Depth_bin'].value_counts(normalize=True))

    # Optionally, drop the 'Gt_Depth_bin' column if no longer needed
    balanced_df.drop('Gt_Depth_bin', axis=1, inplace=True)
    return balanced_df


def sample_from_bins(df, fraction=0.12):
    # Ensure that each bin has enough data points for sampling
    sampled_data = df.groupby('Gt_Depth_bin').apply(
        lambda x: x.sample(frac=fraction, random_state=110) if len(x) > 1 else x)

    return sampled_data.reset_index(drop=True)
