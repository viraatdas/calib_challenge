import cv2
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def extract_optical_flow_features(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_flow_magnitude = np.mean(flow_magnitude)
        flow_features.append(mean_flow_magnitude)
        prev_frame_gray = frame_gray
    cap.release()
    return np.array(flow_features)

def load_labels(txt_path):
    return np.loadtxt(txt_path)

def process_labeled_videos(folder_path):
    features = []
    yaw_labels = []
    roll_labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.hevc'):
            video_path = os.path.join(folder_path, filename)
            txt_path = os.path.join(folder_path, filename.replace('.hevc', '.txt'))
            video_features = extract_optical_flow_features(video_path)
            video_labels = load_labels(txt_path)
            if len(video_features) == len(video_labels):
                features.append(video_features)
                yaw_labels.append(video_labels[:, 0])
                roll_labels.append(video_labels[:, 1])
    return np.concatenate(features), np.concatenate(yaw_labels), np.concatenate(roll_labels)
    
def train_models_and_plot(X, yaw_labels, roll_labels):
    # Splitting the dataset for evaluation
    X_train, X_test, y_train_yaw, y_test_yaw = train_test_split(X, yaw_labels, test_size=0.2, random_state=42)
    _, _, y_train_roll, y_test_roll = train_test_split(X, roll_labels, test_size=0.2, random_state=42)

    eval_set_yaw = [(X_train.reshape(-1, 1), y_train_yaw), (X_test.reshape(-1, 1), y_test_yaw)]
    eval_set_roll = [(X_train.reshape(-1, 1), y_train_roll), (X_test.reshape(-1, 1), y_test_roll)]
    
    yaw_model = xgb.XGBRegressor(objective='reg:squarederror')
    roll_model = xgb.XGBRegressor(objective='reg:squarederror')
    
    yaw_model.fit(X_train.reshape(-1, 1), y_train_yaw, eval_metric="rmse", eval_set=eval_set_yaw, verbose=False)
    roll_model.fit(X_train.reshape(-1, 1), y_train_roll, eval_metric="rmse", eval_set=eval_set_roll, verbose=False)

    # Plotting training and testing loss for Yaw Model
    results_yaw = yaw_model.evals_result()
    epochs = len(results_yaw['validation_0']['rmse'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results_yaw['validation_0']['rmse'], label='Train Yaw')
    ax.plot(x_axis, results_yaw['validation_1']['rmse'], label='Test Yaw')
    ax.legend()
    plt.ylabel('RMSE')
    plt.title('XGBoost Yaw Model RMSE')
    plt.show()

    # Plotting training and testing loss for Roll Model
    results_roll = roll_model.evals_result()
    epochs = len(results_roll['validation_0']['rmse'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results_roll['validation_0']['rmse'], label='Train Roll')
    ax.plot(x_axis, results_roll['validation_1']['rmse'], label='Test Roll')
    ax.legend()
    plt.ylabel('RMSE')
    plt.title('XGBoost Roll Model RMSE')
    plt.show()

    return yaw_model, roll_model

def predict_and_save_unlabeled_videos(yaw_model, roll_model, folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.hevc'):
            video_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename.replace('.hevc', '.txt'))
            video_features = extract_optical_flow_features(video_path)
            yaw_predictions = yaw_model.predict(video_features.reshape(-1, 1))
            roll_predictions = roll_model.predict(video_features.reshape(-1, 1))
            combined_predictions = np.vstack((yaw_predictions, roll_predictions)).T
            np.savetxt(output_path, combined_predictions)

# Main workflow
labeled_folder = 'labeled'
unlabeled_folder = 'unlabeled'
output_folder = 'predicted_labels'

X, yaw_labels, roll_labels = process_labeled_videos(labeled_folder)
yaw_model, roll_model = train_models(X, yaw_labels, roll_labels)
predict_and_save_unlabeled_videos(yaw_model, roll_model, unlabeled_folder, output_folder)

