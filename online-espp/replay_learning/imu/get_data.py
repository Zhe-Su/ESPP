import zipfile
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

zip_file_path = '/home/tasci/Documents/repo_thesis/thesis_imu_har/data/archive.zip'

# Calculate class weights for training based on the balanced data
def calculate_class_weights(y):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict

def fold(n, data):
    # Separate the data for the `n`-th subject as test data
    test = data[data['Object'] == n]
    # Use all other subjects as training data
    train = data[(data['Object'] != n)] # ADDDED THIS LINE FOR USING THE SUBSET OF THE DATASET:  & (data['Object'] < 3)
    return train, test

def most_common(lst): # returns the most frequent element in the list lst
    return max(set(lst), key=lst.count)

def load_recGym_data(zip_file_path:str, window_length:int=40, stride:int=40, sensor:str='imu', subject:int=10, DEBUG:bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if DEBUG:
        print("NumPy version:", np.__version__)
        print("SciPy version:", scipy.__version__)
        print("scikit-learn version:", sklearn.__version__)

    # Load the csv file from the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        file_list = zf.namelist()
        print("Files in ZIP archive for RecGym data:", file_list)
        
        csv_files = [file_name for file_name in file_list if file_name.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the ZIP archive for RecGym data.")
        else:
            csv_filename = csv_files[0]
            print("CSV file found:", csv_filename)
            
            with zf.open(csv_filename) as csvfile:
                data_session = pd.read_csv(csvfile)
            
    # Filter the dataset to only include wrist data
    data_session = data_session[data_session['Sensor_Position'] == 'wrist']
    
    train, test = fold(subject, data_session)
    train = train.drop(['Object', 'Day', 'Sensor_Position'], axis=1)
    test = test.drop(["Object", "Day", "Sensor_Position"], axis=1)
    
    if DEBUG:
        print("Shape train: ", train.shape)

    # Select features based on sensor type
    if sensor == "cap": ## Cap only
        X_train = train["Body_Capacitance"].to_numpy()
        X_test= test["Body_Capacitance"].to_numpy()
        channel = 1
    elif sensor == "imu": ## IMU only
        X_train = train[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]].to_numpy()
        X_test = test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]].to_numpy()
        channel = 6
    elif sensor == "combine": ## Cap and IMU
        X_train = train[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "Body_Capacitance"]].to_numpy()
        X_test = test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "Body_Capacitance"]].to_numpy()
        channel = 7
    else:
        raise ValueError("Invalid sensor type specified. Choose from 'cap', 'imu', or 'combine'.")

    
    # For the demonstrator we only want these classes
    valid_classes = ["Running", "RopeSkipping", "Squat", "Walking", "Riding", "Null"]
    
    # Sliding window segmentation: Training data
    segmented_X_train = []
    segmented_y_train = []
    for i in range(0, len(X_train) - window_length + 1, stride):
        # Get the window of sensor data
        window_x = X_train[i:i+window_length]
        # Get the corresponding labels from the first column of the DataFrame
        window_y = train.iloc[i:i+window_length, 0].to_numpy()
        # Determine the most common label in the window
        majority_label = most_common(list(window_y))
        if majority_label in valid_classes:
            segmented_X_train.append(window_x.reshape(window_length, channel))
            segmented_y_train.append(majority_label)

    # Sliding window segmentation: Test data
    segmented_X_test = []
    segmented_y_test = []
    for i in range(0, len(X_test) - window_length + 1, stride):
        window_x = X_test[i:i+window_length]
        window_y = test.iloc[i:i+window_length, 0].to_numpy()
        majority_label = most_common(list(window_y))
        if majority_label in valid_classes:
            segmented_X_test.append(window_x.reshape(window_length, channel))
            segmented_y_test.append(majority_label)
    
    # Convert segmented lists to numpy arrays
    X_train = np.array(segmented_X_train)
    y_train = np.array(segmented_y_train)
    X_test = np.array(segmented_X_test)
    y_test = np.array(segmented_y_test)
    
    if DEBUG:
        print("X_train shape after sliding window segmentation:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("Unique training labels:", np.unique(y_train))
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
        print("Unique testing labels:", np.unique(y_test))
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train)
    y_test_int = label_encoder.transform(y_test)
    
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", mapping)
    
    if DEBUG:
        print("y_test shape (after encoding):", y_test_int.shape)
        print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
        
        
    return X_train, X_test, y_train_int, y_test_int


if __name__ == "__main__":
    # # Example usage
    # zip_file_path = './imu/archive.zip'
    # X_train, X_test, y_train, y_test = load_recGym_data(zip_file_path, sensor='imu', subject=10, DEBUG=True)

    # print(X_train.mean(axis=(1, 2))) # should give roughly 0.5 
    # print(X_test.mean(axis=(1, 2))) # should give roughly 0.5
    # print(y_train.shape)
    # print(y_test.shape)
    from src.util.dataloading import create_imu_dataloaders
    train_loader, test_loader = create_imu_dataloaders(batch_size=16, window_length=40, stride=40, num_workers=2, drop_last=True)
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)