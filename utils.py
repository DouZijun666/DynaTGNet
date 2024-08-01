import torch
import torch.utils.data
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import TensorDataset,DataLoader,Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import scipy.stats as st
import pywt

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets all the meter values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int): The number of occurrences of this value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a string representation of the meter."""
        fmtstr = f'{self.name} {self.val{self.fmt}} ({self.avg{self.fmt}})'
        return fmtstr.format(**self.__dict__)



def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model predictions with shape (N, C), where N is batch size and C is number of classes.
        target (torch.Tensor): Ground truth labels with shape (N).
        topk (tuple): A tuple of integers specifying the top-k accuracy to compute.

    Returns:
        list: List of top-k accuracies.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Check if the predictions are correct
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Compute accuracy for each k in topk
        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracy_k = (correct_k * 100.0 / batch_size).item()
            accuracies.append(accuracy_k)

        return accuracies


def setup_logger(log_file: str):
    """
    Sets up a logger to write messages to the specified log file.

    Args:
        log_file (str): Path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

def log_msg(message: str, log_file: str):
    """
    Logs a message to the specified log file.

    Args:
        message (str): The message to log.
        log_file (str): Path to the log file.
    """
    # Setup the logger
    setup_logger(log_file)
    
    # Log the message
    logger = logging.getLogger()
    logger.info(message)


        
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_excel(file_path, usecols=range(2, 34))
    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {e}")
    return df

def denoise_sensor_data(sensor_data: pd.Series, method: str, threshold: float) -> pd.Series:
    """
    Applies Discrete Wavelet Transform (DWT) to denoise sensor data.

    Args:
        sensor_data (pd.Series): Sensor data to denoise.
        method (str): Wavelet method to use.
        threshold (float): Threshold for wavelet coefficients.

    Returns:
        pd.Series: Denoised sensor data.
    """
    coeffs = pywt.wavedec(sensor_data, method, level=6)
    # Threshold high-frequency coefficients
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]]
    denoised_data = pywt.waverec(coeffs, method)
    return pd.Series(denoised_data)

def dwt(file_path: str) -> pd.DataFrame:
    """
    Applies Discrete Wavelet Transform (DWT) to denoise data from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Denoised data.
    """
    df = load_data(file_path)
    
    if df.shape[0] != 6000:
        return df

    method = 'db8'
    threshold = 0.5
    
    denoised_df = df.copy()

    for i in range(32):
        sensor_data = df[f"ai{i}"]
        denoised_df[f"ai{i}"] = denoise_sensor_data(sensor_data, method, threshold)
    
    return denoised_df



def load_labels(file_path: str) -> pd.DataFrame:
    """
    Loads the labels from the specified Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded labels.
    """
    try:
        df = pd.read_excel(file_path)
        return df.values[:300, 1:3]
    except Exception as e:
        raise RuntimeError(f"Error loading labels from {file_path}: {e}")

def process_file(file_path: str) -> Any:
    """
    Applies DWT to process data from the specified file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        np.ndarray: Processed data.
    """
    try:
        return dwt(file_path)
    except Exception as e:
        raise RuntimeError(f"Error processing file {file_path}: {e}")

def determine_label(value: float) -> int:
    """
    Determines the label based on the provided value.

    Args:
        value (float): Value to determine the label.

    Returns:
        int: The determined label.
    """
    if 7 < value <= 12:
        return 1
    else:
        return 2

def get_data_labels(base_path: str) -> Tuple[List[Any], List[int]]:
    """
    Retrieves and processes data and labels from the specified base path.

    Args:
        base_path (str): Base path for the data files.

    Returns:
        Tuple[List[np.ndarray], List[int]]: Processed data and corresponding labels.
    """
    labels_file = f'{base_path}loss.xlsx'
    table = load_labels(labels_file)
    
    datas = []
    labels = []
    count = 0
    
    for index, value in table:
        count += 1
        file_path = f'{base_path}Selected_data/{int(index)}.xlsx'
        file_data = process_file(file_path)
        
        if file_data.shape[0] != 6000:
            continue
        
        datas.append(file_data)
        labels.append(determine_label(value))
        
        if count == 300:
            break
    
    return datas, labels


class MyDataset(Dataset):
    """
    Custom Dataset class for handling data and labels.

    Args:
        datas (List[np.ndarray]): List of data arrays.
        labels (List[int]): List of labels corresponding to the data arrays.
    """
    
    def __init__(self, datas: List[np.ndarray], labels: List[int]):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Retrieves the data and label at the specified index.

        Args:
            index (int): Index of the data and label to retrieve.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the data and its corresponding label.
        """
        data = self.datas[index]
        label = self.labels[index]
        return data, label

    def __len__(self) -> int:
        """
        Returns the number of data-label pairs in the dataset.

        Returns:
            int: Number of data-label pairs.
        """
        return len(self.datas)

    
def smooth(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Smooths the input data using a moving average with a specified window size.

    Args:
        data (np.ndarray): Input data array to be smoothed.
        window_size (int): Size of the moving average window.

    Returns:
        np.ndarray: Smoothed data array.
    
    Raises:
        ValueError: If the data length is less than the window size.
    """
    if data.shape[0] < window_size:
        raise ValueError(f"Data length must be at least as large as the window size. Current length: {data.shape[0]}, window size: {window_size}")

    # Calculate moving average
    smoothed_data = np.array([
        np.mean(data[i:i + window_size], axis=0) 
        for i in range(0, len(data), window_size)
    ])
    
    return smoothed_data

def scaler(data: np.ndarray, fit: bool = True) -> Optional[np.ndarray]:
    """
    Scales the input data using StandardScaler.

    Args:
        data (np.ndarray): Input data array to be scaled.
        fit (bool): If True, fit the scaler to the data and return the scaled data.
                    If False, only return the scaler (useful for applying the same scaling to different datasets).

    Returns:
        Optional[np.ndarray]: Scaled data array if fit is True, otherwise None.
    """
    scaler = StandardScaler()
    
    if fit:
        scaled_data = scaler.fit_transform(data)
        return scaled_data
    else:
        return None

    
def get_default_train_val_test_loader(args: Any) -> Tuple[DataLoader, DataLoader, int, int, int]:
    """
    Loads training and validation data, and creates DataLoader objects for each dataset.

    Args:
        args (Any): Arguments object containing dataset paths, batch sizes, and other parameters.

    Returns:
        Tuple[DataLoader, DataLoader, int, int, int]: 
            - DataLoader for the training set
            - DataLoader for the validation set
            - Number of nodes (features) in the dataset
            - Sequence length of the data
            - Number of unique classes in the dataset
    """
    # Extract dataset identifier
    dataset_id = args.dataset

    # Load datasets from .npy files
    data_train = np.load(f'/root/autodl-tmp/MTS_DATA/{dataset_id}/X_train.npy')
    data_val = np.load(f'/root/autodl-tmp/MTS_DATA/{dataset_id}/X_valid.npy')
    label_train = np.load(f'/root/autodl-tmp/MTS_DATA/{dataset_id}/y_train.npy')
    label_val = np.load(f'/root/autodl-tmp/MTS_DATA/{dataset_id}/y_valid.npy')

    # Initialize number of nodes, sequence length, and number of classes
    num_nodes = data_val.shape[-1]
    seq_length = data_val.shape[-2]
    num_classes = len(np.unique(label_train)) + 1

    # Convert data and labels to Dataset objects
    train_dataset = MyDataset(data_train, label_train)
    val_dataset = MyDataset(data_val, label_val)

    # Create DataLoader objects for training and validation datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    return train_loader, val_loader, num_nodes, seq_length, num_classes
