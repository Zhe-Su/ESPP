import torch
import torch.nn.functional as F
import tonic
import torchvision
import numpy as np

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tonic import DiskCachedDataset, MemoryCachedDataset

import tonic.transforms as transforms
from tonic.datasets import SHD, NMNIST, DVSGesture
from torchvision.transforms import Compose
from sklearn.utils.class_weight import compute_class_weight

# from dhp19.dataset import HDF5Dataset as TennisDataset
from imu.get_data import load_recGym_data
# from imu.fortiss_data.preprocess import load_all_csvs_in_dir as load_fortiss_imu_data
from imu.BinaryEncoder import Binary_Encoder


def create_nmnist_dataloaders(batch_size: int, time_window: int, filter_time:int, save_path:str='./data', cache_path:str='./cache/nmnist/', num_workers:int=2, reset_cache:bool=True, drop_last:bool=False, split_train:bool=False, flatten:bool=False) -> tuple[DataLoader, DataLoader]:
    sensor_size = NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=filter_time),
                                        transforms.ToFrame(sensor_size=sensor_size,
                                                            n_time_bins=30)
                                        ])

    trainset = NMNIST(save_to=save_path, transform=frame_transform, train=True)
    testset = NMNIST(save_to=save_path, transform=frame_transform, train=False)
    
    if flatten:
        train_transform = Compose([
            torch.from_numpy, 
            torchvision.transforms.RandomRotation([-10,10]),
            lambda x: torch.flatten(x, start_dim=1)
        ])
        test_transform = Compose([
            torch.from_numpy, 
            lambda x: torch.flatten(x, start_dim=1)
        ])
    else:
        train_transform = Compose([
            torch.from_numpy, 
            torchvision.transforms.RandomRotation([-10,10]),
        ])
        test_transform = Compose([
            torch.from_numpy, 
        ])
    
    if split_train: 
        trainset, valset = torch.utils.data.random_split(trainset, [0.99, 0.01])

        cached_trainset = DiskCachedDataset(trainset, transform=train_transform, cache_path=cache_path + "train/", reset_cache=reset_cache)
        # cached_trainset = MemoryCachedDataset(trainset)
        cached_valset = DiskCachedDataset(valset, transform=train_transform, cache_path=cache_path + "val/", reset_cache=reset_cache)


        # no augmentations for the testset
        cached_testset = DiskCachedDataset(testset, transform=test_transform, cache_path=cache_path + "test/", reset_cache=reset_cache)
        # cached_testset = MemoryCachedDataset(testset)
        train_loader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=num_workers, drop_last=drop_last)
        val_loader = DataLoader(cached_valset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=num_workers, drop_last=drop_last)
        test_loader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=num_workers, drop_last=drop_last)
        return train_loader, val_loader, test_loader
    else:
        cached_trainset = DiskCachedDataset(trainset, transform=train_transform, cache_path=cache_path + "train/", reset_cache=reset_cache)
        cached_testset = DiskCachedDataset(testset, transform=test_transform, cache_path=cache_path + "test/", reset_cache=reset_cache)
        
        train_loader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=num_workers, drop_last=drop_last)
        test_loader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return train_loader, test_loader



def create_poisson_mnist_dataloaders(batch_size:int, bins:int=30, save_path:str="./data", num_workers:int=2, drop_last:bool=False, flatten:bool=False) -> tuple[DataLoader, DataLoader]:
    def poisson_enc(x:torch.Tensor) -> torch.Tensor:
        x = x * 255 / 4 # conversion to Hz
        x = x / 1000 # conversion to mHz

        enc = torch.rand(bins, 28, 28)
        enc = (enc < x[0,None,...])
        return enc[:,None,...]
    
    
    transform = Compose([
        torchvision.transforms.ToTensor(),
        poisson_enc
        # torchvision.transforms.RandomRotation([-10,10]),
    ])

    trainset = torchvision.datasets.MNIST(save_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(save_path, train=False, transform=transform, download=True)
    

    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, num_workers=num_workers, drop_last=drop_last)

    return train_loader, test_loader



def create_shd_dataloaders(batch_size:int, bins:int=30, filter_time:int=10000, save_path:str='./data', cache_path:str="./cache/shd_new/", num_workers:int=2, reset_cache:bool=True) -> tuple[DataLoader, DataLoader]:
    shd_sensor_size = SHD.sensor_size
    if filter_time > 0:
        frame_transform = Compose([
            # transforms.Denoise(filter_time=filter_time),
            tonic.transforms.ToFrame(
                sensor_size=shd_sensor_size, 
                n_time_bins=bins,
            ),
            lambda x: x[:,0,:],
            torch.from_numpy
        ])
    else:
        frame_transform = Compose([
            tonic.transforms.ToFrame(
                sensor_size=shd_sensor_size, 
                n_time_bins=bins,
            ),
            lambda x: x[:,0,:],
            torch.from_numpy
        ])

    train_dataset = SHD(save_to=save_path, train=True, transform=frame_transform)
    test_dataset = SHD(save_to=save_path, train=False, transform=frame_transform)

    train_dataset = MemoryCachedDataset(train_dataset)
    test_dataset = MemoryCachedDataset(test_dataset)

    # train_dataset = DiskCachedDataset(train_dataset, cache_path=cache_path + "train/")#, reset_cache=reset_cache)
    # test_dataset = DiskCachedDataset(test_dataset, cache_path=cache_path + "test/")#,  reset_cache=reset_cache)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=num_workers)
    return train_loader, test_loader


def create_dvs_dataloaders(batch_size:int, bins:int, filter_time:int, downsample:bool=False, save_path:str='./data', cache_path:str="./cache/dvs/", num_workers:int=2, reset_cache:bool=True, drop_last:bool=False) -> tuple[DataLoader, DataLoader]:
    sensor_size = DVSGesture.sensor_size
    frame_transform = []
    if filter_time > 0:
        frame_transform.append(transforms.Denoise(filter_time=filter_time))
    frame_transform.append(transforms.ToFrame(sensor_size=sensor_size, n_time_bins=bins))
    if downsample:
        frame_transform.append(lambda x: F.max_pool2d(torch.Tensor(x), kernel_size=2, stride=2))
    frame_transform = transforms.Compose(frame_transform)

    train_dataset = DVSGesture(save_to=save_path, train=True, transform=frame_transform)
    test_dataset = DVSGesture(save_to=save_path, train=False, transform=frame_transform)

    cache_transform = transforms.Compose([
        lambda x: torch.Tensor(x),
        torchvision.transforms.CenterCrop(128),
        # torchvision.transforms.RandomErasing(),
        torchvision.transforms.RandomRotation(degrees=30),#
        lambda x: (x > 0.5).float()
    ])

    train_dataset = DiskCachedDataset(train_dataset, cache_path=cache_path + "train/", reset_cache=reset_cache, transform=cache_transform)
    test_dataset = DiskCachedDataset(test_dataset, cache_path=cache_path + "test/",  reset_cache=reset_cache, transform=lambda x: torch.Tensor((x > 0.5)).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=num_workers, drop_last=drop_last)
    return train_loader, test_loader



def create_tennis_dataloaders(batch_size:int, num_bins:int, down_input:int=5, apply_transform:bool=True, num_workers:int=2, drop_last:bool=False) -> tuple[DataLoader, DataLoader]:
    multiclass = False # l
    sequence_length = 1
    bin_time = 5 # larger means fewer samples
    apply_transform = apply_transform
    stride = 4 # does not change anything
    down_input = down_input # probably downsampling factor for image resoluttion 
    binary = True

    train_directory = 'dhp19/data/train_streams_multiclass/' if multiclass else 'dhp19/data/actions_front/train/' 
    test_directory = 'dhp19/data/test_streams_multiclass/' if multiclass else 'dhp19/data/actions_front/test/'
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x[0,...]), # (1, C, H, W, T) -> (C, H, W, T)
        torchvision.transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)) # (C, H, W, T) -> (T, C, H, W)
    ])

    train_dataset = TennisDataset(
        train_directory, 
        sequence_length=sequence_length, 
        bins_per_seq=num_bins,
        bin_time=bin_time, 
        stride=stride, 
        down_input=down_input, 
        apply_transform=apply_transform, 
        classification=True, 
        binary=binary, 
        transform=transform
    )

    test_dataset = TennisDataset(
        test_directory, 
        sequence_length=sequence_length, 
        bins_per_seq=num_bins,
        bin_time=bin_time, 
        stride=stride, 
        down_input=down_input, 
        apply_transform=False, 
        classification=True, 
        binary=binary,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False), num_workers=num_workers, drop_last=drop_last)
    return train_loader, test_loader


def create_imu_dataloaders(batch_size:int, window_length:int, stride:int, num_workers:int=2, drop_last:bool=False, binary:bool=False, balanced_sampling:bool=False, split_data:bool=False) -> tuple[DataLoader, DataLoader]:
    def time_first(data:tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        x, y = zip(*data)
        return torch.stack(x, axis=1).float(), torch.stack(y).long()

    zip_file_path = './imu/archive.zip'
    X_train, X_test, y_train, y_test = load_recGym_data(zip_file_path, window_length=window_length, stride=stride,  sensor='imu', subject=10, DEBUG=False)
    if balanced_sampling:
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        sample_weight = torch.tensor([class_weights_dict[label] for label in y_train])
        sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(y_train), replacement=True)
    
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    

    if not balanced_sampling:
        X_train = X_train[y_train != 0]
        y_train = y_train[y_train != 0]
        X_test = X_test[y_test != 0]
        y_test = y_test[y_test != 0]

        y_train = y_train - 1
        y_test = y_test - 1
        
    if binary:
        bin_enc = Binary_Encoder(n_bits=10)
        X_train = bin_enc(X_train).flatten(start_dim=2)
        X_test = bin_enc(X_test).flatten(start_dim=2)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)


    if split_data: 
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last)
    else:
        val_loader = []

    if balanced_sampling:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last, sampler=sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last)
    
    return train_loader, val_loader, test_loader



def create_fortiss_dataloaders(batch_size:int, window_sec:int=2, stride_sec:int=2, frequency:int=20, num_workers:int=2, shuffle:bool=False, drop_last:bool=False, binary:bool=False, balanced_sampling:bool=False, split_data:bool=False) -> tuple[DataLoader, DataLoader]:
    def time_first(data:tuple[list[torch.Tensor], list[torch.Tensor]]) -> torch.Tensor:
        x, y = zip(*data)
        return torch.stack(x, axis=1).float(), torch.stack(y).long()

    train_file_path = './imu/fortiss_data/train'
    test_file_path = './imu/fortiss_data/test'
    X_train, y_train = load_fortiss_imu_data(train_file_path, target_fs=frequency, window_size_sec=window_sec, stride_sec=stride_sec)
    X_test, y_test = load_fortiss_imu_data(test_file_path, target_fs=frequency, window_size_sec=window_sec, stride_sec=stride_sec)
    
    if balanced_sampling:
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        sample_weight = torch.tensor([class_weights_dict[label] for label in y_train])
        sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(y_train), replacement=True)
    
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    

    if not balanced_sampling:
        X_train = X_train[y_train != 0]
        y_train = y_train[y_train != 0]
        X_test = X_test[y_test != 0]
        y_test = y_test[y_test != 0]
        
    if binary:
        bin_enc = Binary_Encoder(n_bits=10)
        X_train = bin_enc(X_train).flatten(start_dim=2)
        X_test = bin_enc(X_test).flatten(start_dim=2)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)


    if split_data: 
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last)
    else:
        val_loader = []

    if balanced_sampling:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last, sampler=sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=time_first, num_workers=num_workers, drop_last=drop_last)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, _, test_loader = create_fortiss_dataloaders(50, num_workers=1, binary=True)
    x, y = next(iter(train_loader))
    print(x.shape)