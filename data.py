import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

def get_train_test(path, batch_size=32):
    # Number of genes in rna: 5540
    # Number of genes in scna: 5507
    # Number of genes in methy: 4846
    # Total number of genes: 15893
    # Total number of samples in the final dataset: 475
    data = pd.read_csv(path)

    # Separate to make sure that there's an even distribution of 1s and 0s in train and test
    # 352 0's
    # 123 1's
    data_ones = data[data.iloc[:, -1] == 1]
    data_zeros = data[data.iloc[:, -1] == 0]

    # Split the data into train and validation sets.
    # Train test split is 80 20
    train_data_ones, test_data_ones, train_labels_ones, test_labels_ones = train_test_split(
        data_ones.iloc[:, 1:-2], data_ones.iloc[:, -2:], test_size=0.2, random_state=42)

    train_data_zeros, test_data_zeros, train_labels_zeros, test_labels_zeros = train_test_split(
        data_zeros.iloc[:, 1:-2], data_zeros.iloc[:, -2:], test_size=0.2, random_state=42)

    # Concatenate in the end to make train and test
    train_data = pd.concat((train_data_ones, train_data_zeros))
    train_labels = pd.concat((train_labels_ones, train_labels_zeros))
    test_data = pd.concat((test_data_ones, test_data_zeros)) 
    test_labels = pd.concat((test_labels_ones, test_labels_zeros))

    # #rna
    # print(train_data.columns[0])
    # print(train_data.columns[5539])
    # #scna
    # print(train_data.columns[5540])
    # print(train_data.columns[11046])
    # #methy
    # print(train_data.columns[11047])
    # print(train_data.columns[15892])
    # #labels
    # print(train_labels.columns)

    # Create Tensor datasets
    train_dataset = TensorDataset(torch.tensor(train_data.values, dtype=torch.float32), torch.tensor(train_labels.values, dtype=torch.float32))
    test_dataset  = TensorDataset(torch.tensor(test_data.values, dtype=torch.float32), torch.tensor(test_labels.values, dtype=torch.float32))

    # Create DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader