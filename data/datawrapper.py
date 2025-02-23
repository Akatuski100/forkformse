import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.embed import ChannelPositionalEmbedding

class data_loader:
    """
    A unified data loader class that:
      1. Reads a CSV file.
      2. Drops specified columns (by index or by name).
      3. Optionally extracts and drops a date column.
      4. Converts the remaining data to a torch.Tensor.
      5. Creates overlapping sequential batches (with optional zero padding).
      6. Provides token embedding transformation via an integrated module.
      7. Wraps everything in a PyTorch DataLoader.
    """
    class OverlapSequenceDataset(Dataset):
        def __init__(self, data: torch.Tensor, batch_size: int, overlap: int):
            self.data = data
            self.batch_size = batch_size
            self.overlap = overlap
            self.non_overlap = batch_size - overlap
            self.num_batches = math.ceil((len(data) - overlap) / self.non_overlap)

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx: int):
            start_idx = idx * self.non_overlap
            end_idx = start_idx + self.batch_size
            if end_idx > len(self.data):
                batch = self.data[start_idx:len(self.data)]
                pad_length = end_idx - len(self.data)
                if self.data.ndim == 1:
                    padding = torch.zeros(pad_length, dtype=self.data.dtype)
                else:
                    padding = torch.zeros((pad_length,) + self.data.shape[1:], dtype=self.data.dtype)
                batch = torch.cat([batch, padding], dim=0)
            else:
                batch = self.data[start_idx:end_idx]
            return batch

    class TokenEmbedding(nn.Module):
        def __init__(self, c_in, tao=12, m=8, pad=False):
            super(data_loader.TokenEmbedding, self).__init__()
            self.tao = tao
            self.m = m
            self.c_in = c_in
            self.pad = pad
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pos_embedding = ChannelPositionalEmbedding(c_in, self.m)

        def data_extract(self, ts_batch):
            n_seq, cin = ts_batch.shape
            n_valid = n_seq - self.m * self.tao
            if n_valid <= 0:
                raise ValueError(f"Invalid n_valid={n_valid}. Check seq_length, m, and tao values.")
            t_indices = torch.arange(self.m * self.tao, n_seq, device=ts_batch.device)
            offsets = torch.arange(0, self.m + 1, device=ts_batch.device) * self.tao
            time_indices = t_indices.unsqueeze(1) - offsets.unsqueeze(0)
            channel_idx = torch.arange(cin, device=ts_batch.device).view(1, cin, 1).expand(n_valid, cin, self.m + 1)
            time_idx_expanded = time_indices.unsqueeze(1).expand(n_valid, cin, self.m + 1)
            extracted = ts_batch[time_idx_expanded, channel_idx]
            faithful_vec = extracted.reshape(n_valid, cin * (self.m + 1))
            return faithful_vec

        def forward(self, x):
            batch_size, seq_len, cin = x.shape
            x_list = []
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                x = x.to(self.device)
            for batch_val in range(batch_size):
                ts_batch = x[batch_val]
                extracted_data = self.data_extract(ts_batch)
                x_list.append(extracted_data)
            x_embedded = torch.stack(x_list)
            if self.pad:
                x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))
            return x_embedded

    def __init__(self,
                 file_path: str,
                 drop_column_indices: list = None,
                 drop_columns_names: list = None,
                 date_column: str = None,
                 m=8,
                 tao=12,
                 batch_size_divisor: int = 100,
                 shuffle: bool = False,
                 **loader_kwargs):
        df = pd.read_csv(file_path)
        if drop_column_indices is not None and len(drop_column_indices) > 0:
            df = df.drop(df.columns[drop_column_indices], axis=1)
        if drop_columns_names is not None:
            df = df.drop(columns=drop_columns_names)
        if date_column is not None and date_column in df.columns:
            self.date_values = df[date_column].values
            df = df.drop(columns=[date_column])
        else:
            self.date_values = None
        self.cols = df.columns.tolist()
        data_np = df.to_numpy()
        self.data_tensor = torch.tensor(data_np, dtype=torch.float32)
        n_seq = self.data_tensor.shape[0]
        self.batch_size = max(1, int(n_seq / batch_size_divisor))
        self.m = m
        self.tao = tao
        self.overlap = m * tao
        self.dataset = data_loader.OverlapSequenceDataset(self.data_tensor, self.batch_size, self.overlap)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=shuffle, **loader_kwargs)

    def __iter__(self):
        for batch in self.data_loader:
            yield batch.squeeze(0)

    def __len__(self):
        print(f"Total data length: {len(self.data_tensor)}, Overlap: {self.overlap}, Non-overlap: {self.dataset.non_overlap}, Num batches: {self.dataset.num_batches}", flush=True)
        return len(self.dataset)

    def get_loader(self):
        return self.data_loader

    def transform_data(self, pad: bool = False, save_path: str = None):
        batches = [batch for batch in self]
        result = torch.stack(batches, dim=0)
        token_embedder = data_loader.TokenEmbedding(c_in=result.shape[2], tao=self.tao, m=self.m, pad=pad)
        embedded = token_embedder(result)
        flat_embedded = embedded.reshape(embedded.shape[0] * embedded.shape[1], embedded.shape[2])
        n_original = self.data_tensor.shape[0]
        n_valid = n_original - self.m * self.tao
        flat_embedded = flat_embedded[:n_valid, :]
        new_cols = [f"{feature}{i}" for feature in self.cols for i in range(1, self.m + 2)]
        if flat_embedded.shape[1] != len(new_cols):
            raise ValueError(f"Mismatch in feature dimensions: transformed data has {flat_embedded.shape[1]} features, "
                             f"but expected {len(new_cols)} based on the original columns.")
        new_df = pd.DataFrame(flat_embedded.cpu().detach().numpy(), columns=new_cols)
        if self.date_values is not None:
            new_df.insert(0, 'date', self.date_values[self.m * self.tao : self.m * self.tao + len(new_df)])
        if save_path is not None:
            new_df.to_csv(save_path, index=False)
        return new_df
