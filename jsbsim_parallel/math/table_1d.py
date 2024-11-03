from typing import Tuple

import torch

class Table1D:
    def __init__(self, N: int, data = None, device='cpu'):
        """
        Initialize a 1D lookup table with interpolation.

        :param data: 1D tensor of values in the table (size N).
        :param breakpoints: 1D tensor of breakpoints corresponding to data points (size N).
        :param device: Device on which to store tensors (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.N = N
        if data is None:
            self.data = torch.zeros(N, 2, dtype=torch.float32, device=device)
        else:
            self.data = data
       

    def set_data(self, data: torch.Tensor):
        self.data.copy_(data)

    def _linear_interpolate(self, x, x0, y0, x1, y1):
        """
        Perform linear interpolation.
        
        :param x: Query points (tensor) for which we need interpolated values.
        :param x0: Left breakpoint (tensor).
        :param y0: Value at x0 (tensor).
        :param x1: Right breakpoint (tensor).
        :param y1: Value at x1 (tensor).
        :return: Interpolated values (tensor).
        """
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def get_value(self, query_points):
        """
        Get interpolated values from the 1D table based on query points.

        :param query_points: 1D tensor of query points for which to interpolate values.
        :return: 1D tensor of interpolated values.
        """
        # Clip query points to the range of the breakpoints
        query_points_clamped = query_points.clamp(self.data[0, 0], self.data[-1, 0])

        # Locate the indices of the nearest left breakpoints for each query point
        left_indices = torch.searchsorted(self.data[:, 0], query_points_clamped, right=True) - 1
        left_indices = left_indices.clamp(0, self.size - 2)  # Ensure indices are within bounds

        # Gather breakpoints and values for interpolation
        x0 = self.data[left_indices, 0]
        y0 = self.data[left_indices, 1]
        x1 = self.data[left_indices + 1, 0]
        y1 = self.data[left_indices + 1, 1]

        # Interpolate
        return self._linear_interpolate(query_points_clamped, x0, y0, x1, y1)

    
    def __len__(self):
        """
        Returns the number of breakpoints in the table.

        :return: Integer representing the number of breakpoints.
        """
        return len(self.data)
    
    def __getitem__(self, indices):
        """
        Support bracket-style indexing for retrieving interpolated values.

        :param indices: Indices to retrieve the interpolated value.
                        Can be a single index (int or float) or a tuple of indices.
        :return: Interpolated value(s) based on the index/indices provided.
        """
        if type(indices) is int:
            return self.data[indices, 1] #return the data.
        else:
            return self.data[*indices]

# Example Usage
# Define breakpoints and values for the table
# breakpoints = [0.0, 1.0, 2.0, 3.0]
# data = [10.0, 20.0, 30.0, 40.0]

# # Instantiate the table
# table_1d = Table1D(data, breakpoints, device='cpu')

# # Query the table
# query_points = torch.tensor([0.5, 1.5, 2.5, 3.5])
# print(table_1d.get_value(query_points))