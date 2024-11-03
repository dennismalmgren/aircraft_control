import torch

class FGTable1D:
    def __init__(self, data, breakpoints, device='cpu'):
        """
        Initialize a 1D lookup table with interpolation.

        :param data: 1D tensor of values in the table (size N).
        :param breakpoints: 1D tensor of breakpoints corresponding to data points (size N).
        :param device: Device on which to store tensors (e.g., 'cpu' or 'cuda').
        """
        assert len(data) == len(breakpoints), "Data and breakpoints must have the same length."
        
        self.device = device
        self.data = torch.tensor(data, dtype=torch.float32, device=device)
        self.breakpoints = torch.tensor(breakpoints, dtype=torch.float32, device=device)
        self.size = len(data)

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
        query_points = torch.tensor(query_points, dtype=torch.float32, device=self.device)

        # Clip query points to the range of the breakpoints
        query_points_clamped = query_points.clamp(self.breakpoints[0], self.breakpoints[-1])

        # Locate the indices of the nearest left breakpoints for each query point
        left_indices = torch.searchsorted(self.breakpoints, query_points_clamped, right=True) - 1
        left_indices = left_indices.clamp(0, self.size - 2)  # Ensure indices are within bounds

        # Gather breakpoints and values for interpolation
        x0 = self.breakpoints[left_indices]
        y0 = self.data[left_indices]
        x1 = self.breakpoints[left_indices + 1]
        y1 = self.data[left_indices + 1]

        # Interpolate
        return self._linear_interpolate(query_points_clamped, x0, y0, x1, y1)

# Example Usage
# Define breakpoints and values for the table
breakpoints = [0.0, 1.0, 2.0, 3.0]
data = [10.0, 20.0, 30.0, 40.0]

# Instantiate the table
table_1d = FGTable1D(data, breakpoints, device='cpu')

# Query the table
query_points = torch.tensor([0.5, 1.5, 2.5, 3.5])
print(table_1d.get_value(query_points))