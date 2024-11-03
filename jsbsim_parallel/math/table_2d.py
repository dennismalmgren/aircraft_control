import torch

class Table2D:
    def __init__(self, data, row_breakpoints, col_breakpoints, device='cpu'):
        """
        Initialize a 2D lookup table with bilinear interpolation.

        :param data: 2D tensor of values in the table (size R x C).
        :param row_breakpoints: 1D tensor of row breakpoints (size R).
        :param col_breakpoints: 1D tensor of column breakpoints (size C).
        :param device: Device on which to store tensors (e.g., 'cpu' or 'cuda').
        """
        assert data.shape[0] == len(row_breakpoints), "Data row size must match row breakpoints length."
        assert data.shape[1] == len(col_breakpoints), "Data column size must match column breakpoints length."

        self.device = device
        self.data = torch.tensor(data, dtype=torch.float32, device=device)
        self.row_breakpoints = torch.tensor(row_breakpoints, dtype=torch.float32, device=device)
        self.col_breakpoints = torch.tensor(col_breakpoints, dtype=torch.float32, device=device)
        self.num_rows, self.num_cols = data.shape

    def _bilinear_interpolate(self, x, y, x0, y0, x1, y1, f00, f01, f10, f11):
        """
        Perform bilinear interpolation.

        :param x, y: Query points.
        :param x0, y0: Coordinates of the top-left corner.
        :param x1, y1: Coordinates of the bottom-right corner.
        :param f00: Value at (x0, y0).
        :param f01: Value at (x0, y1).
        :param f10: Value at (x1, y0).
        :param f11: Value at (x1, y1).
        :return: Interpolated values.
        """
        return (
            f00 * (x1 - x) * (y1 - y) +
            f10 * (x - x0) * (y1 - y) +
            f01 * (x1 - x) * (y - y0) +
            f11 * (x - x0) * (y - y0)
        ) / ((x1 - x0) * (y1 - y0))

    def get_value(self, row_query, col_query):
        """
        Get interpolated values from the 2D table based on row and column query points.

        :param row_query: 1D tensor of row query points.
        :param col_query: 1D tensor of column query points.
        :return: 1D tensor of interpolated values.
        """
        row_query = torch.tensor(row_query, dtype=torch.float32, device=self.device)
        col_query = torch.tensor(col_query, dtype=torch.float32, device=self.device)

        # Clamp row and column queries to the range of breakpoints
        row_query_clamped = row_query.clamp(self.row_breakpoints[0], self.row_breakpoints[-1])
        col_query_clamped = col_query.clamp(self.col_breakpoints[0], self.col_breakpoints[-1])

        # Locate indices of the nearest breakpoints for each query
        row_indices = torch.searchsorted(self.row_breakpoints, row_query_clamped, right=True) - 1
        row_indices = row_indices.clamp(0, self.num_rows - 2)
        col_indices = torch.searchsorted(self.col_breakpoints, col_query_clamped, right=True) - 1
        col_indices = col_indices.clamp(0, self.num_cols - 2)

        # Gather corner breakpoints and values for interpolation
        x0 = self.row_breakpoints[row_indices]
        x1 = self.row_breakpoints[row_indices + 1]
        y0 = self.col_breakpoints[col_indices]
        y1 = self.col_breakpoints[col_indices + 1]

        f00 = self.data[row_indices, col_indices]
        f01 = self.data[row_indices, col_indices + 1]
        f10 = self.data[row_indices + 1, col_indices]
        f11 = self.data[row_indices + 1, col_indices + 1]

        # Perform bilinear interpolation
        return self._bilinear_interpolate(row_query_clamped, col_query_clamped, x0, y0, x1, y1, f00, f01, f10, f11)


# Example Usage
# Define breakpoints and values for the table
# row_breakpoints = [0.0, 1.0, 2.0]
# col_breakpoints = [0.0, 10.0, 20.0]
# data = [[10.0, 20.0, 30.0],
#         [15.0, 25.0, 35.0],
#         [20.0, 30.0, 40.0]]

# # Instantiate the table
# table_2d = Table2D(data, row_breakpoints, col_breakpoints, device='cpu')

# # Query the table
# row_query = torch.tensor([0.5, 1.5, 2.5])
# col_query = torch.tensor([5.0, 15.0, 25.0])
# print(table_2d.get_value(row_query, col_query))
