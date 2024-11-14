from enum import IntEnum
from typing import Tuple, List, Optional


import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.property_value import PropertyValue

class Axis(IntEnum):
    Row = 0
    Column = 1
    Table = 2


class Table2D:
    def __init__(self, N: int = None, data = None, el: Element = None, prefix="", *, 
                 device=torch.device('cpu')):
        """
        Initialize a 2D lookup table with interpolation.

        :param data: 2D tensor of values in the table (size 2xN).
        : First dimension is 2
        : Along data[0] are the breakpoints corresponding to data points (size N).
        : Along data[1] are the data values (size N).
        :param device: Device on which to store tensors (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.lookupProperty: List[PropertyValue] = [None] * 3 #todo: batch_size
        if N:
            self.N = N
            if data is None:
                self.data = torch.zeros(N, 2, dtype=torch.float32, device=device)
            else:
                self.data = data
        elif el:
            self.InitializeFromXml(el, prefix)

    def InitializeFromXml(self, el: Element, prefix: str):
        """
        Initialize the table from an XML element.
        
        :param el: XML element containing table data
        :param prefix: Optional prefix for handling special characters in breakpoints
        """

        self.Name = el.GetAttributeValue("name")
        brkpt_string = None
        # Check for the table type
        call_type = el.GetAttributeValue("type")
        if call_type == "internal":
            raise ValueError("Only non-internal tables are supported in this initializer.")
        elif len(call_type) > 0:
            raise ValueError(f"Unknown table type: {call_type}")

        dimension = 0

        # Check for 2D independent variable
        axisElement = el.FindElement("independentVar")
        if axisElement:
#            raise ValueError("No independent variable found for 2D table.")
            while axisElement:
                # Handle breakpoint replacement if '#' is used
                property_string = axisElement.GetDataLine()
                if "#" in property_string and prefix.isdigit():
                    property_string = property_string.replace("#", prefix)

                node = PropertyValue(property_string, axisElement)
                lookup_axis = axisElement.GetAttributeValue("lookup")
                if lookup_axis == "row":
                    self.lookupProperty[Axis.Row] = node
                    dimension = max(dimension, 1)
                elif lookup_axis == "column":
                    self.lookupProperty[Axis.Column] = node
                    dimension = max(dimension, 2)
                elif lookup_axis == "table":
                    self.lookupProperty[Axis.Table] = node
                    dimension = max(dimension, 3)
                elif len(lookup_axis) > 0:
                    raise ValueError(f"Unknown lookup axis: {lookup_axis}")
                else: # assumed single dimension table, row lookup
                    self.lookupProperty[Axis.Row] = node
                    dimension = max(dimension, 1)
                axisElement = el.FindNextElement("independentVar")
        else:
            brkpt_string = el.GetAttributeValue("breakPoint")
            if len(brkpt_string) == 0:
                raise ValueError("No independentVars found.")
        # end lookup property code

        if not brkpt_string:
            if el.GetNumElements("tableData") > 1:
                raise ValueError("Multiple tableData elements found.")
            tableData = el.FindElement("tableData")

        if not tableData:
            raise ValueError("No tableData element found.")
        elif tableData.GetNumDataLines() == 0:
            raise ValueError("No data found in tableData element.")

        # Check that the lookup axes match the declared dimension of the table.
        if not brkpt_string:
            if dimension == 2:
                if not self.lookupProperty[Axis.Column]:
                    raise ValueError("Column lookup property not specified.")
            if el.GetNumElements("tableData") > 1:
                raise ValueError("Dim = 3, not supported.")
            tableData = el.FindElement("tableData")
        else:
            tableData = el
            dimension = 2
        
        # TODO: Check for illegal characters.
        self.nRows = tableData.GetNumDataLines() - 1 #remove header row
        # List to store parsed data
        parsed_data = []
        # Get all data lines from the XML element and parse each line
        for i in range(self.nRows + 1):
            line = tableData.GetDataLine(i)
            values = [float(x) for x in line.split()]
            if i == 0:
                values = [float("nan")] + values
            parsed_data.append(values)
        self.nCols = len(parsed_data[0])
        self.data = torch.tensor(parsed_data, dtype=torch.float64, device=self.device)

        # Sanity checks: lookup indices must be increasing monotonically
        nameel = el
        while nameel and nameel.GetAttributeValue("name") == "":
            nameel = nameel.GetParent()

        # check rows
        for r in range(1, self.nRows):
            if self.data[r, 0] <= self.data[r - 1, 0]:
                raise ValueError(f"Row {r} has non-increasing breakpoints.")
        
        # TODO: Check the table has been entirely populated
        # TODO: bind properties

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