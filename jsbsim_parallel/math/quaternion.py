from typing import List, Optional
import torch

class Quaternion:
    def __init__(self, *args, batch_size: torch.Size, device: torch.device =torch.device('cpu')):
        """
        Initializes Quaternion with support for batch operations.
        Each quaternion in the batch can be initialized the same way using Euler angles or from another quaternion.
        
        Args:
            *args: Initialization arguments, which could be:
                - Three Euler angles (phi, tht, psi)
                - An existing FGQuaternion object to copy from
            batch_size (int): Number of quaternions to initialize in a batch.
            device (str): Device to initialize tensors on ('cpu' or 'cuda').
        """
        self.device = device
        self.batch_size = batch_size

        if len(args) == 0:
            # Default constructor to identity quaternion
            self.data = torch.zeros(*batch_size, 4, dtype=torch.float64, device=device)
            self.data[..., 0] = 1.0  # Set w component to 1 for identity quaternion
            self.mCacheValid = False
            self._initialize_cache(device)
        elif len(args) == 1 and isinstance(args[0], Quaternion):
            # Copy constructor from another quaternion
            self.data = args[0].data.clone().expand(*batch_size, -1).to(device)
            self.mCacheValid = args[0].mCacheValid
            if self.mCacheValid:
                self.mT = args[0].mT.clone().expand(*batch_size, -1, -1).to(device)
                self.mTInv = args[0].mTInv.clone().expand(*batch_size, -1, -1).to(device)
                self.mEulerAngles = args[0].mEulerAngles.clone().expand(*batch_size, -1).to(device)
                self.mEulerSines = args[0].mEulerSines.clone().expand(*batch_size, -1).to(device)
                self.mEulerCosines = args[0].mEulerCosines.clone().expand(*batch_size, -1).to(device)
            else:
                self._initialize_cache(device)
        elif len(args) == 3:
            # Initialize with Euler angles (phi, tht, psi) for all in batch
            phi, tht, psi = args
            self.data = torch.zeros(*batch_size, 4, dtype=torch.float64, device=device)
            self.InitializeFromEulerAngles(phi, tht, psi)
            self.mCacheValid = False
            self._initialize_cache(device)
        else:
            raise ValueError("Invalid arguments for FGQuaternion initialization")

    def _initialize_cache(self, device):
        """Helper to initialize cache attributes for batched operations"""
        self.mT = torch.eye(3, dtype=torch.float64, device=device).repeat(*self.batch_size, 1, 1)
        self.mTInv = torch.eye(3, dtype=torch.float64, device=device).repeat(*self.batch_size, 1, 1)
        self.mEulerAngles = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.mEulerSines = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.mEulerCosines = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)

    def InitializeFromEulerAngles(self, phi, tht, psi):
        """Initialize quaternion based on Euler angles for batched inputs"""
        thtd2 = 0.5 * tht
        psid2 = 0.5 * psi
        phid2 = 0.5 * phi

        Sthtd2, Cthtd2 = torch.sin(thtd2), torch.cos(thtd2)
        Spsid2, Cpsid2 = torch.sin(psid2), torch.cos(psid2)
        Sphid2, Cphid2 = torch.sin(phid2), torch.cos(phid2)

        Cphid2Cthtd2 = Cphid2 * Cthtd2
        Cphid2Sthtd2 = Cphid2 * Sthtd2
        Sphid2Sthtd2 = Sphid2 * Sthtd2
        Sphid2Cthtd2 = Sphid2 * Cthtd2

        self.data[..., 0] = Cphid2Cthtd2 * Cpsid2 + Sphid2Sthtd2 * Spsid2
        self.data[..., 1] = Sphid2Cthtd2 * Cpsid2 - Cphid2Sthtd2 * Spsid2
        self.data[..., 2] = Cphid2Sthtd2 * Cpsid2 + Sphid2Cthtd2 * Spsid2
        self.data[..., 3] = Cphid2Cthtd2 * Spsid2 - Sphid2Sthtd2 * Cpsid2
        self.Normalize()

    def Normalize(self):
        """Normalize the quaternion batch-wise"""
        norm = self.data.norm(dim=-1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1.0, dtype=self.data.dtype, device=self.data.device), norm)
        self.data /= norm

    def GetQDot(self, PQR):
        """Calculate quaternion derivative for given angular rates PQR (batched)"""
        q_dot = torch.empty_like(self.data)
        q_dot[..., 0] = -0.5 * (self.data[..., 1] * PQR[..., 0] + self.data[..., 2] * PQR[..., 1] + self.data[..., 3] * PQR[..., 2])
        q_dot[..., 1] = 0.5 * (self.data[..., 0] * PQR[..., 0] - self.data[..., 3] * PQR[..., 1] + self.data[..., 2] * PQR[..., 2])
        q_dot[..., 2] = 0.5 * (self.data[..., 3] * PQR[..., 0] + self.data[..., 0] * PQR[..., 1] - self.data[..., 1] * PQR[..., 2])
        q_dot[..., 3] = 0.5 * (-self.data[..., 2] * PQR[..., 0] + self.data[..., 1] * PQR[..., 1] + self.data[..., 0] * PQR[..., 2])
        return q_dot

    def Inverse(self):
        """Return the inverse of each quaternion in the batch"""
        norm = self.data.norm(dim=-1, keepdim=True)
        inverse_data = self.data.clone()
        inverse_data[..., 1:] *= -1  # Negate the vector part for conjugation
        inverse_data /= norm
        return Quaternion(data=inverse_data, batch_size=inverse_data.shape[:-1], device=self.data.device)

    def Conjugate(self):
        """Return the conjugate of each quaternion in the batch"""
        conjugate_data = self.data.clone()
        conjugate_data[..., 1:] *= -1
        return Quaternion(data=conjugate_data, batch_size=conjugate_data.shape[:-1], device=self.data.device)

    def Magnitude(self):
        """Return the magnitude of each quaternion in the batch"""
        return self.data.norm(dim=-1)

    def SqrMagnitude(self):
        """Return the square of the magnitude of each quaternion in the batch"""
        return self.data.pow(2).sum(dim=-1)

    def __getitem__(self, idx):
        """Get quaternion element by index (1-based indexing), batch-aware"""
        return self.data[..., idx - 1]

    def __setitem__(self, idx, value):
        """Set quaternion element by index (1-based indexing), batch-aware"""
        self.data[..., idx - 1] = value
        self.mCacheValid = False

    def __eq__(self, other):
        """Batch-wise equality comparison"""
        return torch.allclose(self.data, other.data)

    def __add__(self, other):
        """Batch-wise addition of two quaternions"""
        return Quaternion(data=self.data + other.data, batch_size=self.batch_size, device=self.device)

    def __sub__(self, other):
        """Batch-wise subtraction of two quaternions"""
        return Quaternion(data=self.data - other.data, batch_size=self.batch_size, device=self.device)

    def __mul__(self, other):
        """Batch-wise quaternion multiplication or scalar multiplication"""
        if isinstance(other, Quaternion):
            q0 = self.data[..., 0] * other.data[..., 0] - self.data[..., 1] * other.data[..., 1] - self.data[..., 2] * other.data[..., 2] - self.data[..., 3] * other.data[..., 3]
            q1 = self.data[..., 0] * other.data[..., 1] + self.data[..., 1] * other.data[..., 0] + self.data[..., 2] * other.data[..., 3] - self.data[..., 3] * other.data[..., 2]
            q2 = self.data[..., 0] * other.data[..., 2] - self.data[..., 1] * other.data[..., 3] + self.data[..., 2] * other.data[..., 0] + self.data[..., 3] * other.data[..., 1]
            q3 = self.data[..., 0] * other.data[..., 3] + self.data[..., 1] * other.data[..., 2] - self.data[..., 2] * other.data[..., 1] + self.data[..., 3] * other.data[..., 0]
            return Quaternion(q0, q1, q2, q3, batch_size=self.batch_size, device=self.device)
        elif isinstance(other, (float, int)):
            return Quaternion(data=self.data * other, batch_size=self.batch_size, device=self.device)
        else:
            raise TypeError("Unsupported multiplication type for FGQuaternion")