from typing import Optional

import torch
import math

class UnitConversions:
    _instance = None
    # Explicitly declare all properties for Intellisense
    RAD_TO_DEG: Optional[torch.Tensor] = None
    DEG_TO_RAD: Optional[torch.Tensor] = None
    HP_TO_FTLB_SEC: Optional[torch.Tensor] = None
    PSF_TO_INHG: Optional[torch.Tensor] = None
    PSF_TO_PA: Optional[torch.Tensor] = None
    FT_TO_M: Optional[torch.Tensor] = None
    KTS_TO_FPS: Optional[torch.Tensor] = None
    FPS_TO_KTS: Optional[torch.Tensor] = None
    INCH_TO_FT: Optional[torch.Tensor] = None
    M3_TO_FT3: Optional[torch.Tensor] = None
    IN3_TO_M3: Optional[torch.Tensor] = None
    INHG_TO_PA: Optional[torch.Tensor] = None
    SLUG_TO_LB: Optional[torch.Tensor] = None
    LB_TO_SLUG: Optional[torch.Tensor] = None
    KG_TO_LB: Optional[torch.Tensor] = None
    KG_TO_SLUG: Optional[torch.Tensor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnitConversions, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, device: torch.device):
        """Initialize the constants on the specified device."""
        instance = cls._instance or cls()

        # Conversion factors
        instance.RAD_TO_DEG = torch.tensor(180.0 / math.pi, dtype=torch.float64, device=device)
        instance.DEG_TO_RAD = torch.tensor(math.pi / 180.0, dtype=torch.float64, device=device)
        instance.HP_TO_FTLB_SEC = torch.tensor(550.0, dtype=torch.float64, device=device)
        instance.PSF_TO_INHG = torch.tensor(0.014138, dtype=torch.float64, device=device)
        instance.PSF_TO_PA = torch.tensor(47.88, dtype=torch.float64, device=device)
        instance.FT_TO_M = torch.tensor(0.3048, dtype=torch.float64, device=device)
        instance.KTS_TO_FPS = 1852.0 / (3600 * instance.FT_TO_M)
        instance.FPS_TO_KTS = 1.0 / instance.KTS_TO_FPS
        instance.INCH_TO_FT = torch.tensor(1.0 / 12.0, dtype=torch.float64, device=device)
        instance.M3_TO_FT3 = 1.0 / (instance.FT_TO_M ** 3)
        instance.IN3_TO_M3 = (instance.INCH_TO_FT ** 3) / instance.M3_TO_FT3
        instance.INHG_TO_PA = torch.tensor(3386.38, dtype=torch.float64, device=device)

        # Mass conversion factors
        instance.SLUG_TO_LB = torch.tensor(32.174049, dtype=torch.float64, device=device)
        instance.LB_TO_SLUG = 1.0 / instance.SLUG_TO_LB
        instance.KG_TO_LB = torch.tensor(2.20462, dtype=torch.float64, device=device)
        instance.KG_TO_SLUG = torch.tensor(0.06852168, dtype=torch.float64, device=device)
        instance._K_TO_R = torch.tensor(1.8, dtype=torch.float64, device=device)
        return instance

    def KelvinToRankine(self, kelvin: torch.Tensor):
        return kelvin * self._K_TO_R
    
    @classmethod
    def get_instance(cls):
        """Return the singleton instance. Ensure initialize() was called first."""
        if cls._instance is None:
            raise RuntimeError("Constants not initialized. Call Constants.initialize(device) first.")
        return cls._instance