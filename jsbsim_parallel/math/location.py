from typing import Optional
from enum import Enum

import torch

from jsbsim_parallel.models.unit_conversions import UnitConversions

class CoordinateIndex(Enum):
    X: int = 1
    Y: int = 2
    Z: int = 3


class Location:
    '''
    Location holds an arbitrary location in the Earth centered Earth fixed
    reference frame (ECEF). The coordinate frame ECEF has its center in the
    middle of the earth. The X-axis points from the center of the Earth towards
    a location with zero latitude and longitude on the Earth surface. The Y-axis
    points from the center of the Earth towards a location with zero latitude
    and 90 deg East longitude on the Earth surface. The Z-axis points from the
    Earth center to the geographic north pole.

    This class provides access functions to set and get the location as either
    the simple X, Y and Z values in ft or longitude/latitude and the radial
    distance of the location from the Earth center.

    It is common to associate a parent frame with a location. This frame is
    usually called the local horizontal frame or simply the local frame. It is
    also called the NED frame (North, East, Down), as well as the Navigation
    frame. This frame has its X/Y plane parallel to the surface of the Earth.
    The X-axis points towards north, the Y-axis points east and the Z-axis
    is normal to the reference spheroid (WGS84 for Earth).

    Since the local frame is determined by the location (and NOT by the
    orientation of the vehicle IN any frame), this class also provides the
    rotation matrices required to transform from the Earth centered (ECEF) frame
    to the local horizontal frame and back. This class "owns" the
    transformations that go from the ECEF frame to and from the local frame.
    Again, this is because the ECEF, and local frames do not involve the actual
    orientation of the vehicle - only the location on the Earth surface. There
    are conversion functions for conversion of position vectors given in the one
    frame to positions in the other frame.

    The Earth centered reference frame is NOT an inertial frame since it rotates
    with the Earth.

    The cartesian coordinates (X,Y,Z) in the Earth centered frame are the master
    values. All other values are computed from these master values and are
    cached as long as the location is changed by access through a non-const
    member function. Values are cached to improve performance. It is best
    practice to work with a natural set of master values. Other parameters that
    are derived from these master values are calculated only when needed, and IF
    they are needed and calculated, then they are cached (stored and remembered)
    so they do not need to be re-calculated until the master values they are
    derived from are themselves changed (and become stale).

    Accuracy and round off

    Given,

    - that we model a vehicle near the Earth
    - that the Earth surface average radius is about 2*10^7, ft
    - that we use double values for the representation of the location

    we have an accuracy of about

    1e-16*2e7ft/1 = 2e-9 ft

    left. This should be sufficient for our needs. Note that this is the same
    relative accuracy we would have when we compute directly with
    lon/lat/radius. For the radius value this is clear. For the lon/lat pair
    this is easy to see. Take for example KSFO located at about 37.61 deg north
    122.35 deg west, which corresponds to 0.65642 rad north and 2.13541 rad
    west. Both values are of magnitude of about 1. But 1 ft corresponds to about
    1/(2e7*2*pi) = 7.9577e-09 rad. So the left accuracy with this representation
    is also about 1*1e-16/7.9577e-09 = 1.2566e-08 which is of the same magnitude
    as the representation chosen here.

    The advantage of this representation is that it is a linear space without
    singularities. The singularities are the north and south pole and most
    notably the non-steady jump at -pi to pi. It is harder to track this jump
    correctly especially when we need to work with error norms and derivatives
    of the equations of motion within the time-stepping code. Also, the rate of
    change is of the same magnitude for all components in this representation
    which is an advantage for numerical stability in implicit time-stepping.

    Note: Both GEOCENTRIC and GEODETIC latitudes can be used. In order to get
    best matching relative to a map, geodetic latitude must be used.

    @see Stevens and Lewis, "Aircraft Control and Simulation", Second edition
    @see W. C. Durham "Aircraft Dynamics & Control", section 2.2

    @author Mathias Froehlich
  '''
    #mCacheValid (False)..skip for now
    def __init__(self, 
                 latitude: Optional[torch.Tensor]=None, 
                 longitude: Optional[torch.Tensor] = None,
                 radius: Optional[torch.Tensor] = None, 
                 ecef: Optional[torch.Tensor]=None,
                 *,
                 device: torch.device, #todo...
                 batch_size: Optional[torch.Size] = None):
            
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        size = self.size
        self.one = torch.ones(*size, 1, dtype=torch.float64, device=device)
        # Note: All locations need to be initialized in the same way for the entire batch.
        if latitude is None and longitude is None and radius is None and ecef is None:
            # The coordinates in the earth centered frame. This is the master copy.
            # The coordinate frame has its center in the middle of the earth.
            # Its x-axis points from the center of the earth towards a
            # location with zero latitude and longitude on the earths
            # surface. The y-axis points from the center of the earth towards a
            # location with zero latitude and 90deg longitude on the earths
            # surface. The z-axis points from the earths center to the
            # geographic north pole.
            # @see W. C. Durham "Aircraft Dynamics & Control", section 2.2
            self.mECLoc = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device).expand(*size, 3)
            self.e2 = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.c = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.a = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.ec = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.ec2 = torch.ones(*size, 1, dtype=torch.float64, device=device)

            self.mLon = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mLat = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mRadius = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mGeodLat = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.GeodeticAltitude = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mTl2ec = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
            self.mTec2l = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        elif ecef is not None:
            self.mECLoc = ecef
            self.e2 = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.c = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.a = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.ec = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.ec2 = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.mLon = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mLat = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mRadius = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mGeodLat = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.GeodeticAltitude = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mTl2ec = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
            self.mTec2l = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)

        else: #lat, lon radius not None
            self.e2 = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.c = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.a = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.ec = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.ec2 = torch.ones(*size, 1, dtype=torch.float64, device=device)
            self.mLon = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mLat = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mRadius = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mGeodLat = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.GeodeticAltitude = torch.zeros(*size, 1, dtype=torch.float64, device=device)
            self.mTl2ec = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
            self.mTec2l = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)

            sinlat = torch.sin(latitude)
            coslat = torch.cos(latitude)
            sinlon = torch.sin(longitude)
            coslon = torch.cos(longitude)
            
            self.mECLoc = torch.cat((radius * coslat * coslon,
                                    radius * coslat * sinlon,
                                    radius * sinlat), dim=-1)
        self._cache_valid = torch.zeros(*size, 1, dtype=torch.bool, device=device)

    def GetRadius(self):
        self._compute_derived()
        return self.mRadius

    def GetGeodLatitudeDeg(self):
        #assert self._ellipseSet
        self._compute_derived()
        units = UnitConversions.get_instance()
        return units.RAD_TO_DEG * self.mGeodLat

    def GetLongitudeDeg(self):
        pass

    def GetSeaLevelRadius(self):
        #assert mEllipseSet...
        self._compute_derived()
        coslat = torch.cos(self.mLat)
        return self.a * self.ec / torch.sqrt(self.one - coslat * coslat)

    def _compute_derived(self):
        #check _cache_valid
        self._compute_derived_unconditional()
        
    
    def _compute_derived_unconditional(self):
        #todo: Don't allocate
        self._cache_valid = torch.ones(*self.size, 1, dtype=torch.bool, device=self.device)
        indices = [CoordinateIndex.X.value, CoordinateIndex.Y.value]
        self.mRadius = torch.norm(self.mECLoc, p=2, dim=-1, keepdim=True) #magnitude

        rxy = torch.norm(self.mECLoc[..., indices], p=2, dim=-1, keepdim=True) #distance to z-axis (poles)
        zero_indices = rxy == 0
        sinlon = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        coslon = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        sinlon[zero_indices] = 0.0 #todo: scatter
        coslon[zero_indices] = 1.0 #todo: scatter



#   FGColumnVector3 mECLoc;

#   /** The cached lon/lat/radius values. */
#   mutable double mLon;
#   mutable double mLat;
#   mutable double mRadius;
#   mutable double mGeodLat;
#   mutable double GeodeticAltitude;

#   /** The cached rotation matrices from and to the associated frames. */
#   mutable FGMatrix33 mTl2ec;
#   mutable FGMatrix33 mTec2l;

#   /* Terms for geodetic latitude calculation. Values are from WGS84 model */
#   double a;    // Earth semimajor axis in feet
#   double e2;   // Earth eccentricity squared
#   double c;
#   double ec;
#   double ec2;

#   /** A data validity flag.
#       This class implements caching of the derived values like the
#       orthogonal rotation matrices or the lon/lat/radius values. For caching we
#       carry a flag which signals if the values are valid or not.
#       The C++ keyword "mutable" tells the compiler that the data member is
#       allowed to change during a const member function. */
#   mutable bool mCacheValid;
#   // Flag that checks that geodetic methods are called after SetEllipse() has
#   // been called.
#   bool mEllipseSet = false;
        pass

    def set_longitude(self, longitude: torch.Tensor):
        '''
        args:
            longitude: -pi <= longitude <= pi
        '''
        pass

    def set_latitude(self, latitude: torch.Tensor):
        '''
        Sets the geocentric latitude
        args:
            latitude: -pi/2 <= latitude <= pi/2
        '''
        pass

    def set_radius(self, radius: torch.Tensor):
        '''
        Sets the radius in feet.
        args:
            radius: radius >= 0
        '''
        pass

    def set_position(self,
                    latitude: Optional[torch.Tensor]=None, 
                    longitude: Optional[torch.Tensor] = None,
                    radius: Optional[torch.Tensor] = None):
        '''
        args:
        latitude: geocentric latitude
        '''
        pass

    def set_position_geodetic(self,
                            latitude: Optional[torch.Tensor]=None, 
                            longitude: Optional[torch.Tensor] = None,
                            height: Optional[torch.Tensor] = None):
        '''
        args:
            longituce longitude in radians
            latitude GEODETIC latitude in radians
            height distance above the reference ellipsoid to vehicle in feet
        '''
        pass
    def set_ellipse(self, semi_major: torch.Tensor, semi_minor: torch.Tensor):
        '''
        args:
            semi_major: semi-major axis in feet
            semi_minor: semi-minor axis in feet
        '''
        
        self._cache_valid = False
        self._ellipse_set = True
        self.a = semi_major
        self.ec = semi_minor / semi_major
        self.ec2 = self.ec * self.ec
        self.e2 = 1.0 - self.ec2
        self.c = self.a * self.e2