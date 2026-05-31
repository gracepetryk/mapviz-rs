//! Coordinate systems.
//!
//! Conversions between systems are explicit and typed — no raw `[f64; 3]`
//! shuffling. This module currently defines the geographic and earth-centered
//! types; local ENU tangent planes and normalized device coordinates follow as
//! the projection and camera work lands.

use core::fmt;

/// A geographic position in the WGS84 datum: longitude/latitude in degrees,
/// altitude in meters above the ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wgs84 {
    /// Longitude in degrees, east-positive, in `[-180, 180]`.
    pub lon_deg: f64,
    /// Latitude in degrees, north-positive, in `[-90, 90]`.
    pub lat_deg: f64,
    /// Altitude in meters above the WGS84 ellipsoid.
    pub alt_m: f64,
}

impl fmt::Display for Wgs84 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lat, ns) = if self.lat_deg >= 0.0 {
            (self.lat_deg, 'N')
        } else {
            (-self.lat_deg, 'S')
        };
        let (lon, ew) = if self.lon_deg >= 0.0 {
            (self.lon_deg, 'E')
        } else {
            (-self.lon_deg, 'W')
        };
        write!(f, "{lat:.5}°{ns} {lon:.5}°{ew} {:.0}m", self.alt_m)
    }
}

/// An earth-centered, earth-fixed position in meters. Used for 3D globe math
/// where geographic coordinates are awkward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ecef {
    /// X axis, meters, toward the prime meridian at the equator.
    pub x: f64,
    /// Y axis, meters, toward 90°E at the equator.
    pub y: f64,
    /// Z axis, meters, toward the north pole.
    pub z: f64,
}

impl fmt::Display for Ecef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ECEF({:.1}, {:.1}, {:.1})m", self.x, self.y, self.z)
    }
}
