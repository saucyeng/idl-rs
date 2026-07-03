//! FRB wrapper for `idl_rs::tracks` (visit detection) and `idl_rs::gps`
//! (authoring polyline). Track reference polylines cross in as freezed-free
//! args; visit windows cross back as mirrored structs. GPS never crosses for
//! detection — the engine reads it from the handle. `gps_track` exists so the
//! app can author a Track's reference polyline from a session handle.

use flutter_rust_bridge::frb;

use crate::frb_generated::RustOpaque;
use idl_rs::gps::{build_gps_track, GpsFix};
use idl_rs::tracks::{detect_visits as core_detect_visits, TrackRef, VisitParams};
pub use idl_rs::session::handle::SessionHandle;
pub use idl_rs::tracks::VisitWindow;

/// FFI GPS point (raw channel scale, degrees × 1e7). `timestamp_ms` is ignored
/// by detection but carried so the same struct serves `gps_track` authoring.
pub struct GpsFixArg {
    pub timestamp_ms: i64,
    pub lat: f64,
    pub lon: f64,
}

impl GpsFixArg {
    fn into_core(self) -> GpsFix {
        GpsFix { timestamp_ms: self.timestamp_ms, lat: self.lat, lon: self.lon }
    }
    fn from_core(f: GpsFix) -> Self {
        GpsFixArg { timestamp_ms: f.timestamp_ms, lat: f.lat, lon: f.lon }
    }
}

/// One reference track to match against.
pub struct TrackArg {
    pub track_id: String,
    pub polyline: Vec<GpsFixArg>,
}

#[frb(mirror(VisitWindow))]
pub struct _VisitWindow {
    pub track_id: String,
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
}

/// Detect ordered track visits for the retained session handle, using engine
/// default tuning. Identity (`visit_id`) is assigned app-side.
#[frb]
pub fn detect_visits(handle: RustOpaque<SessionHandle>, tracks: Vec<TrackArg>) -> Vec<VisitWindow> {
    let refs: Vec<TrackRef> = tracks
        .into_iter()
        .map(|t| TrackRef {
            track_id: t.track_id,
            polyline: t.polyline.into_iter().map(GpsFixArg::into_core).collect(),
        })
        .collect();
    core_detect_visits(&handle, &refs, VisitParams::default())
}

/// Build the session's GPS polyline (for authoring a Track reference polyline).
#[frb]
pub fn gps_track(handle: RustOpaque<SessionHandle>) -> Vec<GpsFixArg> {
    build_gps_track(&handle).into_iter().map(GpsFixArg::from_core).collect()
}

/// One value per GPS fix (in `gps_track` order): `channel_id` resampled to each
/// fix's instant. `NaN` where the channel is absent or the fix is outside its
/// span. Used to colour the GPS polyline by a channel value.
#[frb]
pub fn gps_channel_values(handle: RustOpaque<SessionHandle>, channel_id: String) -> Vec<f64> {
    handle.gps_channel_values(&channel_id)
}
