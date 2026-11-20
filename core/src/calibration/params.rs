//! A bike's parameters by name (whole-bike sim spec §12): one vocabulary for
//! the setup sheet the rider logs and the numbers the simulator uses. A key
//! is dotted, `family.name` or `family.end.name` (`fork.psi`,
//! `wheels.front.rim_kg`); the bike-log's sheet and notes store values under
//! the same keys. Labels and units here are what every screen shows.
//!
//! The mass table ([`mass_table`]) turns the masses into what the
//! simulator needs: each wheel's rolling mass and inertia, the unsprung
//! masses and the sprung mass.

use std::collections::BTreeMap;

use serde::Serialize;

/// Values by key: numbers for numeric parameters, strings for text ones.
pub type Params = BTreeMap<String, serde_json::Value>;

/// A group of parameters, as screens fold them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// Front and rear tyres.
    Tyres,
    /// Hubs, spokes and rims.
    Wheels,
    /// The fork.
    Fork,
    /// The shock.
    Shock,
    /// Frame settings.
    Frame,
    /// Bars and stem.
    Cockpit,
    /// Component and whole-bike masses.
    Masses,
    /// The rider.
    Rider,
}

/// What a parameter holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    /// A number in [`ParamSpec::unit`].
    Number,
    /// Free text (a model name, a note).
    Text,
}

/// One parameter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ParamSpec {
    /// Dotted key.
    pub key: &'static str,
    /// Label, sentence case.
    pub label: &'static str,
    /// Unit of a number ("psi", "kg", "clicks"); "" for text or a count.
    pub unit: &'static str,
    /// Family.
    pub family: Family,
    /// Number or text.
    pub kind: ParamKind,
    /// A setup change the rider logs (pressures, clicks, tokens): on a real
    /// bike, changing it is a log entry. Otherwise it describes the bike
    /// (masses, sizes) and is edited in place.
    pub logged: bool,
}

const fn n(key: &'static str, label: &'static str, unit: &'static str, family: Family, logged: bool) -> ParamSpec {
    ParamSpec { key, label, unit, family, kind: ParamKind::Number, logged }
}

const fn t(key: &'static str, label: &'static str, family: Family) -> ParamSpec {
    ParamSpec { key, label, unit: "", family, kind: ParamKind::Text, logged: true }
}

use Family::*;

/// Every parameter, in display order. The setup sheet's keys (bike-log
/// spec §5) come first in each family, unchanged.
pub const PARAMS: &[ParamSpec] = &[
    t("front_tyre.model", "Front tyre", Tyres),
    n("front_tyre.psi", "Front pressure", "psi", Tyres, true),
    t("front_tyre.notes", "Front tyre notes", Tyres),
    t("rear_tyre.model", "Rear tyre", Tyres),
    n("rear_tyre.psi", "Rear pressure", "psi", Tyres, true),
    t("rear_tyre.notes", "Rear tyre notes", Tyres),
    n("front_tyre.mass_kg", "Front tyre mass", "kg", Tyres, false),
    n("front_tyre.insert_kg", "Front insert mass", "kg", Tyres, false),
    n("front_tyre.sealant_kg", "Front sealant", "kg", Tyres, false),
    n("rear_tyre.mass_kg", "Rear tyre mass", "kg", Tyres, false),
    n("rear_tyre.insert_kg", "Rear insert mass", "kg", Tyres, false),
    n("rear_tyre.sealant_kg", "Rear sealant", "kg", Tyres, false),
    t("wheels.notes", "Wheel notes", Wheels),
    n("wheels.front.hub_kg", "Front hub", "kg", Wheels, false),
    n("wheels.front.spokes_kg", "Front spokes and nipples", "kg", Wheels, false),
    n("wheels.front.rim_kg", "Front rim", "kg", Wheels, false),
    n("wheels.rear.hub_kg", "Rear hub", "kg", Wheels, false),
    n("wheels.rear.spokes_kg", "Rear spokes and nipples", "kg", Wheels, false),
    n("wheels.rear.rim_kg", "Rear rim", "kg", Wheels, false),
    n("wheels.rim_radius_m", "Rim radius (to the rim's mass)", "m", Wheels, false),
    n("wheels.tyre_radius_m", "Tyre radius (to the tyre's mass)", "m", Wheels, false),
    t("fork.model", "Fork", Fork),
    n("fork.travel_mm", "Travel", "mm", Fork, true),
    n("fork.psi", "Pressure", "psi", Fork, true),
    n("fork.volume_reducers", "Volume reducers", "", Fork, true),
    n("fork.lsc", "LSC", "clicks", Fork, true),
    n("fork.hsc", "HSC", "clicks", Fork, true),
    n("fork.lsr", "LSR", "clicks", Fork, true),
    n("fork.hsr", "HSR", "clicks", Fork, true),
    t("shock.model", "Shock", Shock),
    n("shock.travel_mm", "Shock travel", "mm", Shock, true),
    n("shock.wheel_travel_mm", "Wheel travel", "mm", Shock, true),
    n("shock.psi", "Pressure", "psi", Shock, true),
    n("shock.spring_rate", "Spring rate", "lb/in", Shock, true),
    n("shock.volume_spacers", "Volume spacers", "", Shock, true),
    n("shock.lsc", "LSC", "clicks", Shock, true),
    n("shock.hsc", "HSC", "clicks", Shock, true),
    n("shock.lsr", "LSR", "clicks", Shock, true),
    n("shock.hsr", "HSR", "clicks", Shock, true),
    t("frame.flip_chip", "Flip chip", Frame),
    n("cockpit.bar_height_mm", "Bar height", "mm", Cockpit, true),
    n("mass.bike_kg", "Whole bike, weighed", "kg", Masses, false),
    n("mass.frame_kg", "Frame without shock", "kg", Masses, false),
    n("mass.fork_kg", "Fork", "kg", Masses, false),
    n("mass.shock_kg", "Shock", "kg", Masses, false),
    n("rider.mass_kg", "Rider in kit", "kg", Rider, false),
    n("rider.height_m", "Rider height", "m", Rider, false),
];

/// The spec for `key`, if it is in the vocabulary.
pub fn spec(key: &str) -> Option<&'static ParamSpec> {
    PARAMS.iter().find(|p| p.key == key)
}

/// `base` with `over`'s values on top: a bike's own parameters with its
/// log's running sheet applied.
pub fn overlay(base: &Params, over: &Params) -> Params {
    let mut out = base.clone();
    for (k, v) in over {
        out.insert(k.clone(), v.clone());
    }
    out
}

/// A number at `key`, if set and numeric.
pub fn number(p: &Params, key: &str) -> Option<f64> {
    p.get(key).and_then(serde_json::Value::as_f64)
}

/// Where a rim's mass sits when the parameters don't say, m: a 29" rim's
/// bead seat is 311 mm from the axle, the rim's mass a little inside it.
pub const DEFAULT_RIM_RADIUS_M: f64 = 0.300;

/// Where a tyre's (and insert's and sealant's) mass sits when the
/// parameters don't say, m: about half way up a 2.5" 29er tyre's sidewall.
pub const DEFAULT_TYRE_RADIUS_M: f64 = 0.345;

/// The share of a frame's mass in its rear triangle (chainstays and
/// seatstays) when nothing better is known: a placeholder.
pub const REAR_TRIANGLE_SHARE: f64 = 0.35;

/// The share of a fork's mass below the seals (lowers, axle, brake mount):
/// a placeholder.
pub const FORK_LOWERS_SHARE: f64 = 0.45;

/// One wheel as it rolls.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct WheelMass {
    /// Hub, spokes, rim, tyre, insert and sealant, kg. 0 when nothing is entered.
    pub mass_kg: f64,
    /// Spin inertia about the axle, kg·m²: rim, tyre, insert and sealant
    /// as rings at their radii, spokes as rods from the hub to the rim,
    /// the hub as a point on the axle.
    pub inertia_kgm2: f64,
    /// Whether every part was entered (else the missing parts count as 0).
    pub complete: bool,
}

/// What the simulator needs from the masses.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct MassTable {
    /// Front wheel.
    pub front: WheelMass,
    /// Rear wheel.
    pub rear: WheelMass,
    /// Front unsprung mass: the wheel and the fork's lowers, kg.
    pub front_unsprung_kg: f64,
    /// Rear unsprung mass: the wheel and half the rear triangle (it pivots
    /// about the frame, so half its mass moves with the wheel), kg.
    pub rear_unsprung_kg: f64,
    /// The bike's sprung mass: the weighed bike less the unsprung masses, kg;
    /// None until the bike is weighed.
    pub bike_sprung_kg: Option<f64>,
    /// The bike's weight not in any entered part (drivetrain, brakes,
    /// cockpit, seat post...), kg; None until weighed.
    pub rest_kg: Option<f64>,
    /// Rider in kit, kg.
    pub rider_kg: Option<f64>,
}

fn wheel(p: &Params, end: &str, tyre: &str) -> WheelMass {
    let part = |k: String| number(p, &k);
    let parts = [
        part(format!("wheels.{end}.hub_kg")),
        part(format!("wheels.{end}.spokes_kg")),
        part(format!("wheels.{end}.rim_kg")),
        part(format!("{tyre}.mass_kg")),
        part(format!("{tyre}.insert_kg")),
        part(format!("{tyre}.sealant_kg")),
    ];
    let [hub, spokes, rim, tyre_m, insert, sealant] = parts.map(|x| x.unwrap_or(0.0));
    let r_rim = number(p, "wheels.rim_radius_m").unwrap_or(DEFAULT_RIM_RADIUS_M);
    let r_tyre = number(p, "wheels.tyre_radius_m").unwrap_or(DEFAULT_TYRE_RADIUS_M);
    // a hub flange radius of about 30 mm; spokes as uniform rods out to the rim
    let r_hub = 0.03;
    let spoke_i = spokes * (r_hub * r_hub + r_hub * r_rim + r_rim * r_rim) / 3.0;
    WheelMass {
        mass_kg: hub + spokes + rim + tyre_m + insert + sealant,
        inertia_kgm2: rim * r_rim * r_rim + (tyre_m + insert + sealant) * r_tyre * r_tyre + spoke_i,
        complete: parts.iter().all(Option::is_some),
    }
}

/// The mass table from `p` (whole-bike sim spec §4.2). Missing masses count
/// as 0, and the frame's and fork's splits use [`REAR_TRIANGLE_SHARE`] and
/// [`FORK_LOWERS_SHARE`] until they are measured.
pub fn mass_table(p: &Params) -> MassTable {
    let front = wheel(p, "front", "front_tyre");
    let rear = wheel(p, "rear", "rear_tyre");
    let frame = number(p, "mass.frame_kg").unwrap_or(0.0);
    let fork = number(p, "mass.fork_kg").unwrap_or(0.0);
    let shock = number(p, "mass.shock_kg").unwrap_or(0.0);
    let front_unsprung_kg = front.mass_kg + FORK_LOWERS_SHARE * fork;
    let rear_unsprung_kg = rear.mass_kg + 0.5 * REAR_TRIANGLE_SHARE * frame;
    let bike = number(p, "mass.bike_kg");
    MassTable {
        front,
        rear,
        front_unsprung_kg,
        rear_unsprung_kg,
        bike_sprung_kg: bike.map(|b| b - front_unsprung_kg - rear_unsprung_kg),
        rest_kg: bike.map(|b| b - front.mass_kg - rear.mass_kg - frame - fork - shock),
        rider_kg: number(p, "rider.mass_kg"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn params_every_key_is_unique_and_dotted() {
        // Arrange / Act
        let mut keys: Vec<&str> = PARAMS.iter().map(|p| p.key).collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();

        // Assert
        assert_eq!(keys.len(), before);
        assert!(PARAMS.iter().all(|p| p.key.contains('.') && !p.key.starts_with('.')));
    }

    #[test]
    fn params_the_setup_sheets_keys_are_all_in_the_vocabulary() {
        // Arrange: the bike-log sheet's keys (app/src/routes/pages/Device/sheetFields.ts)
        let sheet = [
            "front_tyre.model", "front_tyre.psi", "front_tyre.notes", "rear_tyre.model", "rear_tyre.psi", "rear_tyre.notes", "wheels.notes",
            "fork.model", "fork.travel_mm", "fork.psi", "fork.volume_reducers", "fork.lsc", "fork.hsc", "fork.lsr", "fork.hsr",
            "shock.model", "shock.travel_mm", "shock.wheel_travel_mm", "shock.psi", "shock.spring_rate", "shock.volume_spacers",
            "shock.lsc", "shock.hsc", "shock.lsr", "shock.hsr", "frame.flip_chip", "cockpit.bar_height_mm",
        ];

        // Act / Assert
        for k in sheet {
            assert!(spec(k).is_some_and(|s| s.logged), "{k}");
        }
    }

    #[test]
    fn overlay_the_log_wins_over_the_bikes_own_values() {
        // Arrange
        let base: Params = [("fork.psi".to_string(), json!(75)), ("mass.fork_kg".to_string(), json!(2.2))].into();
        let over: Params = [("fork.psi".to_string(), json!(80))].into();

        // Act
        let p = overlay(&base, &over);

        // Assert
        assert_eq!(number(&p, "fork.psi"), Some(80.0));
        assert_eq!(number(&p, "mass.fork_kg"), Some(2.2));
    }

    #[test]
    fn mass_table_a_29er_wheel_rolls_about_2_8_kg_and_0_28_kg_m2() {
        // Arrange: DT 350 front, 32 Competition spokes, 620 g rim, Assegai DH, insert, sealant
        let p: Params = [
            ("wheels.front.hub_kg", 0.144),
            ("wheels.front.spokes_kg", 0.202),
            ("wheels.front.rim_kg", 0.62),
            ("front_tyre.mass_kg", 1.48),
            ("front_tyre.insert_kg", 0.26),
            ("front_tyre.sealant_kg", 0.1),
            ("mass.bike_kg", 18.1),
            ("mass.frame_kg", 3.63),
            ("mass.fork_kg", 2.2),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();

        // Act
        let m = mass_table(&p);

        // Assert
        assert!((m.front.mass_kg - 2.806).abs() < 1e-9);
        assert!(m.front.complete && !m.rear.complete);
        assert!(m.front.inertia_kgm2 > 0.25 && m.front.inertia_kgm2 < 0.31, "{}", m.front.inertia_kgm2);
        assert!((m.front_unsprung_kg - (2.806 + 0.45 * 2.2)).abs() < 1e-9);
        assert!((m.rest_kg.unwrap() - (18.1 - 2.806 - 3.63 - 2.2)).abs() < 1e-9);
    }
}
