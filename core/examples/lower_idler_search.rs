//! Where to put a 12T lower idler so the clutch never has to pay chain out
//! through travel, in the small and middle cogs.
//!
//! For each candidate centre on the front triangle (a grid round the BB), with
//! the lower span run over or under the idler: size the chain for that routing,
//! run every gear from top-out to full travel, and measure the clutch's work as
//! the most the cage must turn forward during a compression (the largest rise
//! of `cage_deg` from an earlier travel to a later one). With the clutch's free
//! play, a candidate with less than that in a gear never engages the clutch there.
//!
//! The GT Force linkage is the photo's; the drivetrain is `Drivetrain::typical`
//! (BB 12 mm behind and 100 mm below the main pivot, 32T, 10-52, a guessed
//! derailleur) until the real points are measured. The map's shape, not its
//! numbers, is what this shows. Keep-outs: the chainring (8 mm between pitch
//! circles), no lower than the chainring's bottom, and the chainstay as a 30 mm
//! deep bar from the main pivot to the Horst pivot, posed through travel. The
//! chain's own clearance to the stays is not checked (no contact surfaces yet).
//!
//! Usage: `cargo run -p idl-rs --example lower_idler_search -- [out.csv] [step_mm]`

use idl_rs::calibration::drivetrain::{Body, ChainSide, Drivetrain, Roller, Span, CHAIN_PITCH_M};
use idl_rs::calibration::linkage::{Linkage, LinkagePoints};
use idl_rs::sim::bike::SimBike;
use idl_rs::sim::kinematics::gear_curves;
use idl_rs::sim::path::chain_fit;

/// Travel samples per gear.
const SAMPLES: usize = 17;
/// Gears whose clutch work is minimised: 0 = the 10T, 6 = the 24T.
const TARGET_GEARS: std::ops::RangeInclusive<usize> = 0..=6;
/// Gap from the idler's pitch circle to the chainring's, m.
const RING_GAP_M: f64 = 0.008;
/// The chainstay as a bar from the main pivot to the Horst pivot: half its
/// depth, m (a guess until the stay's outline is clicked on the photo).
const STAY_HALF_M: f64 = 0.015;
/// Tooth height above the pitch circle, m.
const TOOTH_M: f64 = 0.004;

/// Distance from `p` to the segment `a`–`b`, m.
fn to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
    ((p[0] - a[0] - t * dx).powi(2) + (p[1] - a[1] - t * dy).powi(2)).sqrt()
}

/// The closest the posed chainstay comes to `c` through travel, m.
fn stay_clearance(linkage: &Linkage, d: &Drivetrain, c: [f64; 2]) -> f64 {
    let bike = SimBike::new(linkage, d).unwrap();
    let (mp, h) = (linkage.points_m.main_pivot, linkage.points_m.horst);
    (0..=SAMPLES).map(|k| {
        let pose = bike.pose(linkage.rear_travel_m * k as f64 / SAMPLES as f64);
        to_segment(c, pose.chainstay_point(mp), pose.chainstay_point(h))
    }).fold(f64::MAX, f64::min)
}

fn gt_force() -> Linkage {
    Linkage {
        updated_at_ms: 0,
        points_m: LinkagePoints {
            front_axle: [1.28, 0.0],
            rear_axle: [0.0, 0.0],
            main_pivot: [0.43717, 0.08031],
            horst: [0.07805, -0.01296],
            ss_rocker: [0.37094, 0.18068],
            rocker_frame: [0.48793, 0.19457],
            shock_frame: Some([0.52277, 0.04682]),
            shock_rocker: Some([0.55827, 0.25971]),
            imu0: [0.47677, 0.36738],
            fork_pod: None,
            rear_pod: None,
            centre_of_mass: None,
        },
        wheelbase_m: 1.28,
        rear_travel_m: 0.16,
        front_travel_m: None,
        shock_eye_to_eye_m: None,
        source_image: None,
    }
}

/// The most the cage turns forward from an earlier travel to a later one, degrees.
fn max_rise(v: &[f64]) -> f64 {
    let mut lo = f64::MAX;
    let mut worst = 0.0f64;
    for &x in v {
        lo = lo.min(x);
        worst = worst.max(x - lo);
    }
    worst
}

/// Per gear: clutch work in cage degrees, or None where the loop does not close.
fn clutch_work(linkage: &Linkage, d: &Drivetrain) -> Option<(u32, Vec<Option<f64>>)> {
    let bike = SimBike::new(linkage, d).ok()?;
    let (lo, hi) = chain_fit(&bike, &|_| true)?;
    let mut b = bike.clone();
    b.drivetrain.chain_links = (((lo + hi) / 4.0).round() * 2.0) as u32;
    let per_gear = (0..d.cassette_teeth.len())
        .map(|g| {
            let c = gear_curves(&b, g, SAMPLES, &|_| true).ok()?;
            if !c.roller_touching.iter().all(|r| r.iter().all(|&t| t)) {
                return None;
            }
            Some(max_rise(&c.cage_deg))
        })
        .collect();
    Some((b.drivetrain.chain_links, per_gear))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "lower_idler_search.csv".into());
    let step = args.get(2).and_then(|s| s.parse::<f64>().ok()).unwrap_or(10.0) / 1000.0;

    let linkage = gt_force();
    let bb = [0.425, -0.02];
    let base = Drivetrain::typical(bb, 0);
    let ring_r = base.chainring_teeth as f64 * CHAIN_PITCH_M / std::f64::consts::TAU;
    let idler_r = 12.0 * CHAIN_PITCH_M / std::f64::consts::TAU;
    let teeth = base.cassette_teeth.clone();

    // without an idler
    let (links0, work0) = clutch_work(&linkage, &base).expect("the typical drivetrain fits");
    println!("no idler ({links0} links): cage forward on compression, deg, per cog:");
    for (g, w) in work0.iter().enumerate() {
        println!("  {:>2}T  {}", teeth[g], w.map_or("-".into(), |w| format!("{w:5.2}")));
    }

    let mut rows = Vec::new();
    let nx = (0.22 / step).round() as i32;
    let ny = (0.16 / step).round() as i32;
    for ix in 0..=nx {
        for iy in 0..=ny {
            let c = [bb[0] - 0.20 + ix as f64 * step, bb[1] - 0.10 + iy as f64 * step];
            let dist = ((c[0] - bb[0]).powi(2) + (c[1] - bb[1]).powi(2)).sqrt();
            if dist < ring_r + idler_r + RING_GAP_M || c[1] - idler_r < bb[1] - ring_r {
                continue;
            }
            if stay_clearance(&linkage, &base, c) < idler_r + TOOTH_M + STAY_HALF_M {
                continue;
            }
            for side in [ChainSide::Over, ChainSide::Under] {
                let mut d = base.clone();
                d.rollers.push(Roller { name: "Lower idler".into(), body: Body::FrontTriangle, centre_m: c, teeth: Some(12), radius_m: 0.0, span: Span::Lower, side, inertia_kgm2: 2e-6, enabled: true, threaded: true });
                let Some((links, work)) = clutch_work(&linkage, &d) else { continue };
                let target: Vec<f64> = TARGET_GEARS.clone().filter_map(|g| work[g]).collect();
                if target.len() != TARGET_GEARS.clone().count() {
                    continue;
                }
                let worst = target.iter().cloned().fold(0.0, f64::max);
                rows.push((worst, c, side, dist, links, work));
            }
        }
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut csv = String::from("x_mm,y_mm,side,from_bb_mm,links,worst_target_deg");
    for t in &teeth {
        csv.push_str(&format!(",cog{t}_deg"));
    }
    csv.push('\n');
    for (worst, c, side, dist, links, work) in &rows {
        csv.push_str(&format!("{:.1},{:.1},{:?},{:.1},{},{:.3}", c[0] * 1000.0, c[1] * 1000.0, side, dist * 1000.0, links, worst));
        for w in work {
            csv.push_str(&w.map_or(",".into(), |w| format!(",{w:.3}")));
        }
        csv.push('\n');
    }
    std::fs::write(&out, csv).expect("write the csv");

    println!("\n{} candidates that close in every target cog; best 15 (centre from the rear axle, mm; relative to the BB):", rows.len());
    for (worst, c, side, dist, links, work) in rows.iter().take(15) {
        let cogs: Vec<String> = TARGET_GEARS.clone().map(|g| format!("{:.2}", work[g].unwrap())).collect();
        println!(
            "  ({:6.1}, {:6.1})  BB{:+5.0},{:+5.0}  {:<5?} {:3.0} mm from BB  {links} links  worst {worst:5.2} deg  [{}]",
            c[0] * 1000.0, c[1] * 1000.0, (c[0] - bb[0]) * 1000.0, (c[1] - bb[1]) * 1000.0, side, dist * 1000.0, cogs.join(" ")
        );
    }
    println!("\nwrote {out}");
}
