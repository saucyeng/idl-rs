//! Test fixtures: the bike as the PyTorch exporters write it
//! (`ml/experiments/export_golden.py`, `export_window.py`).

use super::bike::{Bike, FourBar, Rear};
use super::la::{M3, V3};

fn v3(j: &serde_json::Value) -> V3<f64> {
    let v: Vec<f64> = serde_json::from_value(j.clone()).unwrap();
    V3([v[0], v[1], v[2]])
}

fn m3(j: &serde_json::Value) -> M3<f64> {
    let v: Vec<f64> = serde_json::from_value(j.clone()).unwrap();
    M3([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
}

/// The bike from an exporter's `"bike"` object.
pub fn bike(b: &serde_json::Value) -> Bike<f64> {
    let lk = &b["linkage"];
    let p2 = |k: &str| -> [f64; 2] { serde_json::from_value(lk["side_rel_imu0"][k].clone()).unwrap() };
    Bike {
        up: v3(&b["up"]),
        fwd: v3(&b["forward"]),
        steer_axis: v3(&b["steer_axis"]),
        steer_point: v3(&b["steer_point"]),
        fork_mount: m3(&b["fork_mount"]),
        fork_lever: v3(&b["fork_lever"]),
        bend_len: b["bend_len"].as_f64().unwrap(),
        bush_overlap: b["bush_overlap"].as_f64().unwrap(),
        bend_vs_travel: true,
        rear_mount: m3(&b["rear_mount"]),
        rear_lever: v3(&b["rear_lever"]),
        rear: Rear::FourBar(FourBar {
            main_pivot: p2("main_pivot"),
            horst: p2("horst"),
            ss_rocker: p2("ss_rocker"),
            rocker_frame: p2("rocker_frame"),
            coef: serde_json::from_value(lk["coef"].clone()).unwrap(),
            sign: lk["sign"].as_f64().unwrap(),
        }),
        link_scale: b["link_scale"].as_f64().unwrap(),
        mount_err_fork: v3(&b["mount_err_fork"]),
        mount_err_rear: v3(&b["mount_err_rear"]),
    }
}
