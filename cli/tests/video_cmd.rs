//! Integration tests for `idl-rs video probe` / `video sync` over a synthetic
//! MP4 (core's `test-fixtures` builder — no real footage exists yet).

use std::process::Command;

use idl_rs::video::mp4box::fixture::synthetic_mp4;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_idl-rs"))
}

#[test]
fn video_probe_json_reports_dims_and_gpmf_presence() {
    // Arrange
    let dir = std::env::temp_dir().join("idlrs_cli_video_probe");
    std::fs::create_dir_all(&dir).unwrap();
    let video = dir.join("synthetic.mp4");
    std::fs::write(&video, synthetic_mp4(0, &[b"x"])).unwrap();

    // Act
    let out = bin()
        .args(["video", "probe", "--format", "json", "--video"])
        .arg(&video)
        .output()
        .unwrap();

    // Assert
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let env: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(env["ok"], true);
    assert_eq!(env["data"]["width"], 1920);
    assert_eq!(env["data"]["height"], 1080);
    assert_eq!(env["data"]["has_gpmd"], true);
}

#[test]
fn video_probe_missing_file_emits_error_envelope() {
    // Arrange + Act
    let out = bin()
        .args([
            "video",
            "probe",
            "--format",
            "json",
            "--video",
            "does-not-exist.mp4",
        ])
        .output()
        .unwrap();

    // Assert — structured commands put the error envelope on stdout.
    assert!(!out.status.success());
    let env: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(env["ok"], false);
    assert_eq!(env["error"]["kind"], "io");
}
