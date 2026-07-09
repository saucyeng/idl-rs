//! Integration tests for `idl-rs overlay` error paths that need no ffmpeg:
//! layout selection and the missing-binary message. The happy path is
//! covered by `idl-rs-video-export`'s end-to-end smoke test.

use std::path::PathBuf;
use std::process::Command;

use idl_rs::parse::test_buffers::{
    cat, frame, gps_payload, imu_payload, session_end, v3_imu_axes_registry, v3_registry_entry,
    Header, RMC_UTC_MS,
};
use idl_rs::video::mp4box::fixture::synthetic_mp4;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_idl-rs"))
}

/// Write a minimal valid v3 session (the parser round-trip fixture) + a
/// synthetic MP4 + a two-layout workbook into a temp dir.
fn fixtures(dir_name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(dir_name);
    std::fs::create_dir_all(&dir).unwrap();

    let session = dir.join("session.idl0");
    let accel_scale: f32 = 32.0 / 32768.0;
    let gyro_scale: f32 = 2000.0 / 32768.0;
    let mut registry = v3_imu_axes_registry(0, 0, 800, accel_scale, gyro_scale);
    registry.push(v3_registry_entry(18, 2, 0, 1.0, 0.0, "WheelFront", "pulse"));
    let buf = cat(&[
        Header {
            schema_version: 3,
            ..Default::default()
        }
        .build(&registry),
        frame(
            0x01,
            &imu_payload(0, 1250, &[16384, -8192, 0, 1000, -500, 0]),
        ),
        frame(
            0x02,
            &gps_payload(
                RMC_UTC_MS,
                1250,
                515_250_000,
                -1_234_567,
                500,
                1000,
                18000,
                1,
                8,
            ),
        ),
        frame(0x03, &frame_channel_u32(18, 2_000_000, 99)),
        session_end(),
    ]);
    std::fs::write(&session, buf).unwrap();

    let video = dir.join("video.mp4");
    std::fs::write(&video, synthetic_mp4(0, &[b"x"])).unwrap();

    let workbook = dir.join("wb.idl0wb");
    std::fs::write(
        &workbook,
        r#"{ "workbook_id": "w1", "name": "wb", "workbook_version": 2,
          "overlay_layouts": [
            { "id": "L1", "name": "A", "canvas": "1920x1080",
              "elements": [ { "type": "track_map", "rect": [0.8, 0.0, 0.2, 0.3] } ] },
            { "id": "L2", "name": "B", "canvas": "1920x1080", "elements": [] } ] }"#,
    )
    .unwrap();

    (session, video, workbook)
}

// `channel_payload_u32` under its test_buffers name.
use idl_rs::parse::test_buffers::channel_payload_u32 as frame_channel_u32;

#[test]
fn overlay_without_layout_flag_lists_available_layout_names() {
    // Arrange
    let (session, video, workbook) = fixtures("idlrs_cli_overlay_layouts");

    // Act
    let out = bin()
        .arg("overlay")
        .arg(&session)
        .args(["--offset", "0"])
        .arg("--video")
        .arg(&video)
        .arg("--workbook")
        .arg(&workbook)
        .output()
        .unwrap();

    // Assert — bulk command: error envelope on stderr, non-zero exit.
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--layout"), "stderr: {stderr}");
    assert!(
        stderr.contains('A') && stderr.contains('B'),
        "stderr: {stderr}"
    );
}

#[test]
fn overlay_with_missing_ffmpeg_binary_says_install_or_pass_ffmpeg() {
    // Arrange
    let (session, video, workbook) = fixtures("idlrs_cli_overlay_ffmpeg");

    // Act
    let out = bin()
        .arg("overlay")
        .arg(&session)
        .args([
            "--offset",
            "0",
            "--layout",
            "A",
            "--ffmpeg",
            "definitely-missing-binary",
        ])
        .arg("--video")
        .arg(&video)
        .arg("--workbook")
        .arg(&workbook)
        .output()
        .unwrap();

    // Assert
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("install ffmpeg or pass --ffmpeg"),
        "stderr: {stderr}"
    );
}
