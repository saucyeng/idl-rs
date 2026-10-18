//! The `idl1-video` URI scheme (C3 §3.12): how a `<video>` element gets a
//! linked video's bytes without any of them crossing IPC. The app registers
//! [`video_protocol`] on its builder in `app/src-tauri` (video spec §10 item
//! 5); `app/src/ipc/video.ts` builds the URL with
//! `convertFileSrc(sha256, "idl1-video")`.
//!
//! `GET /<sha256>` serves `blobs/sha256/…` with a single `Range` (`206` and
//! `Content-Range`), `Accept-Ranges: bytes` and `Content-Type: video/mp4`.
//! Only a blob that a live (not unlinked) `videos` entry references is
//! served; anything else is `404`. One response carries at most
//! [`MAX_RANGE_BYTES`]; the media element asks again for the rest, as it
//! does for any server that answers ranges.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use tauri::http::{header, Method, Request, Response, StatusCode};
use tauri::Manager;

use crate::state::DataDir;

/// Most bytes one response carries, bytes (8 MiB). Keeps a 4 GB chapter
/// from ever being read whole.
pub const MAX_RANGE_BYTES: u64 = 8 * 1024 * 1024;

/// Why a `Range` header could not be served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeRefusal {
    /// Not `bytes=…`, or a number did not parse: `400`.
    Malformed,
    /// Starts at or past the end, names several ranges, or the file is
    /// empty: `416`.
    Unsatisfiable,
}

/// The inclusive byte span `(start, end)` to send for `header` (the
/// request's `Range`, if any) against a file of `len` bytes, never more
/// than `cap` bytes long. No header is read as `bytes=0-`: the media element
/// always sends one, and a whole-file answer is what this exists to avoid.
/// `bytes=a-` and `bytes=a-b` are clamped to the file and to `cap`;
/// `bytes=-n` is the last `n` bytes, clamped to `cap` from its start.
pub fn plan_range(header: Option<&str>, len: u64, cap: u64) -> Result<(u64, u64), RangeRefusal> {
    if len == 0 || cap == 0 {
        return Err(RangeRefusal::Unsatisfiable);
    }
    let clamp = |start: u64, end: u64| (start, end.min(len - 1).min(start + cap - 1));
    let Some(header) = header else { return Ok(clamp(0, len - 1)) };
    let spec = header.trim().strip_prefix("bytes=").ok_or(RangeRefusal::Malformed)?;
    if spec.contains(',') {
        return Err(RangeRefusal::Unsatisfiable);
    }
    let (start_text, end_text) = spec.split_once('-').ok_or(RangeRefusal::Malformed)?;
    let number = |t: &str| t.trim().parse::<u64>().map_err(|_| RangeRefusal::Malformed);
    if start_text.trim().is_empty() {
        let suffix = number(end_text)?;
        if suffix == 0 {
            return Err(RangeRefusal::Unsatisfiable);
        }
        return Ok(clamp(len.saturating_sub(suffix), len - 1));
    }
    let start = number(start_text)?;
    if start >= len {
        return Err(RangeRefusal::Unsatisfiable);
    }
    if end_text.trim().is_empty() {
        return Ok(clamp(start, len - 1));
    }
    let end = number(end_text)?;
    if end < start {
        return Err(RangeRefusal::Malformed);
    }
    Ok(clamp(start, end))
}

/// The blob hash a request path names: `/<64 lowercase hex>`, nothing else.
fn sha_of_path(path: &str) -> Option<&str> {
    let sha = path.strip_prefix('/').unwrap_or(path).trim_end_matches('/');
    (sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))).then_some(sha)
}

fn status_only(status: StatusCode) -> Response<Vec<u8>> {
    Response::builder().status(status).body(Vec::new()).unwrap()
}

/// Answers one `idl1-video` request against `data_dir`. Pure apart from
/// `std::fs`: the request logic, with no Tauri runtime involved.
pub fn serve(data_dir: &Path, request: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    if request.method() != Method::GET {
        return status_only(StatusCode::METHOD_NOT_ALLOWED);
    }
    let Some(sha) = sha_of_path(request.uri().path()) else { return status_only(StatusCode::NOT_FOUND) };
    if !idl_rs::store::video::live_video_blobs(data_dir).contains(sha) {
        return status_only(StatusCode::NOT_FOUND);
    }
    let path = idl_rs::store::blob::blob_path(data_dir, sha);
    let Ok(mut file) = std::fs::File::open(&path) else { return status_only(StatusCode::NOT_FOUND) };
    let Ok(len) = file.metadata().map(|m| m.len()) else { return status_only(StatusCode::INTERNAL_SERVER_ERROR) };

    let range = request.headers().get(header::RANGE).and_then(|v| v.to_str().ok());
    let (start, end) = match plan_range(range, len, MAX_RANGE_BYTES) {
        Ok(span) => span,
        Err(RangeRefusal::Malformed) => return status_only(StatusCode::BAD_REQUEST),
        Err(RangeRefusal::Unsatisfiable) => {
            return Response::builder()
                .status(StatusCode::RANGE_NOT_SATISFIABLE)
                .header(header::CONTENT_RANGE, format!("bytes */{len}"))
                .body(Vec::new())
                .unwrap()
        }
    };

    let mut body = vec![0u8; (end - start + 1) as usize];
    if file.seek(SeekFrom::Start(start)).and_then(|_| file.read_exact(&mut body)).is_err() {
        return status_only(StatusCode::INTERNAL_SERVER_ERROR);
    }
    Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, "video/mp4")
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{len}"))
        .header(header::CONTENT_LENGTH, body.len().to_string())
        .body(body)
        .unwrap()
}

/// The `idl1-video` scheme handler, registered by the app crate with
/// `register_asynchronous_uri_scheme_protocol("idl1-video", video_protocol)`.
/// The file I/O runs on its own thread, never the caller's; the data
/// directory is managed state [`DataDir`].
pub fn video_protocol<R: tauri::Runtime>(
    ctx: tauri::UriSchemeContext<'_, R>,
    request: tauri::http::Request<Vec<u8>>,
    responder: tauri::UriSchemeResponder,
) {
    let Some(data_dir) = ctx.app_handle().try_state::<DataDir>().map(|d| d.0.clone()) else {
        responder.respond(status_only(StatusCode::SERVICE_UNAVAILABLE));
        return;
    };
    std::thread::spawn(move || responder.respond(serve(&data_dir, &request)));
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn plan_range_no_header_reads_as_from_zero_capped() {
        // Act + Assert
        assert_eq!(plan_range(None, 100, 8 * MIB), Ok((0, 99)));
        assert_eq!(plan_range(None, 100 * MIB, 8 * MIB), Ok((0, 8 * MIB - 1)));
    }

    #[test]
    fn plan_range_open_ended_is_capped_and_a_bounded_range_is_kept_when_small() {
        // Act + Assert
        assert_eq!(plan_range(Some("bytes=10-"), 100 * MIB, 8 * MIB), Ok((10, 10 + 8 * MIB - 1)));
        assert_eq!(plan_range(Some("bytes=10-19"), 100, 8 * MIB), Ok((10, 19)));
        assert_eq!(plan_range(Some("bytes=90-500"), 100, 8 * MIB), Ok((90, 99)), "clamped to the file");
        assert_eq!(plan_range(Some("bytes=0-4000000000"), 5_000_000_000, 8 * MIB), Ok((0, 8 * MIB - 1)), "clamped to the cap");
    }

    #[test]
    fn plan_range_suffix_is_the_last_n_bytes() {
        // Act + Assert
        assert_eq!(plan_range(Some("bytes=-10"), 100, 8 * MIB), Ok((90, 99)));
        assert_eq!(plan_range(Some("bytes=-1000"), 100, 8 * MIB), Ok((0, 99)));
    }

    #[test]
    fn plan_range_refusals_are_typed() {
        // Act + Assert
        assert_eq!(plan_range(Some("bytes=100-"), 100, 8 * MIB), Err(RangeRefusal::Unsatisfiable));
        assert_eq!(plan_range(Some("bytes=0-1,5-9"), 100, 8 * MIB), Err(RangeRefusal::Unsatisfiable));
        assert_eq!(plan_range(Some("bytes=-0"), 100, 8 * MIB), Err(RangeRefusal::Unsatisfiable));
        assert_eq!(plan_range(Some("items=0-1"), 100, 8 * MIB), Err(RangeRefusal::Malformed));
        assert_eq!(plan_range(Some("bytes=a-b"), 100, 8 * MIB), Err(RangeRefusal::Malformed));
        assert_eq!(plan_range(Some("bytes=9-3"), 100, 8 * MIB), Err(RangeRefusal::Malformed));
        assert_eq!(plan_range(None, 0, 8 * MIB), Err(RangeRefusal::Unsatisfiable));
    }

    #[test]
    fn sha_of_path_accepts_only_a_bare_lowercase_digest() {
        // Arrange
        let sha = "ab".repeat(32);

        // Act + Assert
        assert_eq!(sha_of_path(&format!("/{sha}")), Some(sha.as_str()));
        assert_eq!(sha_of_path(&format!("/{}", sha.to_uppercase())), None);
        assert_eq!(sha_of_path("/../catalog.sqlite"), None);
        assert_eq!(sha_of_path(&format!("/x/{sha}")), None);
    }

    /// Writes `bytes` as a blob and links it to a session whose
    /// `session.json` holds one `videos` entry, live or not.
    fn data_dir_with_video(bytes: &[u8], unlinked: bool) -> (tempfile::TempDir, String) {
        use idl_rs::store::session_json::{empty_session_json, write_session_json};
        let root = tempfile::tempdir().unwrap();
        let sha = idl_rs::store::blob::write_blob(root.path(), bytes).unwrap();
        let session_id = "00112233445566778899aabbccddeeff";
        let mut doc = empty_session_json(session_id);
        doc.videos.push(idl_rs::video::VideoLink {
            blob_sha256: sha.clone(),
            size_bytes: bytes.len() as u64,
            file_name: "a.mp4".to_string(),
            video_codec: "h264".to_string(),
            width_px: 1,
            height_px: 1,
            fps: 1.0,
            duration_s: 1.0,
            offset_s: 0.0,
            sync_method: idl_rs::video::SyncMethod::Manual,
            unlinked,
            updated_at_ms: 1,
        });
        write_session_json(root.path(), session_id, &doc, None).unwrap();
        (root, sha)
    }

    fn get(sha: &str, range: Option<&str>) -> Request<Vec<u8>> {
        let mut builder = Request::builder().method(Method::GET).uri(format!("http://idl1-video.localhost/{sha}"));
        if let Some(r) = range {
            builder = builder.header(header::RANGE, r);
        }
        builder.body(Vec::new()).unwrap()
    }

    #[test]
    fn serve_a_live_video_answers_the_range_with_206_and_headers() {
        // Arrange
        let (root, sha) = data_dir_with_video(b"0123456789", false);

        // Act
        let response = serve(root.path(), &get(&sha, Some("bytes=2-5")));

        // Assert
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(response.body(), b"2345");
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes 2-5/10");
        assert_eq!(response.headers()[header::CONTENT_TYPE], "video/mp4");
        assert_eq!(response.headers()[header::ACCEPT_RANGES], "bytes");
    }

    #[test]
    fn serve_an_unlinked_video_or_an_unknown_hash_is_404() {
        // Arrange
        let (root, sha) = data_dir_with_video(b"0123456789", true);

        // Act
        let unlinked = serve(root.path(), &get(&sha, None));
        let unknown = serve(root.path(), &get(&"c".repeat(64), None));

        // Assert
        assert_eq!(unlinked.status(), StatusCode::NOT_FOUND);
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn serve_an_unsatisfiable_range_is_416_naming_the_length() {
        // Arrange
        let (root, sha) = data_dir_with_video(b"0123456789", false);

        // Act
        let response = serve(root.path(), &get(&sha, Some("bytes=10-")));

        // Assert
        assert_eq!(response.status(), StatusCode::RANGE_NOT_SATISFIABLE);
        assert_eq!(response.headers()[header::CONTENT_RANGE], "bytes */10");
    }
}
