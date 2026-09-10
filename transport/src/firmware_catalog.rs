//! The firmware release catalog: GitHub Releases of the repository named in
//! Settings, as a list of installable firmware images (R198).
//!
//! This is the **only** call the app makes to the public internet, and it
//! runs only on "Check now" or on opening the Firmware section with the
//! catalog enabled — never on a timer (R198). It lives in `idl-transport`
//! rather than `idl-rs-tauri` for the reason CLAUDE.md §2 gives: bytes on
//! the wire belong to this crate, and `idl-rs-tauri` stays glue.
//!
//! Everything except [`fetch_releases`] and [`download_image`] is pure and
//! unit-tested without a network: release-JSON parsing, asset matching,
//! semver precedence, channel selection, and the `sha256sum` sidecar format.
//!
//! **Asset naming (R198):** a release is installable only if it carries an
//! asset named `idl1-firmware-<semver>.bin`; an `idl1-firmware-<semver>.bin
//! .sha256` sidecar beside it is optional but is what arms auto-confirm.
//! SPEC §27.7 describes idl0's `idl0-firmware-v<ver>.bin.sha256`; R198 fixes
//! idl1's names and wins here (CLAUDE.md: the app-side SPEC parts describe
//! idl0 and are rewritten lane by lane).

use std::cmp::Ordering;

use crate::{TransportError, TransportErrorKind};

/// Builds a `TransportErrorKind::Wifi` error with `message` — the catalog is
/// an HTTP transfer like the device's own endpoints, and the app routes the
/// two the same way (there is no "internet" error kind, and inventing one
/// would be a cross-lane change to C3 §2 for no behavioural gain).
fn catalog_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Wifi, message)
}

/// Which release channel a check is against (R198).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FirmwareChannel {
    /// The latest non-prerelease release.
    Stable,
    /// The latest release including prereleases.
    Beta,
}

/// One installable firmware release from the catalog.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FirmwareRelease {
    /// The release version as semver text, leading `v` stripped — the same
    /// value the device reports as SPEC §7.3's `Firmware:` line and `/ping`'s
    /// `fw`, because CI builds the image version from the git tag (SPEC
    /// §27.7, "Version of record").
    pub version: String,
    /// The git tag verbatim, e.g. `"v1.6.0-beta.1"`.
    pub tag: String,
    /// The release's display name, or the tag when GitHub reports none.
    pub name: String,
    /// The release body (markdown, as authored). May be empty.
    pub notes: String,
    /// Whether GitHub marks this a prerelease — `true` for anything the
    /// `stable` channel must skip.
    pub prerelease: bool,
    /// Publication timestamp, verbatim from GitHub (RFC 3339). Empty when
    /// GitHub reported none.
    pub published_at: String,
    /// Download URL of the `idl1-firmware-<version>.bin` asset.
    pub image_url: String,
    /// Size of that asset, bytes, as GitHub reports it.
    pub image_size_bytes: u64,
    /// Download URL of the `idl1-firmware-<version>.bin.sha256` sidecar,
    /// when the release publishes one. `None` means no app-side integrity
    /// check is possible, which is what keeps auto-confirm disarmed (R198).
    pub sha256_url: Option<String>,
}

/// A parsed semver version: the three numeric fields plus the prerelease
/// tag (`""` for a release). Build metadata is discarded — it does not take
/// part in precedence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemVer {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    /// The `-`-suffixed prerelease identifier without its dash, e.g.
    /// `"beta.1"`. Empty for a plain release.
    pub prerelease: String,
}

/// Parses `text` as semver, tolerating a leading `v` (git tags carry one,
/// `esp_app_desc_t.version` does not). Returns `None` for anything that is
/// not `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]` with numeric fields — R198's
/// "device version parses as semver (else 'unknown version')" hinges on this
/// returning `None` rather than guessing.
pub fn parse_semver(text: &str) -> Option<SemVer> {
    let text = text.trim();
    let text = text.strip_prefix('v').unwrap_or(text);
    let (text, _build) = match text.split_once('+') {
        Some((head, build)) => (head, build),
        None => (text, ""),
    };
    let (core, prerelease) = match text.split_once('-') {
        Some((head, tail)) => (head, tail),
        None => (text, ""),
    };
    let mut fields = core.split('.');
    let major = fields.next()?.parse().ok()?;
    let minor = fields.next()?.parse().ok()?;
    let patch = fields.next()?.parse().ok()?;
    if fields.next().is_some() {
        return None;
    }
    Some(SemVer { major, minor, patch, prerelease: prerelease.to_string() })
}

/// Semver precedence between two parsed versions: numeric fields first, then
/// "a prerelease is *lower* than the release it precedes" (semver §11), then
/// the prerelease identifiers compared dot-part by dot-part, numeric parts
/// numerically and everything else lexically.
pub fn compare_semver(left: &SemVer, right: &SemVer) -> Ordering {
    let numeric = (left.major, left.minor, left.patch).cmp(&(right.major, right.minor, right.patch));
    if numeric != Ordering::Equal {
        return numeric;
    }
    match (left.prerelease.is_empty(), right.prerelease.is_empty()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => compare_prerelease(&left.prerelease, &right.prerelease),
    }
}

/// Compares two non-empty prerelease identifiers (semver §11.4): dot-part by
/// dot-part, numeric parts numerically and lower than alphanumeric ones, and
/// a shorter identifier lower when every shared part is equal.
fn compare_prerelease(left: &str, right: &str) -> Ordering {
    let mut left_parts = left.split('.');
    let mut right_parts = right.split('.');
    loop {
        match (left_parts.next(), right_parts.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(l), Some(r)) => {
                let ordering = match (l.parse::<u64>(), r.parse::<u64>()) {
                    (Ok(l), Ok(r)) => l.cmp(&r),
                    (Ok(_), Err(_)) => Ordering::Less,
                    (Err(_), Ok(_)) => Ordering::Greater,
                    (Err(_), Err(_)) => l.cmp(r),
                };
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
        }
    }
}

/// Compares two version *strings*, returning `None` when either side does
/// not parse as semver — the caller's cue to say "unknown version" rather
/// than to offer an update (R198).
pub fn compare_versions(left: &str, right: &str) -> Option<Ordering> {
    Some(compare_semver(&parse_semver(left)?, &parse_semver(right)?))
}

/// The asset filename a release of `version` must publish to be installable.
pub fn image_asset_name(version: &str) -> String {
    format!("idl1-firmware-{version}.bin")
}

/// The optional sha256 sidecar filename beside [`image_asset_name`].
pub fn sha256_asset_name(version: &str) -> String {
    format!("{}.sha256", image_asset_name(version))
}

/// Parses the JSON body of GitHub's `GET /repos/{repo}/releases` into the
/// releases that are actually installable, newest version first.
///
/// Skipped, silently and by design (a catalog is a list of what *can* be
/// installed, not a list of complaints): drafts, releases whose tag is not
/// semver, and releases carrying no `idl1-firmware-<version>.bin` asset.
/// A body that is not a JSON array at all is an error, since that means the
/// request did not reach the API this function is written against.
pub fn releases_from_github_json(body: &str) -> Result<Vec<FirmwareRelease>, TransportError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| catalog_error(format!("the release catalog returned malformed JSON: {e}")))?;
    let entries = parsed
        .as_array()
        .ok_or_else(|| catalog_error("the release catalog returned something other than a JSON array"))?;

    let mut releases: Vec<FirmwareRelease> = entries.iter().filter_map(release_from_entry).collect();
    releases.sort_by(|a, b| {
        let ordering = match (parse_semver(&b.version), parse_semver(&a.version)) {
            (Some(newer), Some(older)) => compare_semver(&newer, &older),
            _ => Ordering::Equal,
        };
        // Ties (the same version published twice) keep GitHub's own order.
        ordering
    });
    Ok(releases)
}

/// Builds one [`FirmwareRelease`] from a GitHub release object, or `None`
/// when the release is not installable (see [`releases_from_github_json`]).
fn release_from_entry(entry: &serde_json::Value) -> Option<FirmwareRelease> {
    if entry.get("draft").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        return None;
    }
    let tag = entry.get("tag_name").and_then(serde_json::Value::as_str)?.to_string();
    let semver = parse_semver(&tag)?;
    let version = format!(
        "{}.{}.{}{}{}",
        semver.major,
        semver.minor,
        semver.patch,
        if semver.prerelease.is_empty() { "" } else { "-" },
        semver.prerelease
    );

    let assets = entry.get("assets").and_then(serde_json::Value::as_array)?;
    let asset_named = |wanted: &str| {
        assets.iter().find(|asset| asset.get("name").and_then(serde_json::Value::as_str) == Some(wanted))
    };
    let image = asset_named(&image_asset_name(&version))?;

    Some(FirmwareRelease {
        image_url: image.get("browser_download_url").and_then(serde_json::Value::as_str)?.to_string(),
        image_size_bytes: image.get("size").and_then(serde_json::Value::as_u64).unwrap_or(0),
        sha256_url: asset_named(&sha256_asset_name(&version))
            .and_then(|asset| asset.get("browser_download_url"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        name: entry
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty())
            .unwrap_or(&tag)
            .to_string(),
        notes: entry.get("body").and_then(serde_json::Value::as_str).unwrap_or("").to_string(),
        prerelease: entry.get("prerelease").and_then(serde_json::Value::as_bool).unwrap_or(false),
        published_at: entry.get("published_at").and_then(serde_json::Value::as_str).unwrap_or("").to_string(),
        version,
        tag,
    })
}

/// Filters `releases` to those the channel admits (R198): `stable` keeps
/// only non-prereleases, `beta` keeps everything. Order is preserved, so a
/// list from [`releases_from_github_json`] stays newest-first and its head
/// is the channel's latest.
pub fn for_channel(releases: &[FirmwareRelease], channel: FirmwareChannel) -> Vec<FirmwareRelease> {
    releases
        .iter()
        .filter(|release| match channel {
            FirmwareChannel::Stable => !release.prerelease,
            FirmwareChannel::Beta => true,
        })
        .cloned()
        .collect()
}

/// Reads the digest out of a `sha256sum`-format sidecar — one or more lines
/// of `<64 hex>  <filename>`, of which only the first is used (the sidecar
/// covers one asset). Returns the digest lowercased, or `None` when the file
/// is not in that format.
pub fn parse_sha256_sidecar(text: &str) -> Option<String> {
    let first_line = text.lines().find(|line| !line.trim().is_empty())?;
    let digest = first_line.split_whitespace().next()?;
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(digest.to_ascii_lowercase())
}

/// A downloaded firmware image and what is known about its integrity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadedImage {
    /// The raw `.bin` bytes, ready for `WifiTransport::push_ota`.
    pub bytes: Vec<u8>,
    /// The image's actual SHA-256, lowercase hex.
    pub sha256: String,
    /// `true` only when the release published a sidecar *and* it matched.
    /// This is exactly the condition R198 arms auto-confirm on; a release
    /// with no sidecar downloads fine but leaves this `false`.
    pub sha256_verified: bool,
}

/// GitHub asks every API client to identify itself; an unset `User-Agent`
/// is answered with 403.
const CATALOG_USER_AGENT: &str = "idl1-app";

/// Fetches `repo`'s releases and returns the ones `channel` admits, newest
/// first (R198). `repo` is `"owner/name"`, as typed in Settings.
///
/// Public API, no auth token, no server (SPEC §27.7). An empty `repo` is a
/// caller error, not a network round trip: the catalog is disabled by
/// default and the caller must not reach here with it unset.
pub async fn fetch_releases(
    repo: &str,
    channel: FirmwareChannel,
) -> Result<Vec<FirmwareRelease>, TransportError> {
    fetch_releases_from(&format!("https://api.github.com/repos/{repo}/releases"), repo, channel).await
}

/// [`fetch_releases`] with the API base spelled out, so this module's tests
/// can point it at the local mock HTTP server instead of github.com.
async fn fetch_releases_from(
    url: &str,
    repo: &str,
    channel: FirmwareChannel,
) -> Result<Vec<FirmwareRelease>, TransportError> {
    if repo.trim().is_empty() {
        return Err(catalog_error("no firmware repository is configured"));
    }
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .header(reqwest::header::USER_AGENT, CATALOG_USER_AGENT)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| catalog_error(format!("could not reach the release catalog: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(catalog_error(format!(
            "the release catalog for {repo} returned status {}",
            status.as_u16()
        )));
    }
    let body = response
        .text()
        .await
        .map_err(|e| catalog_error(format!("could not read the release catalog's response: {e}")))?;
    Ok(for_channel(&releases_from_github_json(&body)?, channel))
}

/// Downloads `release`'s image, hashes it, and verifies it against the
/// release's sidecar when there is one.
///
/// A **mismatch is an error** — R198: "a download whose sha256 mismatches is
/// refused before any push". A *missing* sidecar is not an error; it comes
/// back as [`DownloadedImage::sha256_verified`] `== false`, which is what
/// keeps auto-confirm disarmed for that push.
///
/// `on_progress(done_bytes, total_bytes)` reports the image download only,
/// with `total_bytes` from the response's `Content-Length` (or the catalog's
/// own `image_size_bytes` when the server sends none), and `None` when
/// neither is known.
pub async fn download_image(
    release: &FirmwareRelease,
    on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<DownloadedImage, TransportError> {
    use futures::StreamExt;

    let client = reqwest::Client::new();
    let response = client
        .get(&release.image_url)
        .header(reqwest::header::USER_AGENT, CATALOG_USER_AGENT)
        .send()
        .await
        .map_err(|e| catalog_error(format!("could not download the firmware image: {e}")))?;
    if !response.status().is_success() {
        return Err(catalog_error(format!(
            "downloading the firmware image returned status {}",
            response.status().as_u16()
        )));
    }
    let total_bytes = response
        .content_length()
        .or(if release.image_size_bytes > 0 { Some(release.image_size_bytes) } else { None });

    let mut bytes: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| catalog_error(format!("the firmware download failed: {e}")))?;
        bytes.extend_from_slice(&chunk);
        on_progress(bytes.len() as u64, total_bytes);
    }

    let sha256 = idl_rs::store::atomic::sha256_hex(&bytes);
    let expected = match &release.sha256_url {
        None => None,
        Some(url) => {
            let sidecar = client
                .get(url)
                .header(reqwest::header::USER_AGENT, CATALOG_USER_AGENT)
                .send()
                .await
                .map_err(|e| catalog_error(format!("could not download the firmware checksum: {e}")))?
                .text()
                .await
                .map_err(|e| catalog_error(format!("could not read the firmware checksum: {e}")))?;
            parse_sha256_sidecar(&sidecar)
        }
    };
    match expected {
        Some(expected) if expected != sha256 => Err(catalog_error(format!(
            "the downloaded firmware image does not match its published checksum (expected {expected}, got {sha256}) — nothing was sent to the device"
        ))),
        Some(_) => Ok(DownloadedImage { bytes, sha256, sha256_verified: true }),
        None => Ok(DownloadedImage { bytes, sha256, sha256_verified: false }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GitHub `/releases` body: one stable, one prerelease, one draft, one
    /// non-semver tag, and one release with no `.bin` asset.
    pub(super) fn releases_json() -> String {
        serde_json::json!([
            {
                "tag_name": "v1.6.0-beta.1", "name": "1.6.0 beta 1", "body": "try it",
                "draft": false, "prerelease": true, "published_at": "2026-09-08T10:00:00Z",
                "assets": [
                    {"name": "idl1-firmware-1.6.0-beta.1.bin", "size": 1600, "browser_download_url": "https://example.invalid/beta.bin"}
                ]
            },
            {
                "tag_name": "v1.5.0", "name": "", "body": "stable notes",
                "draft": false, "prerelease": false, "published_at": "2026-09-01T10:00:00Z",
                "assets": [
                    {"name": "idl1-firmware-1.5.0.bin", "size": 1500, "browser_download_url": "https://example.invalid/1.5.0.bin"},
                    {"name": "idl1-firmware-1.5.0.bin.sha256", "size": 80, "browser_download_url": "https://example.invalid/1.5.0.sha256"}
                ]
            },
            {
                "tag_name": "v1.7.0", "name": "unreleased", "body": "",
                "draft": true, "prerelease": false, "published_at": "2026-09-09T10:00:00Z",
                "assets": [
                    {"name": "idl1-firmware-1.7.0.bin", "size": 1700, "browser_download_url": "https://example.invalid/1.7.0.bin"}
                ]
            },
            {
                "tag_name": "nightly", "name": "nightly", "body": "",
                "draft": false, "prerelease": false, "published_at": "2026-09-07T10:00:00Z",
                "assets": [
                    {"name": "idl1-firmware-nightly.bin", "size": 10, "browser_download_url": "https://example.invalid/nightly.bin"}
                ]
            },
            {
                "tag_name": "v1.4.0", "name": "1.4.0", "body": "",
                "draft": false, "prerelease": false, "published_at": "2026-08-01T10:00:00Z",
                "assets": [
                    {"name": "release-notes.pdf", "size": 10, "browser_download_url": "https://example.invalid/notes.pdf"}
                ]
            }
        ])
        .to_string()
    }

    #[test]
    fn parse_semver_plain_leading_v_prerelease_and_build_all_parse_to_the_same_core() {
        // Arrange
        let inputs = ["1.5.0", "v1.5.0", "1.5.0+abc123"];

        // Act
        let parsed: Vec<SemVer> = inputs.iter().map(|t| parse_semver(t).unwrap()).collect();
        let prerelease = parse_semver("v1.6.0-beta.1").unwrap();

        // Assert
        for version in &parsed {
            assert_eq!((version.major, version.minor, version.patch), (1, 5, 0));
            assert_eq!(version.prerelease, "");
        }
        assert_eq!((prerelease.major, prerelease.minor, prerelease.patch), (1, 6, 0));
        assert_eq!(prerelease.prerelease, "beta.1");
    }

    #[test]
    fn parse_semver_non_semver_text_returns_none() {
        // Arrange
        let inputs = ["nightly", "1.5", "1.5.0.1", "v1.x.0", ""];

        // Act / Assert
        for input in inputs {
            assert!(parse_semver(input).is_none(), "{input} should not parse as semver");
        }
    }

    #[test]
    fn compare_versions_orders_numeric_fields_then_ranks_a_prerelease_below_its_release() {
        // Arrange / Act
        let newer_patch = compare_versions("1.5.1", "1.5.0");
        let older_minor = compare_versions("1.4.9", "1.5.0");
        let prerelease_vs_release = compare_versions("1.6.0-beta.1", "1.6.0");
        let beta_ordering = compare_versions("1.6.0-beta.2", "1.6.0-beta.1");
        let alpha_vs_beta = compare_versions("1.6.0-alpha", "1.6.0-beta");
        let equal = compare_versions("v1.5.0", "1.5.0");
        let unknown = compare_versions("nightly", "1.5.0");

        // Assert
        assert_eq!(newer_patch, Some(Ordering::Greater));
        assert_eq!(older_minor, Some(Ordering::Less));
        assert_eq!(prerelease_vs_release, Some(Ordering::Less));
        assert_eq!(beta_ordering, Some(Ordering::Greater));
        assert_eq!(alpha_vs_beta, Some(Ordering::Less));
        assert_eq!(equal, Some(Ordering::Equal));
        assert_eq!(unknown, None);
    }

    #[test]
    fn releases_from_github_json_keeps_installable_releases_newest_first() {
        // Arrange
        let body = releases_json();

        // Act
        let releases = releases_from_github_json(&body).unwrap();

        // Assert — draft, non-semver tag and asset-less release all dropped
        let versions: Vec<&str> = releases.iter().map(|r| r.version.as_str()).collect();
        assert_eq!(versions, vec!["1.6.0-beta.1", "1.5.0"]);
        assert_eq!(releases[0].image_url, "https://example.invalid/beta.bin");
        assert_eq!(releases[0].image_size_bytes, 1600);
        assert_eq!(releases[0].sha256_url, None);
        assert_eq!(releases[1].sha256_url.as_deref(), Some("https://example.invalid/1.5.0.sha256"));
    }

    #[test]
    fn releases_from_github_json_empty_release_name_falls_back_to_the_tag() {
        // Arrange
        let body = releases_json();

        // Act
        let releases = releases_from_github_json(&body).unwrap();

        // Assert
        assert_eq!(releases[1].name, "v1.5.0");
        assert_eq!(releases[0].name, "1.6.0 beta 1");
    }

    #[test]
    fn releases_from_github_json_non_array_body_is_a_typed_wifi_error() {
        // Arrange
        let body = r#"{"message":"Not Found"}"#;

        // Act
        let err = releases_from_github_json(body).unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Wifi);
    }

    #[test]
    fn for_channel_stable_drops_prereleases_beta_keeps_them() {
        // Arrange
        let releases = releases_from_github_json(&releases_json()).unwrap();

        // Act
        let stable = for_channel(&releases, FirmwareChannel::Stable);
        let beta = for_channel(&releases, FirmwareChannel::Beta);

        // Assert
        assert_eq!(stable.iter().map(|r| r.version.as_str()).collect::<Vec<_>>(), vec!["1.5.0"]);
        assert_eq!(
            beta.iter().map(|r| r.version.as_str()).collect::<Vec<_>>(),
            vec!["1.6.0-beta.1", "1.5.0"]
        );
    }

    #[test]
    fn parse_sha256_sidecar_sha256sum_format_returns_the_lowercased_digest() {
        // Arrange
        let digest = "A".repeat(64);
        let sidecar = format!("{digest}  idl1-firmware-1.5.0.bin\n");

        // Act
        let parsed = parse_sha256_sidecar(&sidecar);

        // Assert
        assert_eq!(parsed, Some("a".repeat(64)));
    }

    #[test]
    fn parse_sha256_sidecar_wrong_length_or_non_hex_returns_none() {
        // Arrange
        let too_short = format!("{}  f.bin", "a".repeat(63));
        let non_hex = format!("{}  f.bin", "z".repeat(64));

        // Act / Assert
        assert_eq!(parse_sha256_sidecar(&too_short), None);
        assert_eq!(parse_sha256_sidecar(&non_hex), None);
        assert_eq!(parse_sha256_sidecar(""), None);
    }

    #[test]
    fn asset_names_follow_r198s_idl1_firmware_pattern() {
        // Arrange
        let version = "1.6.0-beta.1";

        // Act / Assert
        assert_eq!(image_asset_name(version), "idl1-firmware-1.6.0-beta.1.bin");
        assert_eq!(sha256_asset_name(version), "idl1-firmware-1.6.0-beta.1.bin.sha256");
    }
}

/// Network-shaped tests: [`fetch_releases_from`] and [`download_image`]
/// against the same hand-rolled mock HTTP server the WiFi tests use, never
/// against github.com — this crate's tests make no internet calls.
#[cfg(test)]
mod integration {
    use super::*;
    use crate::wifi_transport::integration::spawn_mock_server;

    #[tokio::test]
    async fn fetch_releases_from_stable_channel_returns_only_the_non_prerelease() {
        // Arrange
        let body = super::tests::releases_json();
        let (addr, _server) = spawn_mock_server(move |_path, _headers, _request_body| {
            (200, "OK", vec![("Content-Type".to_string(), "application/json".to_string())], body.clone().into_bytes())
        })
        .await;

        // Act
        let releases =
            fetch_releases_from(&format!("http://{addr}/releases"), "owner/name", FirmwareChannel::Stable)
                .await
                .unwrap();

        // Assert
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].version, "1.5.0");
    }

    #[tokio::test]
    async fn fetch_releases_from_non_success_status_is_a_typed_error_naming_the_repo() {
        // Arrange
        let (addr, _server) =
            spawn_mock_server(|_path, _headers, _request_body| (404, "Not Found", Vec::new(), Vec::new())).await;

        // Act
        let err = fetch_releases_from(&format!("http://{addr}/releases"), "owner/name", FirmwareChannel::Beta)
            .await
            .unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Wifi);
        assert!(err.message.contains("owner/name"), "message should name the repo: {}", err.message);
    }

    #[tokio::test]
    async fn fetch_releases_empty_repo_never_touches_the_network() {
        // Arrange
        let empty_repo = "   ";

        // Act
        let err = fetch_releases(empty_repo, FirmwareChannel::Stable).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Wifi);
        assert!(err.message.contains("no firmware repository"));
    }

    /// The image bytes both download tests serve, and their digest.
    fn image_fixture() -> (Vec<u8>, String) {
        let bytes: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let digest = idl_rs::store::atomic::sha256_hex(&bytes);
        (bytes, digest)
    }

    /// Serves `/image.bin` and `/image.sha256`, the latter carrying
    /// `sidecar_digest` in `sha256sum` format.
    async fn spawn_image_server(
        bytes: Vec<u8>,
        sidecar_digest: Option<String>,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        spawn_mock_server(move |path, _headers, _request_body| {
            if path.ends_with(".sha256") {
                match &sidecar_digest {
                    Some(digest) => (200, "OK", Vec::new(), format!("{digest}  image.bin\n").into_bytes()),
                    None => (404, "Not Found", Vec::new(), Vec::new()),
                }
            } else {
                (200, "OK", Vec::new(), bytes.clone())
            }
        })
        .await
    }

    /// Builds a catalog entry pointing at `addr`, with or without a sidecar.
    fn release_at(addr: std::net::SocketAddr, with_sidecar: bool) -> FirmwareRelease {
        FirmwareRelease {
            version: "1.5.0".to_string(),
            tag: "v1.5.0".to_string(),
            name: "1.5.0".to_string(),
            notes: String::new(),
            prerelease: false,
            published_at: String::new(),
            image_url: format!("http://{addr}/image.bin"),
            image_size_bytes: 4096,
            sha256_url: with_sidecar.then(|| format!("http://{addr}/image.sha256")),
        }
    }

    #[tokio::test]
    async fn download_image_matching_sidecar_reports_verified_and_streams_progress() {
        // Arrange
        let (bytes, digest) = image_fixture();
        let (addr, _server) = spawn_image_server(bytes.clone(), Some(digest.clone())).await;
        let release = release_at(addr, true);
        let mut progress: Vec<(u64, Option<u64>)> = Vec::new();
        let mut on_progress = |done: u64, total: Option<u64>| progress.push((done, total));

        // Act
        let downloaded = download_image(&release, &mut on_progress).await.unwrap();

        // Assert
        assert_eq!(downloaded.bytes, bytes);
        assert_eq!(downloaded.sha256, digest);
        assert!(downloaded.sha256_verified);
        assert_eq!(progress.last().unwrap().0, bytes.len() as u64);
    }

    #[tokio::test]
    async fn download_image_no_sidecar_succeeds_but_is_not_verified() {
        // Arrange
        let (bytes, digest) = image_fixture();
        let (addr, _server) = spawn_image_server(bytes.clone(), None).await;
        let release = release_at(addr, false);
        let mut on_progress = |_: u64, _: Option<u64>| {};

        // Act
        let downloaded = download_image(&release, &mut on_progress).await.unwrap();

        // Assert
        assert_eq!(downloaded.sha256, digest);
        assert!(!downloaded.sha256_verified);
    }

    #[tokio::test]
    async fn download_image_mismatched_sidecar_is_refused_with_both_digests_named() {
        // Arrange
        let (bytes, digest) = image_fixture();
        let wrong_digest = "b".repeat(64);
        let (addr, _server) = spawn_image_server(bytes, Some(wrong_digest.clone())).await;
        let release = release_at(addr, true);
        let mut on_progress = |_: u64, _: Option<u64>| {};

        // Act
        let err = download_image(&release, &mut on_progress).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Wifi);
        assert!(err.message.contains(&wrong_digest));
        assert!(err.message.contains(&digest));
    }
}
