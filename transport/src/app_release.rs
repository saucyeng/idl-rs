//! The app's own update feed: GitHub Releases of `saucyeng/idl1-app`, one
//! feed per release channel (docs/RELEASING.md, "Channels and in-app update
//! checks").
//!
//! The check is against one channel — the build's own, or the one picked
//! in Settings — and only ever offers a release from it:
//!
//! - **stable** — `vX.Y.Z` tags.
//! - **beta** — `vX.Y.Z-<pre>` tags, plus every stable release (a beta
//!   tester is offered the stable release that follows their beta).
//! - **nightly** — the one moving release `release.yml` replaces on every
//!   push to `main` ([`NIGHTLY_TAG`]).
//!
//! **Newer means a higher build number.** Every CI build's number is the
//! workflow's `github.run_number`, which is also the APK's `versionCode`;
//! Android refuses to install a lower `versionCode` over a higher one, so a
//! build number is the one comparison that holds across channels. Release
//! titles carry it as `(build <n>` (`idl1 nightly (build 20, abc1234)`,
//! `idl1 v0.2.0 (build 31)`). A tagged release published before titles
//! carried it, or a running build that does not know its own number, falls
//! back to semver against the running version.
//!
//! Everything except the fetch is pure and unit-tested.

use std::cmp::Ordering;

use crate::firmware_catalog::{compare_semver, https_client, parse_semver};
use crate::{TransportError, TransportErrorKind};

/// The repository the app's own releases are published from.
pub const APP_REPO: &str = "saucyeng/idl1-app";

/// The nightly channel's moving release tag.
pub const NIGHTLY_TAG: &str = "nightly";

/// The nightly release's Android asset (`.github/scripts/nightly-rename.sh`).
pub const NIGHTLY_APK_ASSET: &str = "idl1-app-nightly-arm64.apk";

/// The Android asset a tagged release of `version` publishes (`release.yml`).
pub fn tagged_apk_asset(version: &str) -> String {
    format!("idl1-app-v{version}-arm64.apk")
}

/// Same reason as the firmware catalog's: an unset `User-Agent` gets a 403.
const USER_AGENT: &str = "idl1-app";

/// Builds a `TransportErrorKind::Wifi` error, the kind the firmware catalog
/// already uses for its GitHub calls.
fn release_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Wifi, message)
}

/// A release channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppChannel {
    /// `vX.Y.Z` tags.
    Stable,
    /// `vX.Y.Z-<pre>` tags and stable releases.
    Beta,
    /// The moving build of `main`.
    Nightly,
}

/// What to check for: the channel, and what the running build is.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CurrentBuild {
    /// The channel to check — the build's own unless Settings picked another.
    pub channel: AppChannel,
    /// The running build's semver version (`tauri.conf.json`'s `version`).
    pub version: String,
    /// The running build's CI build number (`github.run_number`); 0 when unknown.
    pub build: u64,
}

/// A newer release on the checked channel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AppUpdate {
    /// The channel the release was found on.
    pub channel: AppChannel,
    /// What to call it: `"build <n>"` on nightly, the semver version otherwise.
    pub label: String,
    /// The release's build number, when its title carries one.
    pub build: Option<u64>,
    /// The release body (markdown, as published).
    pub notes: String,
    /// Publication timestamp, verbatim from GitHub (RFC 3339); may be empty.
    pub published_at: String,
    /// The release's web page — always present, the fallback when there is
    /// no APK to link to.
    pub page_url: String,
    /// Download URL of the release's arm64 APK, or `None` when it carries
    /// none (still uploading, or the Android job failed).
    pub apk_url: Option<String>,
}

/// Reads the build number out of a release title containing `(build <n>`
/// followed by `,` or `)`. `None` for any other shape.
pub fn parse_build_number(title: &str) -> Option<u64> {
    let rest = &title[title.find("(build ")? + "(build ".len()..];
    let end = rest.find([',', ')'])?;
    rest[..end].trim().parse().ok()
}

fn str_field<'a>(entry: &'a serde_json::Value, key: &str) -> &'a str {
    entry.get(key).and_then(serde_json::Value::as_str).unwrap_or("")
}

fn asset_url(entry: &serde_json::Value, name: &str) -> Option<String> {
    entry
        .get("assets")
        .and_then(serde_json::Value::as_array)?
        .iter()
        .find(|asset| asset.get("name").and_then(serde_json::Value::as_str) == Some(name))?
        .get("browser_download_url")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn parse_json(body: &str) -> Result<serde_json::Value, TransportError> {
    serde_json::from_str(body).map_err(|e| release_error(format!("the app's release feed returned malformed JSON: {e}")))
}

/// Whether a release is newer than `current`: by build number when both
/// sides know theirs, else by semver (`None` when either side's version is
/// not semver — then it is not offered).
fn is_newer(build: Option<u64>, version: Option<&str>, current: &CurrentBuild) -> bool {
    match build {
        Some(build) if current.build > 0 => build > current.build,
        _ => match (version.and_then(parse_semver), parse_semver(&current.version)) {
            (Some(release), Some(running)) => compare_semver(&release, &running) == Ordering::Greater,
            _ => false,
        },
    }
}

/// The nightly release from GitHub's `GET /releases/tags/<NIGHTLY_TAG>`
/// body, if newer than `current`. A body that is not a release object, or
/// whose title has no build number, is an error.
pub fn nightly_update(body: &str, current: &CurrentBuild) -> Result<Option<AppUpdate>, TransportError> {
    let entry = parse_json(body)?;
    let title = str_field(&entry, "name");
    let build = parse_build_number(title)
        .ok_or_else(|| release_error(format!("the nightly release title {title:?} carries no build number")))?;
    // A nightly has no semver of its own; a running build that does not know
    // its number is offered it.
    if current.build > 0 && build <= current.build {
        return Ok(None);
    }
    Ok(Some(AppUpdate {
        channel: AppChannel::Nightly,
        label: format!("build {build}"),
        build: Some(build),
        notes: str_field(&entry, "body").to_string(),
        published_at: str_field(&entry, "published_at").to_string(),
        page_url: str_field(&entry, "html_url").to_string(),
        apk_url: asset_url(&entry, NIGHTLY_APK_ASSET),
    }))
}

/// The newest tagged release `current.channel` admits, from GitHub's
/// `GET /releases` body, if newer than `current`. Drafts, non-semver tags
/// (the nightly tag among them) and, on stable, prereleases are skipped.
pub fn tagged_update(body: &str, current: &CurrentBuild) -> Result<Option<AppUpdate>, TransportError> {
    let parsed = parse_json(body)?;
    let entries = parsed
        .as_array()
        .ok_or_else(|| release_error("the app's release feed returned something other than a JSON array"))?;

    let newest = entries
        .iter()
        .filter(|entry| !entry.get("draft").and_then(serde_json::Value::as_bool).unwrap_or(false))
        .filter_map(|entry| Some((entry, parse_semver(str_field(entry, "tag_name"))?)))
        .filter(|(_, version)| current.channel != AppChannel::Stable || version.prerelease.is_empty())
        .max_by(|(_, a), (_, b)| compare_semver(a, b));
    let Some((entry, _)) = newest else { return Ok(None) };

    let tag = str_field(entry, "tag_name");
    let version = tag.strip_prefix('v').unwrap_or(tag);
    let build = parse_build_number(str_field(entry, "name"));
    if !is_newer(build, Some(version), current) {
        return Ok(None);
    }
    Ok(Some(AppUpdate {
        channel: current.channel,
        label: version.to_string(),
        build,
        notes: str_field(entry, "body").to_string(),
        published_at: str_field(entry, "published_at").to_string(),
        page_url: str_field(entry, "html_url").to_string(),
        apk_url: asset_url(entry, &tagged_apk_asset(version)),
    }))
}

/// Checks `current.channel`'s feed for a release newer than `current`.
/// `Ok(None)` means checked and up to date. Public API, no token.
pub async fn check_app_update(current: &CurrentBuild) -> Result<Option<AppUpdate>, TransportError> {
    let base = format!("https://api.github.com/repos/{APP_REPO}/releases");
    check_app_update_at(&base, current).await
}

/// [`check_app_update`] with the API base spelled out, for the mock-server tests.
async fn check_app_update_at(base: &str, current: &CurrentBuild) -> Result<Option<AppUpdate>, TransportError> {
    match current.channel {
        AppChannel::Nightly => nightly_update(&fetch_text(&format!("{base}/tags/{NIGHTLY_TAG}")).await?, current),
        AppChannel::Stable | AppChannel::Beta => tagged_update(&fetch_text(base).await?, current),
    }
}

async fn fetch_text(url: &str) -> Result<String, TransportError> {
    let response = https_client()?
        .get(url)
        .header(reqwest::header::USER_AGENT, USER_AGENT)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| release_error(format!("could not reach the app's release feed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(release_error(format!("the app's release feed returned status {}", status.as_u16())));
    }
    response.text().await.map_err(|e| release_error(format!("could not read the app's release feed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn nightly_json(build: u64, with_apk: bool) -> String {
        let assets = if with_apk {
            serde_json::json!([
                {"name": "idl1-app-nightly-arm64.apk.sha256", "browser_download_url": "https://example.invalid/apk.sha256"},
                {"name": NIGHTLY_APK_ASSET, "browser_download_url": "https://example.invalid/nightly.apk"}
            ])
        } else {
            serde_json::json!([])
        };
        serde_json::json!({
            "tag_name": "nightly", "name": format!("idl1 nightly (build {build}, 0123abc)"), "body": "Latest change",
            "prerelease": true, "published_at": "2026-10-15T09:00:00Z",
            "html_url": "https://example.invalid/releases/nightly", "assets": assets
        })
        .to_string()
    }

    /// Tagged releases: stable 0.2.0 (build 31), beta 0.3.0-beta.1 (build
    /// 40), a draft 0.4.0, the nightly tag, and 0.1.0-alpha.4 from before
    /// titles carried a build number.
    pub(super) fn tagged_json() -> String {
        let release = |tag: &str, title: &str, prerelease: bool, draft: bool| {
            let version = tag.trim_start_matches('v');
            serde_json::json!({
                "tag_name": tag, "name": title, "body": format!("notes {tag}"),
                "draft": draft, "prerelease": prerelease, "published_at": "2026-10-15T09:00:00Z",
                "html_url": format!("https://example.invalid/releases/{tag}"),
                "assets": [{"name": tagged_apk_asset(version), "browser_download_url": format!("https://example.invalid/{tag}.apk")}]
            })
        };
        serde_json::json!([
            release("nightly", "idl1 nightly (build 45, abc1234)", true, false),
            release("v0.4.0", "idl1 v0.4.0 (build 50)", false, true),
            release("v0.3.0-beta.1", "idl1 v0.3.0-beta.1 (build 40)", true, false),
            release("v0.2.0", "idl1 v0.2.0 (build 31)", false, false),
            release("v0.1.0-alpha.4", "idl1 v0.1.0-alpha.4", true, false),
        ])
        .to_string()
    }

    fn current(channel: AppChannel, version: &str, build: u64) -> CurrentBuild {
        CurrentBuild { channel, version: version.to_string(), build }
    }

    #[test]
    fn parse_build_number_nightly_and_tagged_titles_return_the_number() {
        // Arrange
        let titles = ["idl1 nightly (build 18, bdc3c54)", "idl1 v0.2.0 (build 31)", "idl1 alpha (build 7, x)"];

        // Act
        let builds: Vec<Option<u64>> = titles.iter().map(|t| parse_build_number(t)).collect();

        // Assert
        assert_eq!(builds, vec![Some(18), Some(31), Some(7)]);
    }

    #[test]
    fn parse_build_number_other_shapes_return_none() {
        // Arrange
        let titles = ["idl1 v0.1.0", "idl1 nightly (build x, bdc3c54)", "idl1 nightly (build 18"];

        // Act / Assert
        for title in titles {
            assert_eq!(parse_build_number(title), None, "{title}");
        }
    }

    #[test]
    fn nightly_update_higher_build_offers_it_with_the_apk_url() {
        // Arrange
        let body = nightly_json(19, true);

        // Act
        let update = nightly_update(&body, &current(AppChannel::Nightly, "0.1.0", 18)).unwrap().unwrap();

        // Assert
        assert_eq!(update.label, "build 19");
        assert_eq!(update.build, Some(19));
        assert_eq!(update.apk_url.as_deref(), Some("https://example.invalid/nightly.apk"));
        assert_eq!(update.page_url, "https://example.invalid/releases/nightly");
        assert_eq!(update.notes, "Latest change");
    }

    #[test]
    fn nightly_update_same_or_lower_build_is_up_to_date() {
        // Arrange
        let body = nightly_json(18, true);

        // Act
        let same = nightly_update(&body, &current(AppChannel::Nightly, "0.1.0", 18)).unwrap();
        let older = nightly_update(&body, &current(AppChannel::Nightly, "0.1.0", 25)).unwrap();

        // Assert
        assert_eq!(same, None);
        assert_eq!(older, None);
    }

    #[test]
    fn nightly_update_no_apk_yet_still_offers_the_release_page() {
        // Arrange
        let body = nightly_json(19, false);

        // Act
        let update = nightly_update(&body, &current(AppChannel::Nightly, "0.1.0", 18)).unwrap().unwrap();

        // Assert
        assert_eq!(update.apk_url, None);
        assert!(!update.page_url.is_empty());
    }

    #[test]
    fn nightly_update_untitled_or_not_an_object_is_a_typed_error() {
        // Arrange
        let untitled = r#"{"name": "idl1 v0.1.0"}"#;

        // Act
        let title_err = nightly_update(untitled, &current(AppChannel::Nightly, "0.1.0", 1)).unwrap_err();
        let json_err = nightly_update("not json", &current(AppChannel::Nightly, "0.1.0", 1)).unwrap_err();

        // Assert
        assert_eq!(title_err.kind, TransportErrorKind::Wifi);
        assert_eq!(json_err.kind, TransportErrorKind::Wifi);
    }

    #[test]
    fn tagged_update_stable_skips_prereleases_drafts_and_the_nightly_tag() {
        // Arrange
        let body = tagged_json();

        // Act
        let update = tagged_update(&body, &current(AppChannel::Stable, "0.1.0", 20)).unwrap().unwrap();

        // Assert
        assert_eq!(update.label, "0.2.0");
        assert_eq!(update.build, Some(31));
        assert_eq!(update.channel, AppChannel::Stable);
        assert_eq!(update.apk_url.as_deref(), Some("https://example.invalid/v0.2.0.apk"));
    }

    #[test]
    fn tagged_update_beta_takes_the_newest_prerelease() {
        // Arrange
        let body = tagged_json();

        // Act
        let update = tagged_update(&body, &current(AppChannel::Beta, "0.2.0", 31)).unwrap().unwrap();

        // Assert
        assert_eq!(update.label, "0.3.0-beta.1");
        assert_eq!(update.notes, "notes v0.3.0-beta.1");
    }

    #[test]
    fn tagged_update_nightly_build_switched_to_stable_is_offered_nothing_it_cannot_install() {
        // Arrange — a nightly at build 35 with Settings set to stable: v0.2.0
        // is build 31, and Android refuses the lower versionCode.
        let body = tagged_json();

        // Act
        let update = tagged_update(&body, &current(AppChannel::Stable, "0.1.0", 35)).unwrap();

        // Assert
        assert_eq!(update, None);
    }

    #[test]
    fn tagged_update_unknown_build_number_falls_back_to_semver() {
        // Arrange
        let body = tagged_json();

        // Act
        let older_version = tagged_update(&body, &current(AppChannel::Stable, "0.1.0", 0)).unwrap();
        let same_version = tagged_update(&body, &current(AppChannel::Stable, "0.2.0", 0)).unwrap();

        // Assert
        assert_eq!(older_version.map(|u| u.label), Some("0.2.0".to_string()));
        assert_eq!(same_version, None);
    }

    #[test]
    fn is_newer_untitled_release_compares_semver_even_when_the_build_is_known() {
        // Arrange
        let running = current(AppChannel::Beta, "0.1.0-alpha.3", 18);

        // Act / Assert
        assert!(is_newer(None, Some("0.1.0-alpha.4"), &running));
        assert!(!is_newer(None, Some("0.1.0-alpha.2"), &running));
        assert!(!is_newer(None, Some("nightly"), &running));
    }
}

/// [`check_app_update_at`] against the WiFi tests' mock HTTP server, never
/// github.com.
#[cfg(test)]
mod integration {
    use super::*;
    use crate::wifi_transport::integration::spawn_mock_server;

    #[tokio::test]
    async fn check_app_update_at_routes_each_channel_to_its_feed() {
        // Arrange
        let (addr, _server) = spawn_mock_server(|path, _headers, _request_body| {
            let body = if path.ends_with("/tags/nightly") { super::tests::nightly_json(30, true) } else { super::tests::tagged_json() };
            (200, "OK", vec![("Content-Type".to_string(), "application/json".to_string())], body.into_bytes())
        })
        .await;
        let base = format!("http://{addr}/releases");
        let nightly = CurrentBuild { channel: AppChannel::Nightly, version: "0.1.0".to_string(), build: 18 };
        let stable = CurrentBuild { channel: AppChannel::Stable, version: "0.1.0".to_string(), build: 18 };

        // Act
        let nightly_update = check_app_update_at(&base, &nightly).await.unwrap().unwrap();
        let stable_update = check_app_update_at(&base, &stable).await.unwrap().unwrap();

        // Assert
        assert_eq!(nightly_update.label, "build 30");
        assert_eq!(stable_update.label, "0.2.0");
    }

    #[tokio::test]
    async fn check_app_update_at_404_is_a_typed_error() {
        // Arrange
        let (addr, _server) =
            spawn_mock_server(|_path, _headers, _request_body| (404, "Not Found", Vec::new(), Vec::new())).await;
        let nightly = CurrentBuild { channel: AppChannel::Nightly, version: "0.1.0".to_string(), build: 18 };

        // Act
        let err = check_app_update_at(&format!("http://{addr}/releases"), &nightly).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Wifi);
        assert!(err.message.contains("404"), "{}", err.message);
    }
}
