//! The app's own update feed: GitHub Releases of `saucyeng/idl1-app`, one
//! feed per release channel (docs/RELEASING.md, "Channels and in-app update
//! checks").
//!
//! A build knows its channel from build time and is only ever offered a
//! newer release *on that channel*:
//!
//! - **stable** — `vX.Y.Z` tags; compared by semver.
//! - **beta** — `vX.Y.Z-<pre>` tags, plus every stable release (a beta
//!   tester is offered the stable release that follows their beta);
//!   compared by semver.
//! - **nightly** — the one moving release `release.yml` replaces on every
//!   push to `main` ([`NIGHTLY_TAG`]); compared by CI build number
//!   (`github.run_number`, also the APK's `versionCode`). Its title
//!   `idl1 alpha (build <n>, <sha7>)` is the wire format for the number.
//!
//! Everything except the two `fetch` functions is pure and unit-tested.

use std::cmp::Ordering;

use crate::firmware_catalog::{compare_semver, https_client, parse_semver};
use crate::{TransportError, TransportErrorKind};

/// The repository the app's own releases are published from.
pub const APP_REPO: &str = "saucyeng/idl1-app";

/// The nightly channel's moving release tag.
pub const NIGHTLY_TAG: &str = "alpha";

/// The nightly release's Android asset (`.github/scripts/alpha-rename.sh`).
pub const NIGHTLY_APK_ASSET: &str = "idl1-app-alpha-arm64.apk";

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

/// Which release channel a build belongs to, fixed when it is built.
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

/// What the running build is, as the update check compares it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CurrentBuild {
    /// The channel it was built for.
    pub channel: AppChannel,
    /// Its semver version (`tauri.conf.json`'s `version`).
    pub version: String,
    /// Its CI build number (`github.run_number`); 0 when unknown.
    pub build: u64,
}

/// A newer release on the build's channel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AppUpdate {
    /// The channel the release was found on (the build's own).
    pub channel: AppChannel,
    /// What to call it: the semver version, or `"build <n>"` on nightly.
    pub label: String,
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

/// Reads `(build, commit)` out of a nightly title of the form
/// `idl1 alpha (build <n>, <sha7>)`. `None` for any other shape.
pub fn parse_nightly_title(title: &str) -> Option<(u64, String)> {
    let rest = &title[title.find("(build ")? + "(build ".len()..];
    let (build, rest) = rest.split_once(',')?;
    let build = build.trim().parse().ok()?;
    let commit = rest.split_once(')')?.0.trim();
    if commit.is_empty() {
        return None;
    }
    Some((build, commit.to_string()))
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

/// The nightly release from GitHub's `GET /releases/tags/<NIGHTLY_TAG>`
/// body, if its build number is above `current.build`. A body that is not a
/// release object, or whose title has no build number, is an error.
pub fn nightly_update(body: &str, current: &CurrentBuild) -> Result<Option<AppUpdate>, TransportError> {
    let entry = parse_json(body)?;
    let title = str_field(&entry, "name");
    let (build, _commit) = parse_nightly_title(title)
        .ok_or_else(|| release_error(format!("the nightly release title {title:?} carries no build number")))?;
    if build <= current.build {
        return Ok(None);
    }
    Ok(Some(AppUpdate {
        channel: AppChannel::Nightly,
        label: format!("build {build}"),
        notes: str_field(&entry, "body").to_string(),
        published_at: str_field(&entry, "published_at").to_string(),
        page_url: str_field(&entry, "html_url").to_string(),
        apk_url: asset_url(&entry, NIGHTLY_APK_ASSET),
    }))
}

/// The newest tagged release `current.channel` admits, from GitHub's
/// `GET /releases` body, if its semver is above `current.version`. Drafts,
/// non-semver tags (the nightly tag among them) and, on stable, prereleases
/// are skipped. A `current.version` that is not semver is an error: there is
/// nothing to compare against.
pub fn tagged_update(body: &str, current: &CurrentBuild) -> Result<Option<AppUpdate>, TransportError> {
    let parsed = parse_json(body)?;
    let entries = parsed
        .as_array()
        .ok_or_else(|| release_error("the app's release feed returned something other than a JSON array"))?;
    let running = parse_semver(&current.version)
        .ok_or_else(|| release_error(format!("this build's version {:?} is not semver", current.version)))?;

    let newest = entries
        .iter()
        .filter(|entry| !entry.get("draft").and_then(serde_json::Value::as_bool).unwrap_or(false))
        .filter_map(|entry| Some((entry, parse_semver(str_field(entry, "tag_name"))?)))
        .filter(|(_, version)| current.channel != AppChannel::Stable || version.prerelease.is_empty())
        .max_by(|(_, a), (_, b)| compare_semver(a, b));
    let Some((entry, version)) = newest else { return Ok(None) };
    if compare_semver(&version, &running) != Ordering::Greater {
        return Ok(None);
    }

    let tag = str_field(entry, "tag_name");
    let version_text = tag.strip_prefix('v').unwrap_or(tag);
    Ok(Some(AppUpdate {
        channel: current.channel,
        label: version_text.to_string(),
        notes: str_field(entry, "body").to_string(),
        published_at: str_field(entry, "published_at").to_string(),
        page_url: str_field(entry, "html_url").to_string(),
        apk_url: asset_url(entry, &tagged_apk_asset(version_text)),
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
                {"name": "idl1-app-alpha-arm64.apk.sha256", "browser_download_url": "https://example.invalid/apk.sha256"},
                {"name": NIGHTLY_APK_ASSET, "browser_download_url": "https://example.invalid/nightly.apk"}
            ])
        } else {
            serde_json::json!([])
        };
        serde_json::json!({
            "tag_name": "alpha", "name": format!("idl1 alpha (build {build}, 0123abc)"), "body": "Latest change",
            "prerelease": true, "published_at": "2026-10-15T09:00:00Z",
            "html_url": "https://example.invalid/releases/alpha", "assets": assets
        })
        .to_string()
    }

    /// Tagged releases: stable 0.2.0, beta 0.3.0-beta.1, a draft 0.4.0, the
    /// nightly tag, and the older 0.1.0-alpha.4 prerelease.
    pub(super) fn tagged_json() -> String {
        let release = |tag: &str, prerelease: bool, draft: bool| {
            let version = tag.trim_start_matches('v');
            serde_json::json!({
                "tag_name": tag, "name": format!("idl1 {tag}"), "body": format!("notes {tag}"),
                "draft": draft, "prerelease": prerelease, "published_at": "2026-10-15T09:00:00Z",
                "html_url": format!("https://example.invalid/releases/{tag}"),
                "assets": [{"name": tagged_apk_asset(version), "browser_download_url": format!("https://example.invalid/{tag}.apk")}]
            })
        };
        serde_json::json!([
            release("alpha", true, false),
            release("v0.4.0", false, true),
            release("v0.3.0-beta.1", true, false),
            release("v0.2.0", false, false),
            release("v0.1.0-alpha.4", true, false),
        ])
        .to_string()
    }

    fn current(channel: AppChannel, version: &str, build: u64) -> CurrentBuild {
        CurrentBuild { channel, version: version.to_string(), build }
    }

    #[test]
    fn parse_nightly_title_release_yml_title_returns_build_and_commit() {
        // Arrange
        let title = "idl1 alpha (build 18, bdc3c54)";

        // Act
        let parsed = parse_nightly_title(title);

        // Assert
        assert_eq!(parsed, Some((18, "bdc3c54".to_string())));
    }

    #[test]
    fn parse_nightly_title_other_shapes_return_none() {
        // Arrange
        let titles = ["idl1 v0.1.0", "idl1 alpha (build x, bdc3c54)", "idl1 alpha (build 18, )", "idl1 alpha (build 18"];

        // Act / Assert
        for title in titles {
            assert_eq!(parse_nightly_title(title), None, "{title}");
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
        assert_eq!(update.apk_url.as_deref(), Some("https://example.invalid/nightly.apk"));
        assert_eq!(update.page_url, "https://example.invalid/releases/alpha");
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
        let update = tagged_update(&body, &current(AppChannel::Stable, "0.1.0", 0)).unwrap().unwrap();

        // Assert
        assert_eq!(update.label, "0.2.0");
        assert_eq!(update.channel, AppChannel::Stable);
        assert_eq!(update.apk_url.as_deref(), Some("https://example.invalid/v0.2.0.apk"));
    }

    #[test]
    fn tagged_update_beta_takes_the_newest_prerelease() {
        // Arrange
        let body = tagged_json();

        // Act
        let update = tagged_update(&body, &current(AppChannel::Beta, "0.2.0", 0)).unwrap().unwrap();

        // Assert
        assert_eq!(update.label, "0.3.0-beta.1");
        assert_eq!(update.notes, "notes v0.3.0-beta.1");
    }

    #[test]
    fn tagged_update_beta_ahead_of_every_stable_is_offered_nothing_older() {
        // Arrange
        let body = tagged_json();

        // Act
        let stable_up_to_date = tagged_update(&body, &current(AppChannel::Stable, "0.2.0", 0)).unwrap();
        let beta_up_to_date = tagged_update(&body, &current(AppChannel::Beta, "0.3.0-beta.1", 0)).unwrap();

        // Assert
        assert_eq!(stable_up_to_date, None);
        assert_eq!(beta_up_to_date, None);
    }

    #[test]
    fn tagged_update_non_semver_running_version_is_a_typed_error() {
        // Arrange
        let body = tagged_json();

        // Act
        let err = tagged_update(&body, &current(AppChannel::Stable, "dev", 0)).unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Wifi);
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
            let body = if path.ends_with("/tags/alpha") { super::tests::nightly_json(30, true) } else { super::tests::tagged_json() };
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
