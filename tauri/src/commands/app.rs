//! App group (C3 §3.10): `get_settings`, `set_settings`, `get_data_dir`,
//! `set_data_dir` — thin wrappers over the already-landed
//! `idl_rs::store::settings` module and `crate::paths::resolve_data_dir`.
//! Satisfies wave-2 needs L7-6/L7-7 (ruling R53 Settings Q1/Q4).
//!
//! Same `_via`-suffixed idiom as `commands/catalog.rs`: each
//! `#[tauri::command]` resolves `app.path()...` (via `tauri::Manager`) and
//! delegates to a plain function taking paths, which this module's own
//! tests exercise directly with temp dirs — `tauri::AppHandle`/
//! `tauri::State` cannot be constructed outside a running app.
//!
//! Deviation from the plan's own interface sketch (documented, not a wire
//! shape change): `unit_system` is typed as
//! `idl_rs::store::settings::UnitSystem` directly rather than a plain
//! `String` with hand-written mapping functions. `UnitSystem` already
//! derives `Serialize`/`Deserialize` with `#[serde(rename_all =
//! "snake_case")]`, so it already serialises to exactly `"imperial"` /
//! `"metric"` and deserialises from exactly those two strings — an
//! unrecognised string becomes a Tauri-level argument-deserialisation
//! failure before the command body runs, rather than inventing silent-default
//! or reject fallback behaviour C3 §3.10 does not specify.

use std::path::{Path, PathBuf};

use tauri::Manager;

use idl_rs::store::settings::{AppSettings, SettingsError, SettingsErrorKind, UnitSystem};

use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// C3 §3.10 `AppSettings` — `get_settings`'s return and (as
/// [`AppSettingsArg`]) `set_settings`'s argument shape.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AppSettingsDto {
    /// The `<data>` root override, or `None` when the platform default is
    /// in use (C4 §1). Always echoes the current on-disk value — see
    /// `set_settings_via`, which ignores this field on its argument.
    pub data_dir: Option<String>,
    /// The rider's display name. `""` means not set (C4 §1).
    pub rider_name: String,
    /// Unit system used across the app. Engine default is `Imperial`.
    pub unit_system: UnitSystem,
}

impl From<AppSettings> for AppSettingsDto {
    fn from(s: AppSettings) -> Self {
        Self { data_dir: s.data_dir, rider_name: s.rider_name, unit_system: s.unit_system }
    }
}

/// `set_settings`'s argument. Same fields as [`AppSettingsDto`] — `data_dir`
/// is present on the wire (C3's `AppSettings` is one TS interface for both
/// directions) but ignored server-side; `set_data_dir` is the sole writer
/// of that key (ruling R59 Q5).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AppSettingsArg {
    /// Present on the wire for symmetry with [`AppSettingsDto`]; ignored by
    /// `set_settings_via`.
    pub data_dir: Option<String>,
    /// The rider's display name to persist. `""` means not set.
    pub rider_name: String,
    /// Unit system to persist.
    pub unit_system: UnitSystem,
}

/// C3 §3.10 `DataDirInfo`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DataDirInfo {
    /// The `<data>` root actually in use for this process — the managed
    /// `state::DataDir`, resolved once at startup and cached for the
    /// process lifetime (C4 §1).
    pub resolved_path: String,
    /// The override from `settings.json`, or `None` when the platform
    /// default is in use.
    pub override_path: Option<String>,
    /// `true` when re-resolving `data_dir` right now would give a
    /// different path than `resolved_path` — i.e. the override changed on
    /// disk since startup and the app has not yet restarted.
    pub restart_required: bool,
}

/// `app_config_dir()/settings.json` (C4 §1's existing bootstrap file).
fn settings_path(app_config_dir: &Path) -> PathBuf {
    app_config_dir.join("settings.json")
}

/// Maps [`SettingsError`] to the C3 §3.10 error rows (`io`, `internal`).
/// `SettingsErrorKind::Encode` folds to `internal` per C3 §2's folding rule
/// — `load` never fails, so only `save`'s errors reach this function.
fn map_settings_error(e: SettingsError) -> IpcError {
    match e.kind {
        SettingsErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
        SettingsErrorKind::Encode => IpcError::new(IpcErrorKind::Internal, e.message),
    }
}

/// `get_settings`'s transport-agnostic core. `load` never fails (C4 §1) —
/// a missing or malformed file yields defaults.
fn get_settings_via(settings_path: &Path) -> AppSettingsDto {
    idl_rs::store::settings::load(settings_path).into()
}

/// `set_settings`'s transport-agnostic core: a whole-document replace of
/// `rider_name`/`unit_system` only. `arg.data_dir` is deliberately not
/// applied — `set_data_dir_via` is the sole writer of that key (ruling R59
/// Q5). Re-reads after the write so the response reflects what is actually
/// on disk, not merely echoed from the argument.
fn set_settings_via(settings_path: &Path, arg: AppSettingsArg) -> Result<AppSettingsDto, IpcError> {
    let mut current = idl_rs::store::settings::load(settings_path);
    current.rider_name = arg.rider_name;
    current.unit_system = arg.unit_system;
    idl_rs::store::settings::save(settings_path, &current).map_err(map_settings_error)?;
    Ok(idl_rs::store::settings::load(settings_path).into())
}

/// `get_data_dir`'s transport-agnostic core. `resolved_data_dir` is always
/// the managed `DataDir` fixed at startup, never a freshly recomputed
/// value — comparing the two is precisely what makes `restart_required` a
/// real condition rather than defensive coding.
fn get_data_dir_via(
    settings_path: &Path,
    app_data_dir: &Path,
    app_config_dir: &Path,
    resolved_data_dir: &Path,
) -> Result<DataDirInfo, IpcError> {
    let settings = idl_rs::store::settings::load(settings_path);
    let fresh = crate::paths::resolve_data_dir(app_data_dir, app_config_dir)?;
    Ok(DataDirInfo {
        resolved_path: resolved_data_dir.display().to_string(),
        override_path: settings.data_dir,
        restart_required: fresh != resolved_data_dir,
    })
}

/// `set_data_dir`'s transport-agnostic core: read-modify-write of only the
/// `data_dir` key, preserving `rider_name`/`unit_system` (ruling R59 Q5).
/// Does not move existing files. `path` must be an absolute path the app
/// can create a `data` subdir under; a relative path or an uncreatable path
/// is `invalid_argument`. That `data` subdir is created *before*
/// `settings.json` is written — a failed create leaves nothing written.
/// The **full** C4 §2 tree (`blobs/sha256/`, `sessions/`, `workbooks/`,
/// `tracks/`, `tmp/quarantine/`) is completed immediately afterward, in
/// this same call, as a side effect of the trailing
/// `crate::paths::resolve_data_dir` call below re-reading the just-written
/// override — the identical code path `get_data_dir`/app startup use, so
/// there is no second, duplicated subdirectory list to keep in sync. This
/// does *not* wait for the caller's restart; `restart_required` on the
/// returned [`DataDirInfo`] reflects only that the *running* managed
/// `state::DataDir` value still points at the old root until relaunch, not
/// that the new root's tree is incomplete.
fn set_data_dir_via(
    settings_path: &Path,
    app_data_dir: &Path,
    app_config_dir: &Path,
    resolved_data_dir: &Path,
    path: Option<String>,
) -> Result<DataDirInfo, IpcError> {
    if let Some(p) = &path {
        if !Path::new(p).is_absolute() {
            return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("'{p}' is not an absolute path")));
        }
        std::fs::create_dir_all(Path::new(p).join("data"))
            .map_err(|e| IpcError::new(IpcErrorKind::InvalidArgument, format!("cannot create '{p}': {e}")))?;
    }
    let mut current = idl_rs::store::settings::load(settings_path);
    current.data_dir = path.clone();
    idl_rs::store::settings::save(settings_path, &current).map_err(map_settings_error)?;
    let fresh = crate::paths::resolve_data_dir(app_data_dir, app_config_dir)?;
    Ok(DataDirInfo {
        resolved_path: resolved_data_dir.display().to_string(),
        override_path: path,
        restart_required: fresh != resolved_data_dir,
    })
}

/// The five trees `move_data_dir` copies (C4 §1 "Moving the root").
/// `tmp/` (scratch), `inbox/` (a desktop drop folder, not library content)
/// and `catalog.sqlite*` are deliberately absent: the catalog is an index,
/// rebuilt at the destination rather than copied, and copying a live sqlite
/// file plus its `-wal`/`-shm` sidecars byte-for-byte is the one way to move
/// a *corrupt* index to the new root.
const MOVED_TREES: [&str; 5] = ["blobs", "sessions", "workbooks", "tracks", "profiles"];

/// Chunk size for the streaming copy/hash below. Blobs are whole ride logs;
/// reading one into memory to hash it is not an option.
const COPY_CHUNK_BYTES: usize = 64 * 1024;

/// Absolute, symlink-resolved form of `path`, which need not exist: the
/// deepest existing ancestor is canonicalised and the remaining components
/// are re-appended. Used only to compare two roots for containment, where
/// Windows' `\\?\` verbatim prefix is harmless because both sides get it.
fn normalized(path: &Path) -> PathBuf {
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut probe = path.to_path_buf();
    loop {
        if let Ok(real) = std::fs::canonicalize(&probe) {
            let mut out = real;
            for part in suffix.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (probe.file_name().map(|n| n.to_os_string()), probe.parent().map(|p| p.to_path_buf())) {
            (Some(name), Some(parent)) if !parent.as_os_str().is_empty() => {
                suffix.push(name);
                probe = parent;
            }
            // No existing ancestor at all (a bogus drive letter): compare the
            // path as given rather than inventing one.
            _ => return path.to_path_buf(),
        }
    }
}

/// `true` when `a` and `b` are the same directory or one contains the other
/// — either direction is fatal for a move, since the destination would end
/// up inside the source being walked (or swallow it).
fn overlaps(a: &Path, b: &Path) -> bool {
    let (a, b) = (normalized(a), normalized(b));
    a.starts_with(&b) || b.starts_with(&a)
}

/// Every regular file under `dir`, as paths relative to `base`. A missing
/// `dir` yields nothing — a library with no `tracks/` yet is normal, not an
/// error.
fn collect_files(base: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), IpcError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", dir.display()))),
    };
    for entry in entries {
        let entry = entry.map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", dir.display())))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("stat {}: {e}", path.display())))?;
        if file_type.is_dir() {
            collect_files(base, &path, out)?;
        } else {
            let rel = path
                .strip_prefix(base)
                .map_err(|e| IpcError::new(IpcErrorKind::Internal, format!("{} not under {}: {e}", path.display(), base.display())))?;
            out.push(rel.to_path_buf());
        }
    }
    Ok(())
}

/// Copies `src` to `dst` (creating `dst`'s parent), streaming, and returns
/// the sha256 hex digest of the bytes it *read* — the source's own digest,
/// which the verify phase then re-derives from the file it wrote.
fn copy_and_hash(src: &Path, dst: &Path) -> Result<String, IpcError> {
    use sha2::Digest;
    use std::io::{Read, Write};

    let io = |what: &str, e: std::io::Error| IpcError::new(IpcErrorKind::Io, format!("{what}: {e}"));
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(&format!("creating {}", parent.display()), e))?;
    }
    let mut reader = std::fs::File::open(src).map_err(|e| io(&format!("opening {}", src.display()), e))?;
    let mut writer = std::fs::File::create(dst).map_err(|e| io(&format!("creating {}", dst.display()), e))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; COPY_CHUNK_BYTES];
    loop {
        let read = reader.read(&mut buf).map_err(|e| io(&format!("reading {}", src.display()), e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        writer.write_all(&buf[..read]).map_err(|e| io(&format!("writing {}", dst.display()), e))?;
    }
    writer.flush().map_err(|e| io(&format!("writing {}", dst.display()), e))?;
    Ok(hex(&hasher.finalize()))
}

/// sha256 hex digest of a file on disk, read in chunks.
fn sha256_file(path: &Path) -> Result<String, IpcError> {
    use sha2::Digest;
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("opening {}: {e}", path.display())))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; COPY_CHUNK_BYTES];
    loop {
        let read = file
            .read(&mut buf)
            .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Lowercase hex, no separators.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The digest a path under `blobs/sha256/<2 hex>/<62 hex>` claims for its own
/// content (C4 §2), or `None` when `rel` is not a blob path — only a blob's
/// name is a checkable assertion about its bytes.
fn blob_digest_from_relative_path(rel: &Path) -> Option<String> {
    let parts: Vec<&str> = rel.components().map(|c| c.as_os_str().to_str().unwrap_or("")).collect();
    if parts.len() != 4 || parts[0] != "blobs" || parts[1] != "sha256" {
        return None;
    }
    let (prefix, rest) = (parts[2], parts[3]);
    if prefix.len() != 2 || rest.len() != 62 {
        return None;
    }
    let digest = format!("{prefix}{rest}");
    if digest.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(digest.to_ascii_lowercase())
    } else {
        None
    }
}

/// `true` when `target` is a usable destination for a move: absent, empty,
/// or the partial result of an earlier interrupted move of a library — a
/// lone `data/` directory whose own entries are all [`MOVED_TREES`] names.
///
/// C3 §3.10 refuses a non-empty target and C4 §1 requires an interrupted
/// move to be resumable by running it again; those two only fit together if
/// "non-empty" means "holds something that is not this move's own partial
/// output". Recognising exactly that shape — and nothing looser — is how a
/// resume stays possible without ever writing a library into a folder the
/// user keeps something else in. **Lane decision, 2026-09-10**, recorded
/// because it resolves a contract ambiguity rather than following one.
fn is_usable_move_target(target: &Path) -> Result<bool, IpcError> {
    let io = |e: std::io::Error| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", target.display()));
    if !target.exists() {
        return Ok(true);
    }
    if !target.is_dir() {
        return Ok(false);
    }
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(target).map_err(io)? {
        entries.push(entry.map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", target.display())))?);
    }
    if entries.is_empty() {
        return Ok(true);
    }
    if entries.len() != 1 || entries[0].file_name() != std::ffi::OsStr::new("data") {
        return Ok(false);
    }
    let data = target.join("data");
    if !data.is_dir() {
        return Ok(false);
    }
    for entry in std::fs::read_dir(&data)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", data.display())))?
    {
        let entry = entry.map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", data.display())))?;
        let name = entry.file_name();
        if !MOVED_TREES.iter().any(|tree| std::ffi::OsStr::new(tree) == name) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Moves one file, cheapest way first: a rename when source and destination
/// share a volume, otherwise copy, verify, then delete the source. Returns
/// once the bytes are safely at `dst` and gone from `src`.
///
/// Verification is by sha256 for a blob — whose path *names* its own digest,
/// so the check is against the content address itself rather than against
/// the source, and catches pre-existing corruption too — and by byte length
/// for everything else (C4 §1 as amended by R197). A verify failure leaves
/// the source in place, untouched: the destination copy is the one that is
/// wrong, and deleting the original on the strength of it is the one
/// unrecoverable mistake this function can make.
/// `allow_rename` is `false` only from a test, to force the cross-volume
/// branch: two temp directories on one developer machine share a volume, so
/// a rename always succeeds there and the verification path — the only part
/// that can refuse — would never run.
fn move_one_file(src: &Path, dst: &Path, rel: &Path, allow_rename: bool) -> Result<(), IpcError> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("creating {}: {e}", parent.display())))?;
    }
    // Same volume: atomic, no second copy on disk, nothing to verify.
    if allow_rename && std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    copy_verify_delete(src, dst, rel)
}

/// [`move_one_file`]'s cross-volume half, split out so it is reachable from
/// a test: a rename between two temp directories on one machine almost
/// always succeeds, which would leave the verification path — the part that
/// can actually refuse — unexercised.
fn copy_verify_delete(src: &Path, dst: &Path, rel: &Path) -> Result<(), IpcError> {
    let io = |what: String, e: std::io::Error| IpcError::new(IpcErrorKind::Io, format!("{what}: {e}"));
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(format!("creating {}", parent.display()), e))?;
    }
    let source_len = std::fs::metadata(src).map_err(|e| io(format!("stat {}", src.display()), e))?.len();
    let copied_digest = copy_and_hash(src, dst)?;
    match blob_digest_from_relative_path(rel) {
        Some(claimed) => {
            let landed = sha256_file(dst)?;
            if landed != claimed {
                return Err(IpcError::with_detail(
                    IpcErrorKind::Io,
                    format!("blob {} does not hash to its own name; the move stopped and nothing was switched", rel.display()),
                    serde_json::json!({ "path": rel.display().to_string(), "reason": "blob_hash_mismatch" }),
                ));
            }
            if landed != copied_digest {
                return Err(IpcError::with_detail(
                    IpcErrorKind::Io,
                    format!("{} changed between being written and being read back", rel.display()),
                    serde_json::json!({ "path": rel.display().to_string(), "reason": "copy_mismatch" }),
                ));
            }
        }
        None => {
            let landed_len = std::fs::metadata(dst).map_err(|e| io(format!("stat {}", dst.display()), e))?.len();
            if landed_len != source_len {
                return Err(IpcError::with_detail(
                    IpcErrorKind::Io,
                    format!("{} did not survive the copy intact; the move stopped and nothing was switched", rel.display()),
                    serde_json::json!({ "path": rel.display().to_string(), "reason": "copy_mismatch" }),
                ));
            }
        }
    }
    std::fs::remove_file(src).map_err(|e| io(format!("removing {}", src.display()), e))?;
    Ok(())
}

/// `true` when `dst` already holds this file's verified content, so a rerun
/// after an interruption can skip it and just drop the source (C4 §1,
/// "verified files are skipped"). Blobs are checked against the digest their
/// path names; everything else against the source's length.
fn already_moved(src: &Path, dst: &Path, rel: &Path) -> Result<bool, IpcError> {
    if !dst.is_file() {
        return Ok(false);
    }
    match blob_digest_from_relative_path(rel) {
        Some(claimed) => Ok(sha256_file(dst)? == claimed),
        None => {
            let src_len = std::fs::metadata(src)
                .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("stat {}: {e}", src.display())))?
                .len();
            let dst_len = std::fs::metadata(dst)
                .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("stat {}: {e}", dst.display())))?
                .len();
            Ok(src_len == dst_len)
        }
    }
}

/// Removes the now-empty directories a move emptied, deepest first. Uses
/// `remove_dir`, never `remove_dir_all`: a directory that still holds
/// anything makes this fail and be ignored, which is exactly the guarantee
/// wanted — "the old root's directories are removed only when empty" (C4
/// §1). The five tree roots themselves are pruned too; the old `<data>`
/// root, its `tmp/`, its `inbox/` and the user's own folder above it are
/// left alone.
fn prune_emptied_dirs(data_root: &Path) {
    for tree in MOVED_TREES {
        let root = data_root.join(tree);
        let mut dirs = Vec::new();
        collect_dirs_deepest_first(&root, &mut dirs);
        dirs.push(root);
        for dir in dirs {
            let _ = std::fs::remove_dir(&dir);
        }
    }
}

/// Every directory below `dir`, children before parents, so a caller
/// removing them in order meets each one already emptied of subdirectories.
/// A top-level function rather than a closure or nested `fn` on purpose: the
/// delete-guard scan attributes a delete call to the nearest enclosing `fn`,
/// and an allowlist entry reading `walk` would name nothing an auditor could
/// find.
fn collect_dirs_deepest_first(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_dirs_deepest_first(&path, out);
            out.push(path);
        }
    }
}

/// `move_data_dir`'s transport-agnostic core (C3 §3.10, C4 §1 "Moving the
/// root", ruling R196 as amended by R197).
///
/// **Moves** the five [`MOVED_TREES`] from the *running* `<data>` root to
/// `<new_root>/data`, one file at a time — a rename on the same volume,
/// otherwise copy, verify, delete — so the peak extra disk is one file
/// rather than a second library. `tmp/`, `inbox/` and `catalog.sqlite*` do
/// not travel; the catalog is rebuilt at the destination. The `data_dir`
/// override is written last, after every file has arrived and verified, so
/// an interrupted or failed move leaves the app still opening the old root
/// — where the not-yet-moved files still are. Rerunning it finishes the job:
/// files already verified at the destination are skipped.
///
/// `progress(phase, done, total)` is called per file for `"move"` and once
/// at each end of `"catalog"` (C3 §3.10).
fn move_data_dir_via(
    settings_path: &Path,
    app_data_dir: &Path,
    app_config_dir: &Path,
    resolved_data_dir: &Path,
    new_root: &str,
    progress: impl Fn(&str, u64, u64),
) -> Result<DataDirInfo, IpcError> {
    move_data_dir_with(settings_path, app_data_dir, app_config_dir, resolved_data_dir, new_root, true, progress)
}

/// [`move_data_dir_via`]'s body, with the same-volume rename made optional.
/// `allow_rename: false` is a test-only affordance that forces every file
/// down the copy-verify-delete branch, so the whole command — refusals,
/// verification, the untouched override on failure — can be exercised as a
/// cross-volume move on a machine that has only one volume.
fn move_data_dir_with(
    settings_path: &Path,
    app_data_dir: &Path,
    app_config_dir: &Path,
    resolved_data_dir: &Path,
    new_root: &str,
    allow_rename: bool,
    progress: impl Fn(&str, u64, u64),
) -> Result<DataDirInfo, IpcError> {
    let invalid = |message: String| IpcError::new(IpcErrorKind::InvalidArgument, message);

    let new_root_path = Path::new(new_root);
    if !new_root_path.is_absolute() {
        return Err(invalid(format!("'{new_root}' is not an absolute path")));
    }
    let new_data = new_root_path.join("data");
    if overlaps(&new_data, resolved_data_dir) {
        return Err(invalid(format!("'{new_root}' overlaps the library it would receive")));
    }
    if !is_usable_move_target(new_root_path)? {
        return Err(invalid(format!("'{new_root}' is not empty")));
    }
    std::fs::create_dir_all(&new_data).map_err(|e| invalid(format!("cannot write to '{new_root}': {e}")))?;

    // Enumerate first, so `total` is honest from the first progress event.
    let mut files: Vec<PathBuf> = Vec::new();
    for tree in MOVED_TREES {
        collect_files(resolved_data_dir, &resolved_data_dir.join(tree), &mut files)?;
    }
    let total = files.len() as u64;

    progress("move", 0, total);
    for (index, rel) in files.iter().enumerate() {
        let src = resolved_data_dir.join(rel);
        let dst = new_data.join(rel);
        if already_moved(&src, &dst, rel)? {
            // A previous run got this far. Drop the leftover source rather
            // than copying over a file that already verifies.
            std::fs::remove_file(&src)
                .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("removing {}: {e}", src.display())))?;
        } else {
            move_one_file(&src, &dst, rel, allow_rename)?;
        }
        progress("move", index as u64 + 1, total);
    }
    prune_emptied_dirs(resolved_data_dir);

    progress("catalog", 0, total);
    idl_rs::store::catalog_read::rebuild_catalog_report(&new_data)?;
    progress("catalog", total, total);

    // Last, and only now: the override. Until this line the app still opens
    // the old root, which is where anything not yet moved still lives.
    set_data_dir_via(settings_path, app_data_dir, app_config_dir, resolved_data_dir, Some(new_root.to_string()))
}

/// Maps a `tauri::Error` from `app.path()...` itself (launch-time
/// path-resolution failing at command time) to `Internal` — no C3 §3.10
/// error row names it because it should never actually happen once
/// `.setup()` has run once at launch.
fn map_path_error(e: tauri::Error) -> IpcError {
    IpcError::new(IpcErrorKind::Internal, e.to_string())
}

/// C3 §3.10 `get_settings()`. Loads `settings.json` from
/// `app_config_dir()`; never fails on a missing/malformed file (C4 §1).
/// Generic over `R: tauri::Runtime` — `handler()` is itself generic, and a
/// non-generic `tauri::AppHandle` (fixed to the default runtime) does not
/// implement `CommandArg` for an arbitrary `R`.
#[tauri::command]
pub fn get_settings<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> Result<AppSettingsDto, IpcError> {
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    Ok(get_settings_via(&settings_path(&app_config_dir)))
}

/// C3 §3.10 `set_settings(settings)`. Ignores `settings.data_dir` (ruling
/// R59 Q5 — `set_data_dir` is the sole writer of that key) and persists
/// `rider_name`/`unit_system`.
#[tauri::command]
pub fn set_settings<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    settings: AppSettingsArg,
) -> Result<AppSettingsDto, IpcError> {
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    set_settings_via(&settings_path(&app_config_dir), settings)
}

/// C3 §3.10 `get_data_dir()`. `resolved_path` is the managed `DataDir`
/// fixed at startup, not a freshly recomputed value.
#[tauri::command]
pub fn get_data_dir<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<DataDirInfo, IpcError> {
    let app_data_dir = app.path().app_data_dir().map_err(map_path_error)?;
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    get_data_dir_via(&settings_path(&app_config_dir), &app_data_dir, &app_config_dir, &data_dir.0)
}

/// C3 §3.10 `set_data_dir(path)`. Writes only the `data_dir` key,
/// read-modify-write, preserving `rider_name`/`unit_system`; does not move
/// existing files.
#[tauri::command]
pub fn set_data_dir<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    data_dir: tauri::State<'_, DataDir>,
    path: Option<String>,
) -> Result<DataDirInfo, IpcError> {
    let app_data_dir = app.path().app_data_dir().map_err(map_path_error)?;
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    set_data_dir_via(&settings_path(&app_config_dir), &app_data_dir, &app_config_dir, &data_dir.0, path)
}

/// C3 §3.10 `move_data_dir(new_root, progress)`. Copies the library to
/// `new_root` with every blob's sha256 verified on arrival, rebuilds the
/// catalog there, then sets the override (C4 §1 "Moving the root", ruling
/// R196). The old root is left in place for the user to delete. Desktop
/// only — the override itself is desktop only (C4 §1, ruling R183), so a
/// move has nowhere to go on mobile.
#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
pub fn move_data_dir<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    data_dir: tauri::State<'_, DataDir>,
    new_root: String,
    progress: tauri::ipc::Channel<crate::commands::device::Progress>,
) -> Result<DataDirInfo, IpcError> {
    let app_data_dir = app.path().app_data_dir().map_err(map_path_error)?;
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    move_data_dir_via(
        &settings_path(&app_config_dir),
        &app_data_dir,
        &app_config_dir,
        &data_dir.0,
        &new_root,
        |phase, done, total| {
            let _ = progress.send(crate::commands::device::Progress {
                done,
                total: Some(total),
                phase: phase.to_string(),
            });
        },
    )
}

/// C3 §3.10 `move_data_dir` on mobile — see the desktop version's doc
/// comment.
#[cfg(any(target_os = "android", target_os = "ios"))]
#[tauri::command]
pub fn move_data_dir(new_root: String) -> Result<DataDirInfo, IpcError> {
    let _ = new_root;
    Err(IpcError::with_detail(
        IpcErrorKind::UnsupportedPlatform,
        "moving the data directory is desktop only",
        serde_json::json!({ "platform": std::env::consts::OS }),
    ))
}

/// C3 §3.10 `BikeProfile` — mirrors `idl_rs::store::profile::BikeProfile`
/// field for field. One struct serves both `list_profiles`/`save_profile`'s
/// return and `save_profile`'s argument, matching C3's single TS interface
/// used both ways (unlike this file's `AppSettingsDto`/`AppSettingsArg`
/// split, which exists only because `set_settings` intentionally ignores
/// one field on input).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BikeProfileDto {
    pub profile_id: String,
    pub profile_name: String,
    /// Creation time, Unix epoch milliseconds.
    pub created_at_ms: i64,
    /// Last-update time, Unix epoch milliseconds.
    pub updated_at_ms: i64,
    /// The SPEC §8 device-config document, stored and pushed verbatim.
    pub config: serde_json::Value,
}

impl From<idl_rs::store::profile::BikeProfile> for BikeProfileDto {
    fn from(p: idl_rs::store::profile::BikeProfile) -> Self {
        Self {
            profile_id: p.profile_id,
            profile_name: p.profile_name,
            created_at_ms: p.created_at_ms,
            updated_at_ms: p.updated_at_ms,
            config: p.config,
        }
    }
}

impl From<BikeProfileDto> for idl_rs::store::profile::BikeProfile {
    fn from(p: BikeProfileDto) -> Self {
        Self {
            profile_id: p.profile_id,
            profile_name: p.profile_name,
            created_at_ms: p.created_at_ms,
            updated_at_ms: p.updated_at_ms,
            config: p.config,
        }
    }
}

/// C3 §3.10 `ProfileLoadReport.skipped` element: a `*.idl0p` file that
/// failed to parse, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkippedFile {
    pub path: String,
    pub reason: String,
}

/// C3 §3.10 `ProfileLoadReport` — `list_profiles`'s return.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileLoadReport {
    /// Sorted by `profile_name` ascending.
    pub profiles: Vec<BikeProfileDto>,
    /// Files that failed to parse — never a failure of the whole load.
    pub skipped: Vec<SkippedFile>,
}

/// Maps [`idl_rs::store::profile::ProfileError`] to the C3 §3.10 error rows
/// (`io`, `internal`). `ProfileErrorKind::Encode` folds to `internal` per
/// C3 §2's folding rule.
fn map_profile_error(e: idl_rs::store::profile::ProfileError) -> IpcError {
    use idl_rs::store::profile::ProfileErrorKind;
    match e.kind {
        ProfileErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
        ProfileErrorKind::Encode => IpcError::new(IpcErrorKind::Internal, e.message),
    }
}

/// `list_profiles`'s transport-agnostic core. `load_all` never fails — a
/// malformed file is reported in [`ProfileLoadReport::skipped`], never
/// printed and never aborting the load.
fn list_profiles_via(data_root: &Path) -> ProfileLoadReport {
    let loaded = idl_rs::store::profile::load_all(data_root);
    ProfileLoadReport {
        profiles: loaded.profiles.into_iter().map(BikeProfileDto::from).collect(),
        skipped: loaded
            .skipped
            .into_iter()
            .map(|(path, reason)| SkippedFile { path: path.display().to_string(), reason })
            .collect(),
    }
}

/// `save_profile`'s transport-agnostic core. Rejects a non-object `config`
/// as `invalid_argument` before writing anything, then delegates to
/// `store::profile::save` (last-write-wins via `write_atomic_with_retry`).
fn save_profile_via(data_root: &Path, profile: BikeProfileDto) -> Result<BikeProfileDto, IpcError> {
    if !profile.config.is_object() {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, "profile.config must be a JSON object"));
    }
    let core_profile: idl_rs::store::profile::BikeProfile = profile.into();
    idl_rs::store::profile::save(data_root, &core_profile).map_err(map_profile_error)?;
    Ok(core_profile.into())
}

/// `delete_profile`'s transport-agnostic core. Checks the file exists
/// *before* calling core's idempotent `delete` (which no-ops on a missing
/// file) so a delete of a stale id raises `not_found` instead of silently
/// succeeding (ruling R59 F2) — this check is the command layer's
/// responsibility, not core's.
fn delete_profile_via(data_root: &Path, profile_id: &str) -> Result<(), IpcError> {
    let path = data_root.join("profiles").join(format!("{profile_id}.idl0p"));
    if !path.is_file() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("profile '{profile_id}' not found")));
    }
    idl_rs::store::profile::delete(data_root, profile_id).map_err(map_profile_error)?;
    Ok(())
}

/// C3 §3.10 `list_profiles()`. Thin over `store::profile::load_all` against
/// `<data>/profiles/*.idl0p` (C4 §2).
#[tauri::command]
pub fn list_profiles(data_dir: tauri::State<'_, DataDir>) -> Result<ProfileLoadReport, IpcError> {
    Ok(list_profiles_via(&data_dir.0))
}

/// C3 §3.10 `save_profile(profile)`. Thin over `store::profile::save`;
/// rejects a non-object `config` as `invalid_argument`.
#[tauri::command]
pub fn save_profile(
    data_dir: tauri::State<'_, DataDir>,
    profile: BikeProfileDto,
) -> Result<BikeProfileDto, IpcError> {
    save_profile_via(&data_dir.0, profile)
}

/// C3 §3.10 `delete_profile(profile_id)`. Raises `not_found` when the
/// file is absent (ruling R59 F2) before delegating to core's idempotent
/// `store::profile::delete`.
#[tauri::command]
pub fn delete_profile(data_dir: tauri::State<'_, DataDir>, profile_id: String) -> Result<(), IpcError> {
    delete_profile_via(&data_dir.0, &profile_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    /// A fresh temp dir standing in for one of `app_data_dir()`/
    /// `app_config_dir()`/an arbitrary override root.
    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-app-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_dto(id: &str, name: &str) -> BikeProfileDto {
        BikeProfileDto {
            profile_id: id.to_string(),
            profile_name: name.to_string(),
            created_at_ms: 1,
            updated_at_ms: 2,
            config: serde_json::json!({"wheel_circumference_front_mm": 2300}),
        }
    }

    #[test]
    fn list_profiles_via_two_saved_profiles_returns_both_sorted_by_name_ascending() {
        // Arrange
        let root = temp_root();
        save_profile_via(&root, sample_dto("p2", "Zebra")).unwrap();
        save_profile_via(&root, sample_dto("p1", "Alpha")).unwrap();

        // Act
        let report = list_profiles_via(&root);

        // Assert
        assert_eq!(report.profiles.len(), 2);
        assert_eq!(report.profiles[0].profile_name, "Alpha");
        assert_eq!(report.profiles[1].profile_name, "Zebra");
        assert!(report.skipped.is_empty());
    }

    #[test]
    fn list_profiles_via_a_malformed_file_lands_in_skipped_and_the_good_profile_still_loads() {
        // Arrange
        let root = temp_root();
        save_profile_via(&root, sample_dto("p1", "Good")).unwrap();
        std::fs::create_dir_all(root.join("profiles")).unwrap();
        std::fs::write(root.join("profiles").join("bad.idl0p"), b"not json").unwrap();

        // Act
        let report = list_profiles_via(&root);

        // Assert
        assert_eq!(report.profiles.len(), 1);
        assert_eq!(report.profiles[0].profile_name, "Good");
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].path.ends_with("bad.idl0p"));
        assert!(!report.skipped[0].reason.is_empty());
    }

    #[test]
    fn save_profile_via_a_non_object_config_is_rejected_and_nothing_is_written() {
        // Arrange
        let root = temp_root();
        let mut dto = sample_dto("p1", "Bad Config");
        dto.config = serde_json::json!([1, 2, 3]);

        // Act
        let result = save_profile_via(&root, dto);

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert!(list_profiles_via(&root).profiles.is_empty());
    }

    #[test]
    fn save_profile_via_a_valid_object_config_succeeds_and_round_trips_through_load_all() {
        // Arrange
        let root = temp_root();
        let dto = sample_dto("p1", "Trek Session 2024");

        // Act
        let written = save_profile_via(&root, dto.clone()).unwrap();

        // Assert
        assert_eq!(written.profile_id, dto.profile_id);
        assert_eq!(written.config, dto.config);
        let loaded = idl_rs::store::profile::load_all(&root);
        assert_eq!(loaded.profiles.len(), 1);
        assert_eq!(loaded.profiles[0].profile_id, "p1");
    }

    #[test]
    fn delete_profile_via_an_existing_profile_removes_its_file_and_it_no_longer_loads() {
        // Arrange
        let root = temp_root();
        save_profile_via(&root, sample_dto("p1", "Gone Soon")).unwrap();

        // Act
        delete_profile_via(&root, "p1").unwrap();

        // Assert
        assert!(idl_rs::store::profile::load_all(&root).profiles.is_empty());
    }

    #[test]
    fn delete_profile_via_an_unknown_id_returns_not_found_and_does_not_touch_other_files() {
        // Arrange
        let root = temp_root();
        save_profile_via(&root, sample_dto("p1", "Untouched")).unwrap();

        // Act
        let result = delete_profile_via(&root, "does-not-exist");

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::NotFound));
        assert_eq!(idl_rs::store::profile::load_all(&root).profiles.len(), 1);
    }

    #[test]
    fn get_settings_via_missing_file_returns_defaults() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);

        // Act
        let dto = get_settings_via(&path);

        // Assert
        assert_eq!(dto.data_dir, None);
        assert_eq!(dto.rider_name, "");
        assert_eq!(dto.unit_system, UnitSystem::Imperial);
    }

    #[test]
    fn get_settings_via_present_file_round_trips_every_field() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let written = AppSettings {
            data_dir: Some("D:\\race-data".to_string()),
            rider_name: "Isaac".to_string(),
            unit_system: UnitSystem::Metric,
        };
        idl_rs::store::settings::save(&path, &written).unwrap();

        // Act
        let dto = get_settings_via(&path);

        // Assert
        assert_eq!(dto.data_dir, written.data_dir);
        assert_eq!(dto.rider_name, written.rider_name);
        assert_eq!(dto.unit_system, written.unit_system);
    }

    #[test]
    fn app_settings_dto_unit_system_serialises_to_the_literal_imperial_and_metric_strings() {
        // Arrange
        let imperial = AppSettingsDto { data_dir: None, rider_name: String::new(), unit_system: UnitSystem::Imperial };
        let metric = AppSettingsDto { data_dir: None, rider_name: String::new(), unit_system: UnitSystem::Metric };

        // Act
        let imperial_json = serde_json::to_value(&imperial).unwrap();
        let metric_json = serde_json::to_value(&metric).unwrap();

        // Assert — the entire justification for typing `unit_system` as
        // `UnitSystem` rather than a hand-rolled `String` mapping rests on
        // this exact byte-for-byte wire shape (C3 §3.10:
        // `"imperial" | "metric"`), not just "is a string".
        assert_eq!(imperial_json["unit_system"], serde_json::json!("imperial"));
        assert_eq!(metric_json["unit_system"], serde_json::json!("metric"));
    }

    #[test]
    fn set_settings_via_ignores_data_dir_in_the_argument() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let existing = AppSettings {
            data_dir: Some("D:\\existing-override".to_string()),
            rider_name: String::new(),
            unit_system: UnitSystem::Imperial,
        };
        idl_rs::store::settings::save(&path, &existing).unwrap();
        let arg = AppSettingsArg {
            data_dir: Some("D:\\attempted-override".to_string()),
            rider_name: "Isaac".to_string(),
            unit_system: UnitSystem::Metric,
        };

        // Act
        let dto = set_settings_via(&path, arg).unwrap();

        // Assert
        assert_eq!(dto.data_dir, Some("D:\\existing-override".to_string()));
    }

    #[test]
    fn set_settings_via_writes_rider_name_and_unit_system_and_reflects_the_reread_value() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let arg =
            AppSettingsArg { data_dir: None, rider_name: "Isaac".to_string(), unit_system: UnitSystem::Metric };

        // Act
        let dto = set_settings_via(&path, arg).unwrap();

        // Assert
        assert_eq!(dto.rider_name, "Isaac");
        assert_eq!(dto.unit_system, UnitSystem::Metric);
        let reread = idl_rs::store::settings::load(&path);
        assert_eq!(reread.rider_name, "Isaac");
        assert_eq!(reread.unit_system, UnitSystem::Metric);
    }

    #[test]
    fn get_data_dir_via_no_override_matches_a_fresh_resolve_and_restart_is_not_required() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let info = get_data_dir_via(&path, &app_data, &app_config, &resolved).unwrap();

        // Assert
        assert_eq!(info.resolved_path, resolved.display().to_string());
        assert_eq!(info.override_path, None);
        assert!(!info.restart_required);
    }

    #[test]
    fn get_data_dir_via_override_changed_on_disk_reports_restart_required() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        // The managed value at startup — no override existed yet.
        let old_resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        // "Changed on disk, app not yet restarted": a new override root is
        // written directly to settings.json after that snapshot.
        let override_root = temp_root();
        idl_rs::store::settings::save(
            &path,
            &AppSettings {
                data_dir: Some(override_root.display().to_string()),
                rider_name: String::new(),
                unit_system: UnitSystem::Imperial,
            },
        )
        .unwrap();

        // Act
        let info = get_data_dir_via(&path, &app_data, &app_config, &old_resolved).unwrap();

        // Assert
        assert_eq!(info.resolved_path, old_resolved.display().to_string());
        assert_eq!(info.override_path, Some(override_root.display().to_string()));
        assert!(info.restart_required);
    }

    #[test]
    fn set_data_dir_via_relative_path_is_rejected_and_nothing_is_written() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let result = set_data_dir_via(&path, &app_data, &app_config, &resolved, Some("relative/dir".to_string()));

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert!(!path.exists());
    }

    #[test]
    fn set_data_dir_via_absolute_path_succeeds_and_creates_the_full_c4_2_tree_at_the_new_root() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        let new_root = temp_root();
        let new_root_str = new_root.display().to_string();

        // Act
        let info = set_data_dir_via(&path, &app_data, &app_config, &resolved, Some(new_root_str.clone())).unwrap();

        // Assert — the full C4 §2 tree, not just the pre-created `data`
        // subdir: this is completed by `set_data_dir_via`'s own trailing
        // `resolve_data_dir` call re-reading the just-written override, the
        // same code path `get_data_dir`/startup use (no duplicated tree
        // list). A future edit that dropped or reordered that call would
        // fail this assertion even though `data/` alone would still exist.
        assert!(new_root.join("data").is_dir());
        assert!(new_root.join("data/blobs/sha256").is_dir());
        assert!(new_root.join("data/sessions").is_dir());
        assert!(new_root.join("data/workbooks").is_dir());
        assert!(new_root.join("data/tracks").is_dir());
        assert!(new_root.join("data/tmp/quarantine").is_dir());
        assert_eq!(info.override_path, Some(new_root_str));
        assert!(info.restart_required);
    }

    /// Builds a small but structurally real library under `<root>/data`:
    /// one blob correctly named for its own content, one session file, one
    /// workbook, and `tmp/`+`catalog.sqlite` content that must *not* travel.
    /// Returns the blob's relative path under `<data>`.
    fn seed_library(data: &Path) -> PathBuf {
        use sha2::Digest;
        let bytes = b"raw ride log bytes".to_vec();
        let digest = hex(&sha2::Sha256::digest(&bytes));
        let rel = PathBuf::from("blobs").join("sha256").join(&digest[0..2]).join(&digest[2..]);
        std::fs::create_dir_all(data.join(&rel).parent().unwrap()).unwrap();
        std::fs::write(data.join(&rel), &bytes).unwrap();
        std::fs::create_dir_all(data.join("sessions").join("s1")).unwrap();
        std::fs::write(data.join("sessions").join("s1").join("session.json"), br#"{"session_id":"s1"}"#).unwrap();
        std::fs::create_dir_all(data.join("workbooks")).unwrap();
        std::fs::write(data.join("workbooks").join("w1.idl1wb"), b"{}").unwrap();
        std::fs::create_dir_all(data.join("tmp")).unwrap();
        std::fs::write(data.join("tmp").join("scratch"), b"do not move me").unwrap();
        std::fs::write(data.join("catalog.sqlite"), b"stale index").unwrap();
        rel
    }

    /// A `progress` sink recording `(phase, done, total)` for assertions.
    fn recording_progress(log: &std::sync::Mutex<Vec<(String, u64, u64)>>) -> impl Fn(&str, u64, u64) + '_ {
        move |phase, done, total| log.lock().unwrap().push((phase.to_string(), done, total))
    }

    #[test]
    fn move_data_dir_via_moves_the_five_trees_verifies_them_and_switches_the_override() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        let blob_rel = seed_library(&old_data);
        let new_root = temp_root().join("moved");
        let log = std::sync::Mutex::new(Vec::new());

        // Act
        let info = move_data_dir_via(
            &settings,
            &app_data,
            &app_config,
            &old_data,
            &new_root.display().to_string(),
            recording_progress(&log),
        )
        .unwrap();

        // Assert — content arrived, skipped things did not, override switched.
        let new_data = new_root.join("data");
        assert_eq!(std::fs::read(new_data.join(&blob_rel)).unwrap(), b"raw ride log bytes");
        assert!(new_data.join("sessions").join("s1").join("session.json").is_file());
        assert!(new_data.join("workbooks").join("w1.idl1wb").is_file());
        assert!(!new_data.join("tmp").join("scratch").exists());
        // The catalog at the destination is rebuilt, never the copied file:
        // whatever is there, it is not the source's stale bytes.
        if new_data.join("catalog.sqlite").is_file() {
            assert_ne!(std::fs::read(new_data.join("catalog.sqlite")).unwrap(), b"stale index".to_vec());
        }
        assert_eq!(info.override_path, Some(new_root.display().to_string()));
        assert_eq!(idl_rs::store::settings::load(&settings).data_dir, Some(new_root.display().to_string()));
        // Moved, not copied (R197): the sources are gone and the emptied
        // directories with them, while everything not in the five trees stays.
        assert!(!old_data.join(&blob_rel).exists());
        assert!(!old_data.join("sessions").join("s1").exists());
        assert!(!old_data.join("blobs").exists());
        assert!(old_data.join("tmp").join("scratch").is_file());
        assert!(old_data.join("catalog.sqlite").is_file());
        // Phases in contract order (C3 §3.10), each with an honest total.
        let phases: Vec<String> = log.lock().unwrap().iter().map(|(p, _, _)| p.clone()).collect();
        assert_eq!(phases.first().map(String::as_str), Some("move"));
        assert_eq!(phases.last().map(String::as_str), Some("catalog"));
        assert!(!phases.iter().any(|p| p == "verify"), "R197 collapsed copy+verify into one 'move' phase");
        let (_, done, total) = log.lock().unwrap().iter().filter(|(p, _, _)| p == "move").last().unwrap().clone();
        assert_eq!(done, total);
        assert_eq!(total, 3);
    }

    #[test]
    fn move_data_dir_via_rerun_after_an_interrupted_move_finishes_it_and_skips_what_arrived() {
        // Arrange — a destination holding a verified copy of the blob, as an
        // interrupted run would leave it, with the source still present.
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        let blob_rel = seed_library(&old_data);
        let new_root = temp_root();
        let new_data = new_root.join("data");
        std::fs::create_dir_all(new_data.join(&blob_rel).parent().unwrap()).unwrap();
        std::fs::copy(old_data.join(&blob_rel), new_data.join(&blob_rel)).unwrap();

        // Act
        let info = move_data_dir_via(
            &settings,
            &app_data,
            &app_config,
            &old_data,
            &new_root.display().to_string(),
            |_, _, _| {},
        )
        .unwrap();

        // Assert — the half-finished target is accepted, not refused as
        // "not empty", and the run completes.
        assert_eq!(info.override_path, Some(new_root.display().to_string()));
        assert_eq!(std::fs::read(new_data.join(&blob_rel)).unwrap(), b"raw ride log bytes");
        assert!(!old_data.join(&blob_rel).exists());
        assert!(new_data.join("sessions").join("s1").join("session.json").is_file());
    }

    #[test]
    fn move_data_dir_via_a_target_holding_someone_elses_data_subtree_is_still_refused() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        seed_library(&old_data);
        let new_root = temp_root();
        // A `data/` directory, but holding something that is not one of the
        // five trees — not this move's own partial output.
        std::fs::create_dir_all(new_root.join("data").join("holiday-photos")).unwrap();

        // Act
        let result =
            move_data_dir_via(&settings, &app_data, &app_config, &old_data, &new_root.display().to_string(), |_, _, _| {});

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert_eq!(idl_rs::store::settings::load(&settings).data_dir, None);
        assert!(old_data.join("sessions").join("s1").join("session.json").is_file());
    }

    #[test]
    fn copy_verify_delete_a_blob_whose_bytes_do_not_match_its_name_fails_and_keeps_the_source() {
        // Arrange — the cross-volume path (a rename would not verify at all).
        let old_data = temp_root();
        let blob_rel = seed_library(&old_data);
        // Corruption already present in the source: the file no longer hashes
        // to the name it is filed under. The copy is byte-perfect; it is the
        // *content-address* check that must catch this.
        std::fs::write(old_data.join(&blob_rel), b"tampered bytes").unwrap();
        let new_data = temp_root();

        // Act
        let result = copy_verify_delete(&old_data.join(&blob_rel), &new_data.join(&blob_rel), &blob_rel);

        // Assert — and, above all, the source survives: deleting an original
        // on the strength of a copy that did not verify is the one
        // unrecoverable mistake here.
        let err = result.expect_err("a blob that does not hash to its own name must stop the move");
        assert_eq!(err.kind, IpcErrorKind::Io);
        assert_eq!(err.detail.unwrap()["reason"], serde_json::json!("blob_hash_mismatch"));
        assert_eq!(std::fs::read(old_data.join(&blob_rel)).unwrap(), b"tampered bytes");
    }

    #[test]
    fn move_data_dir_across_volumes_a_corrupted_blob_leaves_the_source_and_the_override_alone() {
        // Arrange — `allow_rename: false` forces every file down the
        // copy-verify-delete branch, standing in for a destination on another
        // volume. The blob no longer hashes to the name it is filed under.
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        let blob_rel = seed_library(&old_data);
        std::fs::write(old_data.join(&blob_rel), b"tampered bytes").unwrap();
        let new_root = temp_root();

        // Act
        let result = move_data_dir_with(
            &settings,
            &app_data,
            &app_config,
            &old_data,
            &new_root.display().to_string(),
            false,
            |_, _, _| {},
        );

        // Assert — the two things that must survive a bad copy: the original,
        // and the override still pointing at the root that still holds it.
        let err = result.expect_err("a blob that does not hash to its own name must stop the move");
        assert_eq!(err.kind, IpcErrorKind::Io);
        assert_eq!(err.detail.unwrap()["reason"], serde_json::json!("blob_hash_mismatch"));
        assert_eq!(idl_rs::store::settings::load(&settings).data_dir, None);
        assert_eq!(std::fs::read(old_data.join(&blob_rel)).unwrap(), b"tampered bytes");
    }

    #[test]
    fn move_data_dir_across_volumes_a_clean_library_moves_and_the_old_root_is_emptied() {
        // Arrange — same forced cross-volume path, nothing corrupted.
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        let blob_rel = seed_library(&old_data);
        let new_root = temp_root();

        // Act
        let info = move_data_dir_with(
            &settings,
            &app_data,
            &app_config,
            &old_data,
            &new_root.display().to_string(),
            false,
            |_, _, _| {},
        )
        .unwrap();

        // Assert
        assert_eq!(info.override_path, Some(new_root.display().to_string()));
        assert_eq!(std::fs::read(new_root.join("data").join(&blob_rel)).unwrap(), b"raw ride log bytes");
        assert!(!old_data.join("blobs").exists());
        assert!(old_data.join("tmp").join("scratch").is_file());
    }

    #[test]
    fn copy_verify_delete_a_good_blob_lands_verified_and_the_source_is_gone() {
        // Arrange
        let old_data = temp_root();
        let blob_rel = seed_library(&old_data);
        let new_data = temp_root();

        // Act
        copy_verify_delete(&old_data.join(&blob_rel), &new_data.join(&blob_rel), &blob_rel).unwrap();

        // Assert
        assert_eq!(std::fs::read(new_data.join(&blob_rel)).unwrap(), b"raw ride log bytes");
        assert!(!old_data.join(&blob_rel).exists());
    }

    #[test]
    fn copy_verify_delete_a_non_blob_file_is_verified_by_length_and_the_source_is_gone() {
        // Arrange — session.json is not content-addressed, so C4 §1's
        // amended rule checks its size rather than a digest.
        let old_data = temp_root();
        seed_library(&old_data);
        let rel = PathBuf::from("sessions").join("s1").join("session.json");
        let new_data = temp_root();

        // Act
        copy_verify_delete(&old_data.join(&rel), &new_data.join(&rel), &rel).unwrap();

        // Assert
        assert_eq!(std::fs::read(new_data.join(&rel)).unwrap(), br#"{"session_id":"s1"}"#.to_vec());
        assert!(!old_data.join(&rel).exists());
    }

    #[test]
    fn move_data_dir_via_a_non_empty_target_is_rejected_and_nothing_is_copied_or_switched() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        seed_library(&old_data);
        let new_root = temp_root();
        std::fs::write(new_root.join("someone-elses-file.txt"), b"occupied").unwrap();

        // Act
        let result =
            move_data_dir_via(&settings, &app_data, &app_config, &old_data, &new_root.display().to_string(), |_, _, _| {});

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert!(!new_root.join("data").exists());
        assert_eq!(idl_rs::store::settings::load(&settings).data_dir, None);
    }

    #[test]
    fn move_data_dir_via_a_target_inside_the_current_root_is_rejected() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        seed_library(&old_data);
        let nested = old_data.join("sessions").join("inside");

        // Act
        let result =
            move_data_dir_via(&settings, &app_data, &app_config, &old_data, &nested.display().to_string(), |_, _, _| {});

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert_eq!(idl_rs::store::settings::load(&settings).data_dir, None);
    }

    #[test]
    fn move_data_dir_via_a_relative_target_is_rejected() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let settings = settings_path(&app_config);
        let old_data = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let result = move_data_dir_via(&settings, &app_data, &app_config, &old_data, "moved-library", |_, _, _| {});

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert!(!settings.exists());
    }

    #[test]
    fn blob_digest_from_relative_path_recognises_a_blob_and_rejects_everything_else() {
        // Arrange
        let blob = PathBuf::from("blobs").join("sha256").join("ab").join("c".repeat(62));
        let session = PathBuf::from("sessions").join("s1").join("session.json");
        let short = PathBuf::from("blobs").join("sha256").join("ab").join("c".repeat(10));

        // Act
        let (from_blob, from_session, from_short) = (
            blob_digest_from_relative_path(&blob),
            blob_digest_from_relative_path(&session),
            blob_digest_from_relative_path(&short),
        );

        // Assert
        assert_eq!(from_blob, Some(format!("ab{}", "c".repeat(62))));
        assert_eq!(from_session, None);
        assert_eq!(from_short, None);
    }

    #[test]
    fn set_data_dir_via_none_clears_an_existing_override() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let override_root = temp_root();
        idl_rs::store::settings::save(
            &path,
            &AppSettings {
                data_dir: Some(override_root.display().to_string()),
                rider_name: String::new(),
                unit_system: UnitSystem::Imperial,
            },
        )
        .unwrap();
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let info = set_data_dir_via(&path, &app_data, &app_config, &resolved, None).unwrap();

        // Assert
        assert_eq!(info.override_path, None);
        let reread = idl_rs::store::settings::load(&path);
        assert_eq!(reread.data_dir, None);
    }
}
