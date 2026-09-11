//! The `session` verbs (ruling R229): `list`, `show`, `laps`, `set-start`,
//! `set-meta`, `import`.
//!
//! Every function here is a wrapper: read the arguments the table declared,
//! make one `idl_rs` call, project the result into the two renderings
//! [`VerbOutput`] carries. The filtering and the metadata patch live in
//! `idl_rs::commands::session_ops`, not here (R230: no CLI-only logic).

use clap::ArgMatches;
use serde_json::{json, Value};

use idl_rs::commands::session_ops::{
    filter_sessions, set_session_meta, visited_track, SessionFilter, SessionMetaPatch,
};
use idl_rs::store::catalog_read::{
    get_session, list_laps, list_sessions, LapSummary, SessionDetail, SessionSummary,
};
use idl_rs::store::import::import_file_path;
use idl_rs::store::session_json::{set_session_start, SessionJson};

use crate::envelope::{CliError, ErrorKind};
use crate::verbs::{integer, opt_integer, opt_text, path, text, Ctx, VerbOutput};

/// `session list` — every catalogued session the filters keep.
pub fn list(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let filter = SessionFilter {
        venue: opt_text(m, "venue"),
        tag: opt_text(m, "tag"),
        since_utc_ms: opt_integer(m, "since"),
        until_utc_ms: opt_integer(m, "until"),
    };

    let all = list_sessions(root)?;
    let mut rows = filter_sessions(&all, &filter);

    // `--track` is the one filter a catalog row cannot answer: track visits
    // live in `session.json`, so it costs one file read per surviving row
    // and is therefore applied last, over the already-narrowed set.
    if let Some(track_id) = opt_text(m, "track") {
        let mut kept = Vec::with_capacity(rows.len());
        for row in rows {
            let detail = get_session(root, &row.session_id)?;
            if visited_track(&detail, &track_id) {
                kept.push(row);
            }
        }
        rows = kept;
    }

    let text = rows
        .iter()
        .map(session_line)
        .chain(std::iter::once(format!("({} session(s))", rows.len())))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(VerbOutput::new(
        text,
        json!({ "sessions": rows.iter().map(summary_json).collect::<Vec<_>>() }),
    ))
}

/// `session show` — one session's metadata, channels and laps.
pub fn show(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let id = text(m, "id")?;

    let detail = get_session(root, &id)?;

    let mut lines = vec![
        format!("session_id     {}", detail.session_id),
        format!("start_utc_ms   {}", detail.timestamp_utc_ms),
        format!("source_format  {}", detail.source_format),
        format!("rider          {}", detail.rider),
        format!("bike           {}", detail.bike),
        format!("venue          {}", detail.venue_name),
        format!("event          {} {}", detail.event_name, detail.event_session),
        format!("tag            {}", detail.tag),
        format!("channels       {}", detail.channels.len()),
        format!("laps           {}", detail.laps.len()),
        format!("track_visits   {}", detail.track_visits.len()),
    ];
    if !detail.long_comment.is_empty() {
        lines.push(format!("notes          {}", detail.long_comment));
    }

    Ok(VerbOutput::new(lines.join("\n"), detail_json(&detail)))
}

/// `session laps` — the indexed lap table.
pub fn laps(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let id = text(m, "id")?;

    let laps = list_laps(root, &id)?;

    let text = laps
        .iter()
        .map(|lap| format!("lap {:>3}  {:>9} ms", lap.lap_number, lap.lap_time_ms))
        .chain(std::iter::once(format!("({} lap(s))", laps.len())))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(VerbOutput::new(
        text,
        json!({ "session_id": id, "laps": laps.iter().map(lap_json).collect::<Vec<_>>() }),
    ))
}

/// `session set-start` — set the recording start time by hand.
pub fn set_start(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let id = text(m, "id")?;
    let utc_ms = integer(m, "utc-ms")?;

    if ctx.dry_run {
        // Nothing is written, but the argument is still validated here so a
        // dry run reports the same rejection the real run would.
        if utc_ms <= 0 {
            return Err(CliError::usage(format!("utc-ms must be > 0, got {utc_ms}")));
        }
        return Ok(VerbOutput::new(
            format!("would set {id}'s start to {utc_ms}"),
            json!({ "session_id": id, "timestamp_utc_ms": utc_ms, "written": false }),
        ));
    }

    let doc = set_session_start(root, &id, utc_ms)?;

    Ok(VerbOutput::new(
        format!("set {id}'s start to {utc_ms}"),
        json!({ "session_id": id, "written": true, "session": session_json(&doc) }),
    ))
}

/// `session set-meta` — set the descriptive metadata fields.
pub fn set_meta(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let id = text(m, "id")?;
    let patch = SessionMetaPatch {
        venue_name: opt_text(m, "venue"),
        rider: opt_text(m, "rider"),
        bike: opt_text(m, "bike"),
        event_name: opt_text(m, "event"),
        event_session: opt_text(m, "event-session"),
        long_comment: opt_text(m, "notes"),
        tag: opt_text(m, "tag"),
    };

    let report = set_session_meta(root, &id, &patch, ctx.dry_run)?;

    let text = match (report.changed.is_empty(), ctx.dry_run) {
        (true, _) => format!("{id}: already set, nothing written"),
        (false, true) => format!("would set {id}'s {}", report.changed.join(", ")),
        (false, false) => format!("set {id}'s {}", report.changed.join(", ")),
    };

    Ok(VerbOutput::new(
        text,
        json!({
            "session_id": id,
            "changed": report.changed,
            "written": !report.unchanged,
            "session": session_json(&report.doc),
        }),
    ))
}

/// `session import` — import one log file into the data directory.
pub fn import(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let file = path(m, "file")?;

    let extension = file
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .ok_or_else(|| {
            CliError::usage(format!("{} has no extension, so no importer covers it", file.display()))
        })?;

    if ctx.dry_run {
        if !file.is_file() {
            return Err(CliError::new(ErrorKind::Io, format!("{} is not a file", file.display())));
        }
        return Ok(VerbOutput::new(
            format!("would import {} as {extension}", file.display()),
            json!({ "file": file.display().to_string(), "extension": extension, "written": false }),
        ));
    }

    let report = import_file_path(root, &extension, &file)?;

    let outcome = format!("{:?}", report.outcome).to_lowercase();
    Ok(VerbOutput::new(
        format!("{outcome}: {}", report.data_parquet.display()),
        json!({
            "file": file.display().to_string(),
            "outcome": outcome,
            "data_parquet": report.data_parquet.display().to_string(),
            "truncation_warning": report.truncation_warning,
            "written": true,
        }),
    ))
}

// ---------------------------------------------------------------------------
// Projections. The core types are not `Serialize` (they are engine types,
// not wire types), so the JSON shapes are written out here, field for field.
// ---------------------------------------------------------------------------

/// One `session list` text line.
fn session_line(row: &SessionSummary) -> String {
    format!(
        "{}  {}  {}  {}  laps={}",
        row.session_id,
        row.timestamp_utc_ms,
        dash_if_empty(&row.venue_name),
        dash_if_empty(&row.bike),
        row.lap_count.unwrap_or(0)
    )
}

/// `—` for an unset string, so a text table's columns stay readable.
fn dash_if_empty(s: &str) -> &str {
    if s.is_empty() {
        "—"
    } else {
        s
    }
}

/// C3's `SessionSummary` shape.
fn summary_json(row: &SessionSummary) -> Value {
    json!({
        "session_id": row.session_id,
        "blob_sha256": row.blob_sha256,
        "source_format": row.source_format,
        "device_id": row.device_id,
        "importer_version": row.importer_version,
        "engine_version": row.engine_version,
        "timestamp_utc_ms": row.timestamp_utc_ms,
        "created_at_ms": row.created_at_ms,
        "rider": row.rider,
        "bike": row.bike,
        "venue_name": row.venue_name,
        "event_name": row.event_name,
        "event_session": row.event_session,
        "short_comment": row.short_comment,
        "tag": row.tag,
        "lap_count": row.lap_count,
        "duration_ms": row.duration_ms,
    })
}

/// C3's `SessionDetail` shape, with channels and laps summarised.
fn detail_json(detail: &SessionDetail) -> Value {
    json!({
        "session_id": detail.session_id,
        "device_id": detail.device_id,
        "timestamp_utc_ms": detail.timestamp_utc_ms,
        "source_format": detail.source_format,
        "blob_sha256": detail.blob_sha256,
        "rider": detail.rider,
        "bike": detail.bike,
        "bike_comment": detail.bike_comment,
        "venue_name": detail.venue_name,
        "event_name": detail.event_name,
        "event_session": detail.event_session,
        "short_comment": detail.short_comment,
        "long_comment": detail.long_comment,
        "tag": detail.tag,
        "channels": detail.channels.iter().map(|c| json!({
            "channel_id": c.channel_id,
            "nominal_rate_hz": c.nominal_rate_hz,
            "unit": c.unit,
            "source_kind": c.source_kind,
            "channel_kind": c.channel_kind,
            "sample_count": c.sample_count,
        })).collect::<Vec<_>>(),
        "lap_count": detail.laps.len(),
        "track_visits": detail.track_visits.iter().map(|v| json!({
            "visit_id": v.visit_id,
            "track_id": v.track_id,
            "start_timestamp_ms": v.start_timestamp_ms,
            "end_timestamp_ms": v.end_timestamp_ms,
            "lap_count": v.laps.len(),
        })).collect::<Vec<_>>(),
        "reference_lap_number": detail.reference_lap_number,
        "main_lap_number": detail.main_lap_number,
        "starred_lap_number": detail.starred_lap_number,
        "ignored_lap_numbers": detail.ignored_lap_numbers,
    })
}

/// C3's `LapSummary` shape.
fn lap_json(lap: &LapSummary) -> Value {
    json!({
        "lap_number": lap.lap_number,
        "lap_time_ms": lap.lap_time_ms,
    })
}

/// The mutable half of `session.json` — what `set-start`/`set-meta` changed.
fn session_json(doc: &SessionJson) -> Value {
    json!({
        "session_id": doc.session_id,
        "timestamp_utc_ms": doc.timestamp_utc_ms,
        "rider": doc.rider,
        "bike": doc.bike,
        "venue_name": doc.venue_name,
        "event_name": doc.event_name,
        "event_session": doc.event_session,
        "short_comment": doc.short_comment,
        "long_comment": doc.long_comment,
        "tag": doc.tag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_string_prints_as_a_dash() {
        // Arrange
        let empty = "";

        // Act
        let rendered = dash_if_empty(empty);

        // Assert
        assert_eq!(rendered, "—");
    }

    #[test]
    fn a_set_string_prints_itself() {
        // Arrange
        let venue = "Cadwell";

        // Act
        let rendered = dash_if_empty(venue);

        // Assert
        assert_eq!(rendered, "Cadwell");
    }

    #[test]
    fn the_session_json_projection_carries_every_field_set_meta_can_change() {
        // Arrange
        let doc = idl_rs::store::session_json::empty_session_json("s1");

        // Act
        let value = session_json(&doc);

        // Assert — one key per `SessionMetaPatch` field, plus identity and start.
        for key in
            ["session_id", "timestamp_utc_ms", "rider", "bike", "venue_name", "event_name",
             "event_session", "long_comment", "tag"]
        {
            assert!(value.get(key).is_some(), "{key} missing");
        }
    }
}
