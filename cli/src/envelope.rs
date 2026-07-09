//! The `idl-rs` CLI's machine-readable output envelope.
//!
//! Every command emits one versioned JSON wrapper so a script or agent has a
//! single shape to parse and one error path to branch on. Structured commands
//! (`info`, `channels`, `laps`, `visits`, `table`) emit a success envelope
//! (`data`) on stdout in JSON mode and an error envelope on stdout on failure;
//! bulk commands (`export`, `math`, `fit`, `recover`, `scan`) write their raw
//! artifact on success and an error envelope to **stderr** on failure.
//!
//! Contract: `schema` (version), `ok` (success discriminator mirroring the exit
//! code), `command`, `engine` (CLI `CARGO_PKG_VERSION`), then exactly one of
//! `data` (object, success) or `error` (`{kind, message, details?}`, failure),
//! plus an optional `warnings` array on success. The error `kind` is the closed
//! seven-variant set the engine's house errors map onto.
//!
//! See `docs/IDL0_SPEC.md` (CLI section) for the published contract.

use std::process::ExitCode;

use serde::Serialize;
use serde_json::{json, Value};

use idl_rs::config::{ConfigError, ConfigErrorKind};
use idl_rs::export::{ExportError, FitExportError};
use idl_rs::math::{MathEvalError, MathEvalErrorKind};
use idl_rs::session::ParseError;

/// Envelope-contract version. One integer covers the whole contract (envelope
/// shape + the union of all `data` shapes). Bumped only on a breaking change;
/// additive changes do not bump it. Starts at `1`.
pub const SCHEMA_VERSION: u32 = 1;

/// The closed set of CLI error classes a consumer branches on. Finer detail
/// lives in [`CliError::details`], never in new kinds — the set stays stable as
/// the engine grows. Serialized snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// A file could not be read or written.
    Io,
    /// A file is present but unusable: bad magic, unsupported schema version,
    /// malformed config JSON, or a required config field is missing.
    InvalidInput,
    /// A named entity does not exist in the loaded data (e.g. unknown channel).
    NotFound,
    /// A math-channel expression failed to evaluate. The specific
    /// `MathEvalErrorKind` is echoed in `details.eval_kind`.
    Eval,
    /// A deferred or not-yet-implemented capability was requested.
    Unsupported,
    /// Invalid arguments or a flag combination not caught by clap (e.g. a
    /// format that cannot be inferred from the output extension).
    Usage,
    /// An unexpected failure — a bug. Should be rare.
    Internal,
}

/// A structured CLI error: a closed `kind` plus a human-readable `message` and
/// an open `details` object. Mirrors the engine's house error shape.
#[derive(Debug, Clone, Serialize)]
pub struct CliError {
    pub kind: ErrorKind,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl CliError {
    /// Construct an error with no `details`.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            details: None,
        }
    }

    /// Construct an error carrying an open `details` object.
    pub fn with_details(kind: ErrorKind, message: impl Into<String>, details: Value) -> Self {
        Self {
            kind,
            message: message.into(),
            details: Some(details),
        }
    }

    /// An `io` error — a file could not be read or written.
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Io, message)
    }

    /// A `usage` error — invalid arguments or flag combination.
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Usage, message)
    }

    /// A `not_found` for an unknown channel, including the available channel
    /// ids so a consumer can self-correct in one retry (design §4).
    pub fn unknown_channel(name: &str, available: &[String]) -> Self {
        Self::with_details(
            ErrorKind::NotFound,
            format!("unknown channel '{name}'"),
            json!({ "entity": "channel", "name": name, "available": available }),
        )
    }
}

/// A non-fatal, machine-readable caveat on a successful result (e.g. a
/// truncated log). Same discriminant discipline as [`CliError`].
#[derive(Debug, Clone, Serialize)]
pub struct Warning {
    pub kind: String,
    pub message: String,
}

impl Warning {
    /// The `truncated_log` warning: the source log ended mid-record and only
    /// the data before the truncation point is present.
    pub fn truncated_log(message: impl Into<String>) -> Self {
        Self {
            kind: "truncated_log".to_string(),
            message: message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Engine-error → CliError mappings (design §4).
// ---------------------------------------------------------------------------

impl From<ParseError> for CliError {
    fn from(e: ParseError) -> Self {
        match e {
            // A file present but unusable for parsing.
            ParseError::InvalidMagicBytes(m) => CliError::with_details(
                ErrorKind::InvalidInput,
                m,
                json!({ "expected_magic": "IDL0" }),
            ),
            ParseError::UnsupportedSchemaVersion(m) => CliError::new(ErrorKind::InvalidInput, m),
            // A *fatal* truncation (the recoverable case becomes a warning, not
            // an error — see `Warning::truncated_log`).
            ParseError::TruncatedRecord(m) => CliError::new(ErrorKind::InvalidInput, m),
            ParseError::Io(m) => CliError::new(ErrorKind::Io, m),
        }
    }
}

impl From<ConfigError> for CliError {
    fn from(e: ConfigError) -> Self {
        let kind = match e.kind {
            ConfigErrorKind::Io => ErrorKind::Io,
            ConfigErrorKind::Parse => ErrorKind::InvalidInput,
            ConfigErrorKind::UnsupportedVersion => ErrorKind::InvalidInput,
        };
        CliError::new(kind, e.message)
    }
}

impl From<MathEvalError> for CliError {
    fn from(e: MathEvalError) -> Self {
        // Echo the engine discriminant in `details.eval_kind` so a consumer can
        // see the precise failure without a new top-level kind per variant.
        let (kind, eval_kind) = match e.kind {
            MathEvalErrorKind::Parse => (ErrorKind::Eval, "parse"),
            MathEvalErrorKind::UnknownFunction => (ErrorKind::Eval, "unknown_function"),
            MathEvalErrorKind::UnknownChannel => (ErrorKind::Eval, "unknown_channel"),
            MathEvalErrorKind::ArgCount => (ErrorKind::Eval, "arg_count"),
            MathEvalErrorKind::Type => (ErrorKind::Eval, "type"),
            MathEvalErrorKind::DivisionByZero => (ErrorKind::Eval, "division_by_zero"),
            MathEvalErrorKind::Runtime => (ErrorKind::Eval, "runtime"),
            MathEvalErrorKind::NotImplemented => (ErrorKind::Unsupported, "not_implemented"),
            MathEvalErrorKind::NoLapContext => (ErrorKind::Unsupported, "no_lap_context"),
        };
        CliError::with_details(kind, e.message, json!({ "eval_kind": eval_kind }))
    }
}

impl From<ExportError> for CliError {
    fn from(e: ExportError) -> Self {
        match e {
            ExportError::UnknownChannel(name) => CliError::with_details(
                ErrorKind::NotFound,
                format!("unknown channel '{name}'"),
                json!({ "entity": "channel", "name": name }),
            ),
            ExportError::Io(e) => CliError::new(ErrorKind::Io, format!("write failed: {e}")),
            ExportError::Json(e) => CliError::new(
                ErrorKind::Internal,
                format!("json serialization failed: {e}"),
            ),
        }
    }
}

impl From<FitExportError> for CliError {
    fn from(e: FitExportError) -> Self {
        match e {
            // The log is well-formed but lacks the GPS fixes a FIT activity
            // requires — present but unusable for this operation.
            FitExportError::NoGpsData => CliError::new(ErrorKind::InvalidInput, e.to_string()),
            FitExportError::Io(io) => CliError::new(ErrorKind::Io, format!("write failed: {io}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Envelope serialization + dispatch.
// ---------------------------------------------------------------------------

/// CLI version string stamped into every envelope's `engine` field.
const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A success envelope: `data` plus optional `warnings`.
#[derive(Serialize)]
struct SuccessEnvelope<'a> {
    schema: u32,
    ok: bool,
    command: &'a str,
    engine: &'a str,
    data: Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<Warning>,
}

/// An error envelope: a single `error` object, no `data`.
#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    schema: u32,
    ok: bool,
    command: &'a str,
    engine: &'a str,
    error: CliError,
}

/// Outcome of a structured command: it either already rendered human text
/// (text mode — the default) or produced JSON `data` + warnings to envelope.
pub enum Structured {
    /// Human text was written to stdout already; nothing to envelope.
    Text,
    /// JSON-mode payload: the command-specific `data` object and any warnings.
    Json { data: Value, warnings: Vec<Warning> },
}

/// Pretty-print a value to a JSON string. Serialization of an envelope of plain
/// data never fails; a failure is a bug, hence the panic.
fn to_pretty<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).expect("envelope serialize")
}

/// Emit a success envelope to stdout (pretty-printed) and return success.
pub fn emit_success(command: &str, data: Value, warnings: Vec<Warning>) -> ExitCode {
    let env = SuccessEnvelope {
        schema: SCHEMA_VERSION,
        ok: true,
        command,
        engine: ENGINE_VERSION,
        data,
        warnings,
    };
    println!("{}", to_pretty(&env));
    ExitCode::SUCCESS
}

/// Build the pretty-printed error envelope for `command`.
fn render_error(command: &str, error: CliError) -> String {
    let env = ErrorEnvelope {
        schema: SCHEMA_VERSION,
        ok: false,
        command,
        engine: ENGINE_VERSION,
        error,
    };
    to_pretty(&env)
}

/// Resolve a structured command's result into output + exit code: a JSON
/// payload is enveloped to stdout, an error is enveloped to stdout, and a
/// text-rendered success has already printed itself.
pub fn emit_structured(command: &str, result: Result<Structured, CliError>) -> ExitCode {
    match result {
        Ok(Structured::Text) => ExitCode::SUCCESS,
        Ok(Structured::Json { data, warnings }) => emit_success(command, data, warnings),
        Err(error) => {
            // Structured commands envelope errors to stdout (the machine stream).
            println!("{}", render_error(command, error));
            ExitCode::FAILURE
        }
    }
}

/// Resolve a bulk command's result into an exit code. Success output was
/// written by the command itself; a failure envelopes the error to **stderr**
/// so the raw-output stream (stdout / `-o`) is never corrupted with an error.
pub fn emit_bulk(command: &str, result: Result<(), CliError>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}", render_error(command, error));
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a `CliError`'s serialized `kind` string.
    fn kind_str(e: &CliError) -> String {
        serde_json::to_value(e.kind)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    // --- ParseError → kind ---------------------------------------------------

    #[test]
    fn parse_error_invalid_magic_maps_to_invalid_input() {
        // Arrange / Act
        let e: CliError = ParseError::InvalidMagicBytes("nope".into()).into();

        // Assert
        assert_eq!(e.kind, ErrorKind::InvalidInput);
        assert_eq!(e.details.unwrap()["expected_magic"], "IDL0");
    }

    #[test]
    fn parse_error_unsupported_schema_maps_to_invalid_input() {
        let e: CliError = ParseError::UnsupportedSchemaVersion("v9".into()).into();
        assert_eq!(e.kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn parse_error_truncated_record_maps_to_invalid_input() {
        let e: CliError = ParseError::TruncatedRecord("cut".into()).into();
        assert_eq!(e.kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn parse_error_io_maps_to_io() {
        let e: CliError = ParseError::Io("disk gone".into()).into();
        assert_eq!(e.kind, ErrorKind::Io);
    }

    // --- ConfigError → kind --------------------------------------------------

    #[test]
    fn config_error_io_maps_to_io() {
        let e: CliError = ConfigError::new(ConfigErrorKind::Io, "no file").into();
        assert_eq!(e.kind, ErrorKind::Io);
    }

    #[test]
    fn config_error_parse_maps_to_invalid_input() {
        let e: CliError = ConfigError::new(ConfigErrorKind::Parse, "bad json").into();
        assert_eq!(e.kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn config_error_unsupported_version_maps_to_invalid_input() {
        let e: CliError = ConfigError::new(ConfigErrorKind::UnsupportedVersion, "too new").into();
        assert_eq!(e.kind, ErrorKind::InvalidInput);
    }

    // --- MathEvalError → kind (+ eval_kind echo) -----------------------------

    #[test]
    fn math_eval_runtime_kinds_map_to_eval_with_eval_kind() {
        // Arrange — the seven non-deferred discriminants and their echoed names.
        let cases = [
            (MathEvalErrorKind::Parse, "parse"),
            (MathEvalErrorKind::UnknownFunction, "unknown_function"),
            (MathEvalErrorKind::UnknownChannel, "unknown_channel"),
            (MathEvalErrorKind::ArgCount, "arg_count"),
            (MathEvalErrorKind::Type, "type"),
            (MathEvalErrorKind::DivisionByZero, "division_by_zero"),
            (MathEvalErrorKind::Runtime, "runtime"),
        ];

        for (kind, expected) in cases {
            // Act
            let e: CliError = MathEvalError::new(kind, "boom").into();

            // Assert
            assert_eq!(e.kind, ErrorKind::Eval, "{expected} should be eval");
            assert_eq!(e.details.unwrap()["eval_kind"], expected);
        }
    }

    #[test]
    fn math_eval_not_implemented_maps_to_unsupported() {
        let e: CliError = MathEvalError::new(MathEvalErrorKind::NotImplemented, "stub").into();
        assert_eq!(e.kind, ErrorKind::Unsupported);
        assert_eq!(e.details.unwrap()["eval_kind"], "not_implemented");
    }

    #[test]
    fn math_eval_no_lap_context_maps_to_unsupported() {
        let e: CliError = MathEvalError::new(MathEvalErrorKind::NoLapContext, "no laps").into();
        assert_eq!(e.kind, ErrorKind::Unsupported);
        assert_eq!(e.details.unwrap()["eval_kind"], "no_lap_context");
    }

    // --- ExportError → kind --------------------------------------------------

    #[test]
    fn export_unknown_channel_maps_to_not_found() {
        let e: CliError = ExportError::UnknownChannel("Fork".into()).into();
        assert_eq!(e.kind, ErrorKind::NotFound);
        assert_eq!(e.details.unwrap()["name"], "Fork");
    }

    #[test]
    fn export_io_maps_to_io() {
        let io = std::io::Error::new(std::io::ErrorKind::Other, "pipe");
        let e: CliError = ExportError::Io(io).into();
        assert_eq!(e.kind, ErrorKind::Io);
    }

    #[test]
    fn export_json_maps_to_internal() {
        // Arrange — provoke a serde_json error to wrap.
        let json_err = serde_json::from_str::<i32>("nope").unwrap_err();

        // Act
        let e: CliError = ExportError::Json(json_err).into();

        // Assert
        assert_eq!(e.kind, ErrorKind::Internal);
    }

    // --- FitExportError → kind -----------------------------------------------

    #[test]
    fn fit_no_gps_maps_to_invalid_input() {
        let e: CliError = FitExportError::NoGpsData.into();
        assert_eq!(e.kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn fit_io_maps_to_io() {
        let io = std::io::Error::new(std::io::ErrorKind::Other, "pipe");
        let e: CliError = FitExportError::Io(io).into();
        assert_eq!(e.kind, ErrorKind::Io);
    }

    // --- constructors + serialization ----------------------------------------

    #[test]
    fn unknown_channel_includes_available_list() {
        // Arrange
        let available = vec!["A".to_string(), "B".to_string()];

        // Act
        let e = CliError::unknown_channel("Z", &available);

        // Assert — the available list is what lets a consumer self-correct.
        assert_eq!(e.kind, ErrorKind::NotFound);
        let d = e.details.unwrap();
        assert_eq!(d["entity"], "channel");
        assert_eq!(d["name"], "Z");
        assert_eq!(d["available"], json!(["A", "B"]));
    }

    #[test]
    fn error_kind_serializes_snake_case() {
        assert_eq!(
            kind_str(&CliError::new(ErrorKind::InvalidInput, "")),
            "invalid_input"
        );
        assert_eq!(
            kind_str(&CliError::new(ErrorKind::NotFound, "")),
            "not_found"
        );
        assert_eq!(kind_str(&CliError::new(ErrorKind::Io, "")), "io");
    }

    #[test]
    fn cli_error_omits_details_when_absent() {
        // Arrange
        let e = CliError::new(ErrorKind::Internal, "boom");

        // Act
        let v = serde_json::to_value(&e).unwrap();

        // Assert — no null `details` key on a detail-less error.
        assert!(v.get("details").is_none());
        assert_eq!(v["kind"], "internal");
        assert_eq!(v["message"], "boom");
    }

    #[test]
    fn warning_truncated_log_has_kind_and_message() {
        // Act
        let w = Warning::truncated_log("log incomplete — 3 records dropped at EOF");

        // Assert
        let v = serde_json::to_value(&w).unwrap();
        assert_eq!(v["kind"], "truncated_log");
        assert_eq!(v["message"], "log incomplete — 3 records dropped at EOF");
    }

    #[test]
    fn success_envelope_has_contract_fields_and_omits_empty_warnings() {
        // Arrange
        let env = SuccessEnvelope {
            schema: SCHEMA_VERSION,
            ok: true,
            command: "laps",
            engine: ENGINE_VERSION,
            data: json!({ "laps": [] }),
            warnings: Vec::new(),
        };

        // Act
        let v = serde_json::to_value(&env).unwrap();

        // Assert
        assert_eq!(v["schema"], SCHEMA_VERSION);
        assert_eq!(v["ok"], true);
        assert_eq!(v["command"], "laps");
        assert_eq!(v["data"], json!({ "laps": [] }));
        assert!(
            v.get("warnings").is_none(),
            "empty warnings must be omitted"
        );
        assert!(v.get("error").is_none());
    }

    #[test]
    fn error_envelope_has_contract_fields() {
        // Arrange
        let env = ErrorEnvelope {
            schema: SCHEMA_VERSION,
            ok: false,
            command: "info",
            engine: ENGINE_VERSION,
            error: CliError::new(ErrorKind::Io, "no file"),
        };

        // Act
        let v = serde_json::to_value(&env).unwrap();

        // Assert
        assert_eq!(v["ok"], false);
        assert_eq!(v["command"], "info");
        assert_eq!(v["error"]["kind"], "io");
        assert!(v.get("data").is_none());
    }
}
