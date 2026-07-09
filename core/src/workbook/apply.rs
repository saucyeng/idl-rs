//! Apply a workbook's math channels to a session: evaluate each in dependency
//! order, write it into the handle's math store (so later channels can
//! reference it), and collect both a per-channel outcome and the owned derived
//! channel (ready to export — the math store cannot lend a borrow past its
//! lock, so the derived channels are returned by value).

use std::collections::{HashMap, HashSet};

use crate::math::channel_def::MathChannelDef;
use crate::math::eval::{evaluate, MathLapContext};
use crate::math::resolve::resolve_dependencies;
use crate::math::MathEvalError;
use crate::session::handle::SessionHandle;
use crate::session::Channel;
use crate::workbook::model::Workbook;

/// Outcome of evaluating one math channel. `error == None` means success.
#[derive(Debug, Clone)]
pub struct ChannelApplyResult {
    pub name: String,
    pub error: Option<MathEvalError>,
}

/// Result of applying a whole workbook: per-channel outcomes plus the owned
/// derived channels that evaluated successfully.
#[derive(Debug, Clone)]
pub struct ApplyReport {
    pub results: Vec<ChannelApplyResult>,
    pub evaluated: Vec<Channel>,
}

/// Evaluate every math channel in `workbook` against `handle`, resolving
/// cross-channel dependencies ([`resolve_dependencies`]) into the handle's math
/// store. Each success is both `store_math`'d (so dependents see it) and pushed
/// into `ApplyReport.evaluated` as an owned [`Channel`]. With an empty `lap_ctx`,
/// lap-aware channels return `Err(NoLapContext)` and appear in `results` with
/// `error: Some(..)`. Declaration order does not affect correctness — a channel
/// defined before its dependency still resolves.
pub fn apply_workbook(
    handle: &SessionHandle,
    workbook: &Workbook,
    lap_ctx: &MathLapContext,
) -> ApplyReport {
    let defs: HashMap<String, MathChannelDef> = workbook
        .math_channels
        .iter()
        .map(|d| (d.name.clone(), d.clone()))
        .collect();

    let mut results = Vec::with_capacity(workbook.math_channels.len());
    let mut evaluated = Vec::new();

    for def in &workbook.math_channels {
        let mut visited = HashSet::from([def.name.clone()]);
        resolve_dependencies(handle, &def.expression, &defs, lap_ctx, &mut visited);
        match evaluate(&def.expression, handle, lap_ctx) {
            Ok(out) => {
                handle.store_math(&def.name, out.sample_rate_hz, out.samples.clone());
                evaluated.push(Channel::from_f64(
                    def.name.clone(),
                    out.sample_rate_hz,
                    out.samples,
                    None,
                ));
                results.push(ChannelApplyResult {
                    name: def.name.clone(),
                    error: None,
                });
            }
            Err(e) => {
                results.push(ChannelApplyResult {
                    name: def.name.clone(),
                    error: Some(e),
                });
            }
        }
    }

    ApplyReport { results, evaluated }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::MathEvalErrorKind;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(meta, channels)
    }

    fn base(id: &str, samples: Vec<f64>) -> ChannelInput {
        ChannelInput {
            channel_id: id.to_string(),
            sample_rate_hz: 10.0,
            samples,
            sample_times_secs: None,
        }
    }

    fn workbook_with(channels: &[(&str, &str)]) -> Workbook {
        Workbook {
            workbook_id: "wb".to_string(),
            name: "test".to_string(),
            math_channels: channels
                .iter()
                .map(|(n, e)| MathChannelDef {
                    name: n.to_string(),
                    expression: e.to_string(),
                })
                .collect(),
            workbook_version: 1,
            worksheets: Vec::new(),
            overlay_layouts: Vec::new(),
        }
    }

    #[test]
    fn evaluates_independent_channels() {
        // Arrange — two channels off the same base, no interdependency.
        let h = handle_with(vec![base("X", vec![2.0, 4.0])]);
        let wb = workbook_with(&[("Double", "[X] * 2"), ("Half", "[X] / 2")]);

        // Act
        let report = apply_workbook(&h, &wb, &MathLapContext::empty());

        // Assert — both ok, both present in evaluated.
        assert!(report.results.iter().all(|r| r.error.is_none()));
        let ids: Vec<&str> = report
            .evaluated
            .iter()
            .map(|c| c.channel_id.as_str())
            .collect();
        assert!(ids.contains(&"Double") && ids.contains(&"Half"));
        let double = report
            .evaluated
            .iter()
            .find(|c| c.channel_id == "Double")
            .unwrap();
        assert_eq!(double.materialize(), vec![4.0, 8.0]);
    }

    #[test]
    fn resolves_dependency_defined_before_its_dependency() {
        // Arrange — A depends on B, but A is listed FIRST.
        let h = handle_with(vec![base("X", vec![1.0, 2.0])]);
        let wb = workbook_with(&[("A", "[B] + 1"), ("B", "[X] * 10")]);

        // Act
        let report = apply_workbook(&h, &wb, &MathLapContext::empty());

        // Assert — A evaluated correctly using B (10, 20) → (11, 21).
        let a = report
            .evaluated
            .iter()
            .find(|c| c.channel_id == "A")
            .unwrap();
        assert_eq!(a.materialize(), vec![11.0, 21.0]);
    }

    #[test]
    fn lap_aware_channel_reports_no_lap_context_under_empty_ctx() {
        // Arrange — variance_time needs a main + overlay lap; an empty context
        // has neither, so it fails with NoLapContext (it takes one channel ref).
        let h = handle_with(vec![base("X", vec![1.0, 2.0])]);
        let wb = workbook_with(&[("Plain", "[X] + 1"), ("V", "variance_time([X])")]);

        // Act
        let report = apply_workbook(&h, &wb, &MathLapContext::empty());

        // Assert — Plain ok and in evaluated; V failed with NoLapContext, absent.
        let v = report.results.iter().find(|r| r.name == "V").unwrap();
        assert!(matches!(
            v.error.as_ref().unwrap().kind,
            MathEvalErrorKind::NoLapContext
        ));
        assert!(report.evaluated.iter().all(|c| c.channel_id != "V"));
        assert!(report.evaluated.iter().any(|c| c.channel_id == "Plain"));
    }
}
