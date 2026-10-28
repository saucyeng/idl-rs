//! The evaluation planner (evaluation-memory spec §4.2.B, Phase 2).
//!
//! Before an evaluation runs, the planner reads the parsed document and the
//! session's per-channel sample counts (from the Parquet footer, no data
//! read) and works out, statically:
//!
//! - **the schedule** — every definition in scope, dependencies first, with
//!   document order breaking ties (which keeps ledger R49's first-wins rule);
//! - **consumer counts** — how many scheduled definitions read each one, so
//!   the resolver can drop a full-rate buffer after its last reader;
//! - **the retain set** — the definitions whose full-rate buffers must
//!   survive the evaluation because something outside the maths reads them
//!   (a `js` cell, a `${…}` span, a table cell). Every other full-rate value
//!   keeps only its metadata once its last consumer has run;
//! - **shapes** — every definition's value as a scalar, a `[lap]` vector or
//!   a series of known length;
//! - **the estimate `E`** — the bytes the evaluation will hold at its peak,
//!   by simulating the schedule with liveness release, which the app's
//!   admission control (Phase 3) reserves before the evaluation starts.
//!
//! Pure: no I/O. The footer counts reach it as [`PlanInputs`].

use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

use crate::math::parse::{parse_with_constants, Ast};
use crate::math::resolve::channel_refs;
use crate::store::parquet::ChannelColumnInfo;

use super::{find_inline_exprs, CellKindToken, MathCellDef, WorkbookDoc};

/// Bytes one `f64` sample occupies.
const F64_BYTES: u64 = 8;

/// The `(numerator, denominator)` safety margin on an estimate, 5/4: the
/// spec's `E = (I + F + T) × 1.25` (§4.2.B), the same margin
/// `idl-rs-tauri`'s decode reservations carry.
pub const ESTIMATE_MARGIN: (u64, u64) = (5, 4);

/// Bytes the session cache holds per stored sample beyond the raw value
/// itself, as the cache is shaped today: the channel's own `t_us` and
/// recorded-time copies, 8 B each (spec §1.1, "18 B/s" for an `i16`
/// channel). Evaluation-memory Phase 4 shares one `t_us` per source and drops
/// this to about 0.
pub const CACHE_AXIS_BYTES_PER_SAMPLE: u64 = 16;

/// One stored channel as the planner sees it: what the footer says, nothing
/// decoded.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDims {
    /// The channel's own sample count
    /// ([`ChannelColumnInfo::sample_count`]).
    pub samples: usize,
    /// Nominal sample rate, Hz; `0.0` for an event-driven channel.
    pub rate_hz: f64,
    /// Bytes per sample at stored width: 2, 4 or 8; 0 for a synthesized
    /// channel with no stored column (`Time`).
    pub sample_bytes: usize,
}

/// What the planner knows about the session before anything is decoded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanInputs {
    /// Every stored channel, by `channel_id`.
    pub channels: HashMap<String, ChannelDims>,
    /// `(samples, rate_hz)` of the session's time base — the channel
    /// `current_lap()`, `sector_number()` and the synthesized `Time` and
    /// `Distance` are built on. `None` when the session has no positive-rate
    /// channel (or there is no session).
    pub time_base: Option<(usize, f64)>,
}

impl PlanInputs {
    /// From a session's footer index ([`crate::store::parquet::read_channel_index`]).
    ///
    /// The time base is the channel `Time` is synthesized from: the highest
    /// positive rate, the longest at that rate.
    pub fn from_channel_index(index: &[ChannelColumnInfo]) -> Self {
        let mut channels = HashMap::new();
        let mut time_base: Option<(usize, f64)> = None;
        for info in index {
            channels.insert(
                info.channel_id.clone(),
                ChannelDims { samples: info.sample_count, rate_hz: info.nominal_rate_hz, sample_bytes: info.sample_bytes },
            );
            if info.nominal_rate_hz > 0.0 {
                let better = match time_base {
                    None => true,
                    Some((len, rate)) => {
                        info.nominal_rate_hz > rate || (info.nominal_rate_hz == rate && info.sample_count > len)
                    }
                };
                if better {
                    time_base = Some((info.sample_count, info.nominal_rate_hz));
                }
            }
        }
        PlanInputs { channels, time_base }
    }

    /// The dims a `[name]` reference to a session channel resolves to: a
    /// stored channel's own, or the time base for the synthesized `Time` and
    /// `Distance` (which have no stored column).
    fn dims(&self, name: &str) -> Option<ChannelDims> {
        if let Some(d) = self.channels.get(name) {
            return Some(d.clone());
        }
        match name {
            "Time" | "Distance" => {
                self.time_base.map(|(samples, rate_hz)| ChannelDims { samples, rate_hz, sample_bytes: 0 })
            }
            _ => None,
        }
    }
}

/// Which definitions' full-rate buffers survive the evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Retain {
    /// Every value is kept: [`super::eval_cells`]'s behaviour.
    All,
    /// Only these names keep their full-rate buffers; every other full-rate
    /// value is released after its last consumer, keeping its metadata.
    Names(BTreeSet<String>),
}

impl Retain {
    /// `true` when `name`'s full-rate buffer must survive.
    pub fn keeps(&self, name: &str) -> bool {
        match self {
            Retain::All => true,
            Retain::Names(names) => names.contains(name),
        }
    }
}

/// What an evaluation is asked for: which definitions to evaluate and which
/// full-rate values to keep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Demand {
    /// `None`: evaluate every definition (the math cells show each one).
    /// `Some(names)`: evaluate only these names and their dependencies.
    pub scope: Option<BTreeSet<String>>,
    /// The values whose full-rate buffers are kept.
    pub retain: Retain,
}

impl Demand {
    /// Every definition, every value kept — what [`super::eval_cells`] does.
    pub fn everything() -> Self {
        Demand { scope: None, retain: Retain::All }
    }

    /// Every definition, keeping the full-rate values the document's
    /// consumers can read ([`consumer_names`]): the app's notebook
    /// evaluation (spec §4.2.B "demand set").
    pub fn document(doc: &WorkbookDoc) -> Self {
        Demand { scope: None, retain: Retain::Names(consumer_names(doc)) }
    }

    /// Only `names` and what they depend on, keeping `names`' own values: a
    /// fetch of one definition that has nothing to share with.
    pub fn closure<I: IntoIterator<Item = String>>(names: I) -> Self {
        let names: BTreeSet<String> = names.into_iter().collect();
        Demand { scope: Some(names.clone()), retain: Retain::Names(names) }
    }

    /// Only `names` and what they depend on, keeping **no** full-rate value:
    /// a run that wants reductions (rank-0 values and `[lap]` vectors) and
    /// nothing else, such as one member of a session set.
    pub fn reductions<I: IntoIterator<Item = String>>(names: I) -> Self {
        Demand { scope: Some(names.into_iter().collect()), retain: Retain::Names(BTreeSet::new()) }
    }
}

/// Every name the document's non-maths consumers mention, found lexically
/// (spec §4.2.B): each identifier in a `js` cell's body (which covers
/// `channel("x")` and bare `x`), in a `table` cell's body, and in every
/// `${…}` span of the prose. A conservative superset of the definitions
/// those consumers can read, with no new wire: the app's own bind pass
/// (`referencedDefinitionNames`) uses the same identifier rule.
pub fn consumer_names(doc: &WorkbookDoc) -> BTreeSet<String> {
    let defined: HashSet<&str> = doc.defs.iter().map(|d| d.name.as_str()).collect();
    let mut found: BTreeSet<String> = BTreeSet::new();
    let mut scan = |text: &str| {
        for ident in identifiers(text) {
            if defined.contains(ident) {
                found.insert(ident.to_string());
            }
        }
    };
    for cell in &doc.cells {
        if matches!(cell.kind_token, CellKindToken::Js | CellKindToken::Table) {
            scan(&cell.raw_fence_body);
        }
        for prose in [&cell.prose_before, &cell.prose_after].into_iter().flatten() {
            for span in find_inline_exprs(prose) {
                scan(&span.js_expr);
            }
        }
    }
    if let Some(prose) = &doc.trailing_prose {
        for span in find_inline_exprs(prose) {
            scan(&span.js_expr);
        }
    }
    found
}

/// Every JavaScript-style identifier in `text`: `[A-Za-z_$][A-Za-z0-9_$]*`,
/// matched greedily left to right, string contents included.
fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_alphabetic() || c == b'_' || c == b'$' {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'$') {
                    i += 1;
                }
                return Some(&text[start..i]);
            }
            if c.is_ascii_digit() {
                // A number's tail (`1e5`, `0x1f`) is not an identifier.
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.') {
                    i += 1;
                }
                continue;
            }
            i += 1;
        }
        None
    })
}

/// A value's shape, known before evaluation (spec §4.2.B "Graph and
/// shapes").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueShape {
    /// One number.
    Scalar,
    /// One value per lap (`[lap]`) — tens of values at most.
    Lap,
    /// A series of `len` samples at `rate_hz` Hz.
    Series {
        /// Samples.
        len: usize,
        /// Sample rate, Hz.
        rate_hz: f64,
    },
    /// Three series of `len` samples (a `Vec3`): an intermediate only — a
    /// definition that evaluates to one is an error.
    Vec3 {
        /// Samples per component.
        len: usize,
    },
}

impl ValueShape {
    /// Bytes this value's buffer holds once evaluated: 8 B per sample of a
    /// series (24 B for a `Vec3`), 0 for a scalar or a `[lap]` vector (tens
    /// of bytes, not worth planning around).
    pub fn bytes(&self) -> u64 {
        match self {
            ValueShape::Scalar | ValueShape::Lap => 0,
            ValueShape::Series { len, .. } => *len as u64 * F64_BYTES,
            ValueShape::Vec3 { len } => *len as u64 * F64_BYTES * 3,
        }
    }

    /// Samples per component; 1 for a scalar, 0 for a `[lap]` vector.
    fn len(&self) -> usize {
        match self {
            ValueShape::Scalar => 1,
            ValueShape::Lap => 0,
            ValueShape::Series { len, .. } | ValueShape::Vec3 { len } => *len,
        }
    }

    /// The wider of two operands of an element-wise operation: a series
    /// beats a `[lap]` vector beats a scalar, and between series the longer
    /// one (a mismatch is an evaluation error, which allocates nothing more).
    fn widest(a: ValueShape, b: ValueShape) -> ValueShape {
        use ValueShape::*;
        match (a, b) {
            (Vec3 { len: x }, Vec3 { len: y }) => Vec3 { len: x.max(y) },
            (Vec3 { .. }, _) => a,
            (_, Vec3 { .. }) => b,
            (Series { len: x, .. }, Series { len: y, .. }) => {
                if y > x {
                    b
                } else {
                    a
                }
            }
            (Series { .. }, _) => a,
            (_, Series { .. }) => b,
            (Lap, _) | (_, Lap) => Lap,
            _ => Scalar,
        }
    }
}

/// The planner's memory estimate for one evaluation (spec §4.1), bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// `I`: the compact inputs the scheduled definitions read, as the
    /// session cache holds them ([`CACHE_AXIS_BYTES_PER_SAMPLE`] plus the
    /// raw value per sample). Already-resident inputs cost nothing new; see
    /// [`EvalPlan::inputs`] to subtract them.
    pub inputs_bytes: u64,
    /// `F`: the most full-rate values (8 B per sample) alive at once over
    /// the schedule, with liveness release and the retain set.
    pub live_peak_bytes: u64,
    /// `T`: the largest single definition's transient — its operands
    /// widened to `f64` and its operations' scratch — beyond its own value.
    pub transient_bytes: u64,
    /// The simulated peak of the evaluation itself: the most bytes alive at
    /// any step, live values plus that step's transient. Never more than
    /// `live_peak_bytes + transient_bytes` (the peaks of `F` and `T` need not
    /// fall on the same step), and the figure admission reserves.
    pub step_peak_bytes: u64,
}

impl MemoryEstimate {
    /// `I + peak`: bytes the evaluation needs, inputs included, before the
    /// margin.
    pub fn total_bytes(&self) -> u64 {
        self.inputs_bytes.saturating_add(self.step_peak_bytes)
    }

    /// [`Self::total_bytes`] with [`ESTIMATE_MARGIN`] applied: the spec's `E`.
    pub fn with_margin(&self) -> u64 {
        let (num, den) = ESTIMATE_MARGIN;
        self.total_bytes().saturating_mul(num) / den
    }
}

/// One stored channel the plan reads, and what holding it costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedInput {
    /// The channel's `channel_id`.
    pub channel: String,
    /// Bytes the session cache holds for it once decoded: raw samples at
    /// stored width plus [`CACHE_AXIS_BYTES_PER_SAMPLE`] per sample.
    pub bytes: u64,
}

/// A planned evaluation: the order to run it in, when each value can go,
/// and what it costs (spec §4.2.B).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalPlan {
    /// Indices into the definitions the plan was built from, in evaluation
    /// order: every definition in scope, dependencies first, document order
    /// breaking ties. Definitions caught in (or behind) a cycle come last,
    /// in document order; the resolver reports each as blocked.
    pub schedule: Vec<usize>,
    /// Per definition index: how many scheduled definitions read it by name.
    /// Only a name's first definition in document order is ever read (R49);
    /// a later duplicate has 0.
    pub consumers: Vec<usize>,
    /// Per definition index: its value's shape, `Scalar` for one out of
    /// scope or one that cannot be parsed.
    pub shapes: Vec<ValueShape>,
    /// Per definition index: whether it is in scope (evaluated at all).
    pub in_scope: Vec<bool>,
    /// The values whose full-rate buffers survive.
    pub retain: Retain,
    /// The stored channels the scheduled definitions read, in first-use
    /// order.
    pub inputs: Vec<PlannedInput>,
    /// The memory estimate.
    pub estimate: MemoryEstimate,
}

impl EvalPlan {
    /// Plans `defs` (the document's flat definitions, in document order)
    /// against `inputs` for `demand`.
    pub fn new(defs: &[MathCellDef], constants: &HashMap<String, f64>, inputs: &PlanInputs, demand: &Demand) -> Self {
        let primary: HashMap<&str, usize> = defs.iter().enumerate().fold(HashMap::new(), |mut m, (i, d)| {
            m.entry(d.name.as_str()).or_insert(i);
            m
        });
        // Each definition's distinct dependencies, as primary indices, in
        // the order `channel_refs` lists them.
        let deps: Vec<Vec<usize>> = defs
            .iter()
            .map(|d| {
                let mut out: Vec<usize> = Vec::new();
                for name in channel_refs(&d.expr_text) {
                    if let Some(&p) = primary.get(name.as_str()) {
                        if !out.contains(&p) {
                            out.push(p);
                        }
                    }
                }
                out
            })
            .collect();

        let in_scope = scope_of(defs, &deps, &primary, demand);
        let schedule = schedule_of(&deps, &in_scope);
        let mut consumers = vec![0usize; defs.len()];
        for &i in &schedule {
            for &p in &deps[i] {
                consumers[p] += 1;
            }
        }

        let mut sim = Simulation::new(defs, inputs, &primary, &consumers);
        for &i in &schedule {
            sim.step(i, constants, &deps, &demand.retain);
        }
        let estimate = sim.estimate();
        EvalPlan {
            schedule,
            consumers,
            shapes: sim.shapes,
            in_scope,
            retain: demand.retain.clone(),
            inputs: sim.inputs,
            estimate,
        }
    }

    /// [`MemoryEstimate::with_margin`] counting only the inputs `missing`
    /// says are not resident yet: `E = (I_missing + peak) × 1.25`.
    pub fn estimate_with_missing_inputs(&self, missing: impl Fn(&str) -> bool) -> u64 {
        let inputs_bytes = self.inputs.iter().filter(|i| missing(&i.channel)).map(|i| i.bytes).sum();
        MemoryEstimate { inputs_bytes, ..self.estimate }.with_margin()
    }
}

/// Which definitions `demand` evaluates: all of them, or the closure of its
/// names over `deps`.
fn scope_of(defs: &[MathCellDef], deps: &[Vec<usize>], primary: &HashMap<&str, usize>, demand: &Demand) -> Vec<bool> {
    let Some(names) = &demand.scope else {
        return vec![true; defs.len()];
    };
    let mut in_scope = vec![false; defs.len()];
    let mut stack: Vec<usize> = names.iter().filter_map(|n| primary.get(n.as_str()).copied()).collect();
    while let Some(i) = stack.pop() {
        if in_scope[i] {
            continue;
        }
        in_scope[i] = true;
        stack.extend(deps[i].iter().copied());
    }
    in_scope
}

/// Kahn's algorithm over the in-scope definitions, the lowest document
/// index first among the ready ones; whatever never becomes ready (a cycle,
/// or a definition behind one) is appended in document order.
fn schedule_of(deps: &[Vec<usize>], in_scope: &[bool]) -> Vec<usize> {
    let n = deps.len();
    let mut waiting: Vec<usize> = deps.iter().map(|d| d.len()).collect();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, d) in deps.iter().enumerate() {
        for &p in d {
            readers[p].push(i);
        }
    }
    let mut ready: BinaryHeap<Reverse<usize>> =
        (0..n).filter(|&i| in_scope[i] && waiting[i] == 0).map(Reverse).collect();
    let mut done = vec![false; n];
    let mut schedule = Vec::with_capacity(n);
    while let Some(Reverse(i)) = ready.pop() {
        done[i] = true;
        schedule.push(i);
        for &r in &readers[i] {
            waiting[r] -= 1;
            if waiting[r] == 0 && in_scope[r] {
                ready.push(Reverse(r));
            }
        }
    }
    schedule.extend((0..n).filter(|&i| in_scope[i] && !done[i]));
    schedule
}

/// The schedule's memory simulation (spec §4.1): live values, released after
/// their last consumer unless retained, plus each step's transient.
struct Simulation<'a> {
    defs: &'a [MathCellDef],
    inputs_dims: &'a PlanInputs,
    primary: &'a HashMap<&'a str, usize>,
    shapes: Vec<ValueShape>,
    /// Consumers still to run, per definition.
    remaining: Vec<usize>,
    /// Bytes each definition's value holds while it is live.
    live: Vec<u64>,
    live_now: u64,
    live_peak: u64,
    transient_peak: u64,
    step_peak: u64,
    inputs: Vec<PlannedInput>,
    seen_inputs: HashSet<String>,
}

impl<'a> Simulation<'a> {
    fn new(
        defs: &'a [MathCellDef],
        inputs_dims: &'a PlanInputs,
        primary: &'a HashMap<&'a str, usize>,
        consumers: &[usize],
    ) -> Self {
        Simulation {
            defs,
            inputs_dims,
            primary,
            shapes: vec![ValueShape::Scalar; defs.len()],
            remaining: consumers.to_vec(),
            live: vec![0; defs.len()],
            live_now: 0,
            live_peak: 0,
            transient_peak: 0,
            step_peak: 0,
            inputs: Vec::new(),
            seen_inputs: HashSet::new(),
        }
    }

    fn step(&mut self, i: usize, constants: &HashMap<String, f64>, deps: &[Vec<usize>], retain: &Retain) {
        let def = &self.defs[i];
        let (shape, step_bytes) = match parse_with_constants(&def.expr_text, constants) {
            Ok(ast) => {
                let mut walk = Walk { sim: self, widened: HashSet::new(), widened_bytes: 0 };
                let cost = walk.node(&ast);
                let shape = cost.shape;
                // The memo keeps every widened input until the definition
                // returns, alongside the expression's own peak.
                (shape, walk.widened_bytes + cost.peak.max(cost.out))
            }
            Err(_) => (ValueShape::Scalar, 0),
        };
        let shape = match shape {
            // A `Vec3` result is an error; nothing survives it.
            ValueShape::Vec3 { .. } => ValueShape::Scalar,
            other => other,
        };
        self.shapes[i] = shape;
        let value = shape.bytes();

        self.step_peak = self.step_peak.max(self.live_now + step_bytes);
        self.transient_peak = self.transient_peak.max(step_bytes.saturating_sub(value));
        self.live[i] = value;
        self.live_now += value;
        self.live_peak = self.live_peak.max(self.live_now);

        for &p in &deps[i] {
            self.remaining[p] = self.remaining[p].saturating_sub(1);
            if self.remaining[p] == 0 {
                self.release(p, retain);
            }
        }
        if self.remaining[i] == 0 {
            self.release(i, retain);
        }
    }

    fn release(&mut self, i: usize, retain: &Retain) {
        if retain.keeps(&self.defs[i].name) || !matches!(self.shapes[i], ValueShape::Series { .. }) {
            return;
        }
        self.live_now -= self.live[i];
        self.live[i] = 0;
    }

    fn estimate(&self) -> MemoryEstimate {
        MemoryEstimate {
            inputs_bytes: self.inputs.iter().map(|i| i.bytes).sum(),
            live_peak_bytes: self.live_peak,
            transient_bytes: self.transient_peak,
            step_peak_bytes: self.step_peak,
        }
    }
}

/// One node's cost: its shape, the new bytes its result holds (`out`, 0 when
/// it shares a buffer that already exists), and the most extra bytes alive
/// while it evaluated (`peak`, `out` included).
#[derive(Debug, Clone, Copy)]
struct NodeCost {
    shape: ValueShape,
    out: u64,
    peak: u64,
}

impl NodeCost {
    fn free(shape: ValueShape) -> Self {
        NodeCost { shape, out: 0, peak: 0 }
    }
}

/// How a builtin allocates, in units of its output's `f64` buffer.
enum OpAlloc {
    /// Returns a scalar or a `[lap]` vector: nothing full-rate.
    Reduce,
    /// A reduction that copies and sorts its input first (`median`,
    /// `percentile`).
    ReduceCopying,
    /// Builds its output straight into its final buffer: one buffer.
    Direct,
    /// Builds a `Vec` and then copies it into its final buffer
    /// (`eval.rs`'s `channel` helper): two buffers at the moment of the
    /// copy, one after.
    Copied,
    /// `butter`: the zero-phase filter's own passes, then
    /// [`OpAlloc::Copied`]'s copy.
    Filtered,
}

/// The per-definition cost walk.
struct Walk<'s, 'a> {
    sim: &'s mut Simulation<'a>,
    /// Session channels this definition has already widened (the
    /// per-definition memo shares them after the first reference).
    widened: HashSet<String>,
    /// Bytes of widened session channels the memo holds.
    widened_bytes: u64,
}

impl Walk<'_, '_> {
    fn node(&mut self, ast: &Ast) -> NodeCost {
        match ast {
            Ast::Number(_) | Ast::Constant { .. } | Ast::Str(_) | Ast::CellRef(_) => NodeCost::free(ValueShape::Scalar),
            Ast::ChannelRef(name) => self.channel_ref(name),
            Ast::Unary { expr, .. } => {
                let c = self.node(expr);
                let out = c.shape.bytes();
                NodeCost { shape: c.shape, out, peak: c.peak.max(c.out + out) }
            }
            Ast::Binary { left, right, .. } => {
                let l = self.node(left);
                let r = self.node(right);
                let shape = ValueShape::widest(l.shape, r.shape);
                let out = shape.bytes();
                let peak = l.peak.max(l.out + r.peak).max(l.out + r.out + out);
                NodeCost { shape, out, peak }
            }
            Ast::Call { name, args, kwargs } => self.call(name, args, kwargs),
        }
    }

    fn channel_ref(&mut self, name: &str) -> NodeCost {
        if let Some(&p) = self.sim.primary.get(name) {
            // A definition: its value is shared, never copied.
            return NodeCost::free(self.sim.shapes[p]);
        }
        let Some(dims) = self.sim.inputs_dims.dims(name) else {
            // Not in this session: an error, which allocates nothing.
            return NodeCost::free(ValueShape::Scalar);
        };
        let shape = ValueShape::Series { len: dims.samples, rate_hz: dims.rate_hz };
        if dims.sample_bytes > 0 && self.sim.seen_inputs.insert(name.to_string()) {
            self.sim.inputs.push(PlannedInput {
                channel: name.to_string(),
                bytes: dims.samples as u64 * (dims.sample_bytes as u64 + CACHE_AXIS_BYTES_PER_SAMPLE),
            });
        }
        if !self.widened.insert(name.to_string()) {
            return NodeCost::free(shape);
        }
        // The lookup widens the raw column into a `Vec<f64>` and copies it
        // into an `Arc` the memo keeps: two buffers for a moment, one held
        // until the definition returns.
        let bytes = shape.bytes();
        self.widened_bytes += bytes;
        NodeCost { shape, out: 0, peak: bytes }
    }

    fn call(&mut self, name: &str, args: &[Ast], kwargs: &[(String, Ast)]) -> NodeCost {
        let mut held: u64 = 0;
        let mut peak: u64 = 0;
        let mut shapes: Vec<ValueShape> = Vec::with_capacity(args.len());
        for a in args.iter().chain(kwargs.iter().map(|(_, a)| a)) {
            let c = self.node(a);
            peak = peak.max(held + c.peak);
            held += c.out;
            shapes.push(c.shape);
        }
        let positional = args.len();
        let first = shapes.first().copied().unwrap_or(ValueShape::Scalar);
        let widest = shapes.iter().copied().fold(ValueShape::Scalar, ValueShape::widest);
        let time_base = || {
            self.sim
                .inputs_dims
                .time_base
                .map(|(len, rate_hz)| ValueShape::Series { len, rate_hz })
                .unwrap_or(ValueShape::Scalar)
        };
        let rolling = !kwargs.is_empty() || positional >= 2;
        let (shape, alloc) = match name {
            "sum" | "count" | "first" | "last" | "main" | "lap_number" | "lap_time" | "sector_time" => {
                let shape = if matches!(name, "lap_number" | "lap_time" | "sector_time") {
                    ValueShape::Lap
                } else {
                    ValueShape::Scalar
                };
                (shape, OpAlloc::Reduce)
            }
            "median" | "percentile" => (ValueShape::Scalar, OpAlloc::ReduceCopying),
            "mean" | "rms" | "std" if rolling => (first, OpAlloc::Copied),
            "mean" | "rms" | "std" | "min" | "max" if positional == 1 && kwargs.is_empty() => {
                (ValueShape::Scalar, OpAlloc::Reduce)
            }
            "lap_start_time" | "lap_start_distance" => (first, OpAlloc::Direct),
            "butter" => (shapes.get(3).copied().unwrap_or(ValueShape::Scalar), OpAlloc::Filtered),
            "differentiate" | "gradient" | "detrend" | "cumulative_trapezoid" | "cumtrapz" | "declip"
            | "lap_delta_time" | "lap_delta_dist" => (first, OpAlloc::Copied),
            "wheel_travel" | "wheel_velocity" | "attitude" | "body_accel" | "sector_number" => {
                (time_base(), OpAlloc::Copied)
            }
            "current_lap" => (time_base(), OpAlloc::Direct),
            "resample" => (shapes.get(1).copied().unwrap_or(ValueShape::Scalar), OpAlloc::Copied),
            "where" => (widest, OpAlloc::Direct),
            "clip" => (first, OpAlloc::Copied),
            "vec" | "vadd" | "vsub" | "vscale" | "cross" | "normalize" | "rotate_mat" | "rotate_axis"
            | "rotate_euler" => (ValueShape::Vec3 { len: widest.len() }, OpAlloc::Direct),
            "periodogram" | "welch" => (first, OpAlloc::Copied),
            _ => {
                // Element-wise (`abs`, `pow`, two-argument `min`/`max`,
                // `clip`, `vx`, `norm`, …): the widest operand's shape.
                let shape = match widest {
                    ValueShape::Vec3 { len } => ValueShape::Series { len, rate_hz: 0.0 },
                    other => other,
                };
                (shape, OpAlloc::Direct)
            }
        };
        let out = match alloc {
            OpAlloc::Reduce | OpAlloc::ReduceCopying => 0,
            _ => shape.bytes(),
        };
        let input_len_bytes = first.len() as u64 * F64_BYTES;
        let scratch = match alloc {
            OpAlloc::Reduce | OpAlloc::Direct => 0,
            // The finite copy grows by doubling to the next power of two,
            // then the stable sort takes a scratch buffer as long again.
            OpAlloc::ReduceCopying => first.len().next_power_of_two() as u64 * F64_BYTES + input_len_bytes,
            OpAlloc::Copied => out,
            OpAlloc::Filtered => out * FILTER_SCRATCH_BUFFERS,
        };
        peak = peak.max(held + scratch + out);
        NodeCost { shape, out, peak }
    }
}

/// `f64` buffers as long as its output that `butter` holds at its peak
/// besides the output itself (sci-rs's `sosfiltfilt` pass buffers, then the
/// `Vec`-to-`Arc` copy). Measured by
/// `plan_cost_model_each_builtin_family_counted_peak_within_25_percent_of_the_model`.
const FILTER_SCRATCH_BUFFERS: u64 = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workbook::v3::{CellDoc, ConstLine, UnitsPref};

    fn def(name: &str, expr: &str) -> MathCellDef {
        MathCellDef {
            cell_id: "aaaaaaaa".to_string(),
            name: name.to_string(),
            expr_text: expr.to_string(),
            label: None,
            unit_annotation: None,
            order: 0,
        }
    }

    fn inputs(channels: &[(&str, usize)]) -> PlanInputs {
        let index: Vec<ChannelColumnInfo> = channels
            .iter()
            .map(|(name, n)| ChannelColumnInfo {
                channel_id: name.to_string(),
                nominal_rate_hz: 800.0,
                unit: String::new(),
                source_kind: "imu0".to_string(),
                sample_bytes: 2,
                file_rows: *n,
                sample_count: *n,
            })
            .collect();
        PlanInputs::from_channel_index(&index)
    }

    fn cell(id: &str, kind: CellKindToken, body: &str, prose_before: Option<&str>) -> CellDoc {
        CellDoc {
            id: id.to_string(),
            kind_token: kind,
            prose_before: prose_before.map(str::to_string),
            prose_after: None,
            raw_fence_body: body.to_string(),
            table: None,
        }
    }

    fn doc(cells: Vec<CellDoc>, defs: Vec<MathCellDef>) -> WorkbookDoc {
        WorkbookDoc {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Test".to_string(),
            constants_raw: HashMap::new(),
            units_pref: UnitsPref::Si,
            version: 3,
            cells,
            trailing_prose: None,
            const_lines: Vec::<ConstLine>::new(),
            defs,
            constants: HashMap::new(),
            front_matter_unknown: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn schedule_a_definition_before_the_one_it_reads_dependency_first_document_order_otherwise() {
        // Arrange — b reads c, which is declared after it.
        let defs = vec![def("a", "1"), def("b", "[c] + 1"), def("c", "2"), def("d", "3")];

        // Act
        let plan = EvalPlan::new(&defs, &HashMap::new(), &PlanInputs::default(), &Demand::everything());

        // Assert
        assert_eq!(plan.schedule, vec![0, 2, 1, 3]);
        assert_eq!(plan.consumers, vec![0, 0, 1, 0]);
    }

    #[test]
    fn schedule_a_cycle_its_members_come_last_in_document_order() {
        // Arrange
        let defs = vec![def("x", "[y]"), def("ok", "1"), def("y", "[x]")];

        // Act
        let plan = EvalPlan::new(&defs, &HashMap::new(), &PlanInputs::default(), &Demand::everything());

        // Assert
        assert_eq!(plan.schedule, vec![1, 0, 2]);
    }

    #[test]
    fn scope_a_closure_demand_only_the_named_definition_and_its_dependencies() {
        // Arrange
        let defs = vec![def("a", "1"), def("b", "[a] * 2"), def("c", "3"), def("d", "[b] + 1")];

        // Act
        let plan = EvalPlan::new(&defs, &HashMap::new(), &PlanInputs::default(), &Demand::closure(["d".to_string()]));

        // Assert
        assert_eq!(plan.in_scope, vec![true, true, false, true]);
        assert_eq!(plan.schedule, vec![0, 1, 3]);
    }

    #[test]
    fn shapes_reductions_lap_functions_and_element_wise_ops_static_shape_and_length() {
        // Arrange
        let defs = vec![
            def("f", "butter(2, 5, \"low\", [IMU0_AccelX])"),
            def("m", "mean([f])"),
            def("rm", "mean([f], 8)"),
            def("lt", "lap_time()"),
            def("cl", "current_lap()"),
            def("g", "[f] * 9.81 + [m]"),
        ];

        // Act
        let plan =
            EvalPlan::new(&defs, &HashMap::new(), &inputs(&[("IMU0_AccelX", 1000)]), &Demand::everything());

        // Assert
        let series = ValueShape::Series { len: 1000, rate_hz: 800.0 };
        assert_eq!(plan.shapes, vec![series, ValueShape::Scalar, series, ValueShape::Lap, series, series]);
        assert_eq!(plan.inputs, vec![PlannedInput { channel: "IMU0_AccelX".to_string(), bytes: 1000 * 18 }]);
    }

    #[test]
    fn liveness_a_value_with_no_reader_left_is_released_unless_retained() {
        // Arrange — three full-rate values; b and c read a, only c is retained.
        let defs = vec![def("a", "[IMU0_AccelX] * 2"), def("b", "[a] + 1"), def("c", "[a] - 1")];
        let n = 1000u64;
        let keep_c = Demand { scope: None, retain: Retain::Names(["c".to_string()].into()) };

        // Act
        let all = EvalPlan::new(&defs, &HashMap::new(), &inputs(&[("IMU0_AccelX", 1000)]), &Demand::everything());
        let some = EvalPlan::new(&defs, &HashMap::new(), &inputs(&[("IMU0_AccelX", 1000)]), &keep_c);

        // Assert — everything kept: three values live at the end; with
        // release, at most a and one reader (a goes after c, b at once).
        assert_eq!(all.estimate.live_peak_bytes, 3 * 8 * n);
        assert_eq!(some.estimate.live_peak_bytes, 2 * 8 * n);
    }

    #[test]
    fn consumer_names_js_identifiers_prose_spans_and_table_bodies_found_math_only_names_not() {
        // Arrange
        let cells = vec![
            cell("c1", CellKindToken::Math, "a = 1\nb = 2\nc = 3\nd = 4\ne = 5", Some("Peak is ${a.toFixed(1)}, not b.")),
            cell("c2", CellKindToken::Js, "plot(channel(\"c\"), d2)", None),
            cell("c3", CellKindToken::Table, "{\"columns\":[{\"expr\":\"mean([d])\"}]}", None),
        ];
        let defs = vec![def("a", "1"), def("b", "2"), def("c", "3"), def("d", "4"), def("e", "5")];

        // Act
        let names = consumer_names(&doc(cells, defs));

        // Assert — `b` is prose text, not a span; `d2` is not `d`.
        assert_eq!(names.into_iter().collect::<Vec<_>>(), vec!["a", "c", "d"]);
    }

    #[test]
    fn estimate_with_missing_inputs_a_resident_input_costs_nothing_new() {
        // Arrange
        let defs = vec![def("a", "sum([IMU0_AccelX])")];
        let plan = EvalPlan::new(&defs, &HashMap::new(), &inputs(&[("IMU0_AccelX", 1000)]), &Demand::everything());

        // Act
        let cold = plan.estimate_with_missing_inputs(|_| true);
        let warm = plan.estimate_with_missing_inputs(|_| false);

        // Assert
        assert_eq!(cold - warm, 1000 * 18 * 5 / 4);
        assert_eq!(warm, plan.estimate.step_peak_bytes * 5 / 4);
    }

    // ---- the memory fixtures (evaluation-memory spec §6, Phase 2) ----

    use crate::math::eval::{ChannelLookup, LookupChannel, MathLapContext};
    use std::sync::Arc;

    /// A session for the memory fixtures: raw `i16` columns widened on every
    /// lookup into a fresh `Arc<[f64]>`, as the app's lazy handle does
    /// (`SessionHandle::lookup`'s `Arc::from(try_materialize())`), and one
    /// `t_us` per source shared by that source's channels, as the cache does.
    struct RawSession {
        channels: HashMap<String, (Vec<i16>, f64, Arc<[i64]>)>,
        time_base: (usize, f64),
    }

    impl ChannelLookup for RawSession {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            let (raw, rate, t_us) = self.channels.get(name)?;
            let widened: Vec<f64> = raw.iter().map(|&r| r as f64 * 0.001).collect();
            Some(LookupChannel { samples: Arc::from(widened), sample_rate_hz: *rate, t_us: t_us.clone() })
        }

        fn best_time_base_dims(&self) -> Option<(usize, f64)> {
            Some(self.time_base)
        }

        fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
            self.channels.get(name).map(|(raw, rate, _)| (raw.len(), *rate))
        }
    }

    impl RawSession {
        /// The footer's view of this session, for the planner.
        fn plan_inputs(&self) -> PlanInputs {
            let channels = self
                .channels
                .iter()
                .map(|(name, (raw, rate, _))| {
                    (name.clone(), ChannelDims { samples: raw.len(), rate_hz: *rate, sample_bytes: 2 })
                })
                .collect();
            PlanInputs { channels, time_base: Some(self.time_base) }
        }
    }

    /// Deterministic noise in `-amp..=amp` (a 32-bit LCG).
    fn noise(seed: &mut u32, amp: i32) -> i32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*seed >> 8) % (2 * amp as u32 + 1)) as i32 - amp
    }

    /// Three 800 Hz IMU sources (`n0` samples for IMU0, 1 % and 3 % fewer for
    /// IMU1 and IMU2, as on session `103bc2bf…`) and 1 Hz GPS and HR, over
    /// the same span. Raw counts are milli-g, milli-dps, 0.001 km/h, ….
    fn mx_session(n0: usize) -> RawSession {
        let rate = 800.0;
        let span_us = (n0 as f64 / rate * 1e6) as i64;
        let axis = |n: usize, offset_us: i64| -> Arc<[i64]> {
            (0..n as i64).map(|i| offset_us + i * span_us / n as i64).collect()
        };
        let mut seed = 7u32;
        let mut column = |n: usize, base: i32, amp: i32| -> Vec<i16> {
            (0..n).map(|i| (base + if i % 4000 < 500 { -base } else { 0 } + noise(&mut seed, amp)) as i16).collect()
        };
        let mut channels = HashMap::new();
        for (source, n, offset) in [("IMU0", n0, 0i64), ("IMU1", n0 * 99 / 100, 400), ("IMU2", n0 * 97 / 100, 900)] {
            let t = axis(n, offset);
            for (axis_name, base) in [("AccelX", 50), ("AccelY", 80), ("AccelZ", 980)] {
                channels.insert(format!("{source}_{axis_name}"), (column(n, base, 400), rate, t.clone()));
            }
            if source == "IMU0" {
                for axis_name in ["GyroX", "GyroY", "GyroZ"] {
                    channels.insert(format!("{source}_{axis_name}"), (column(n, 0, 3000), rate, t.clone()));
                }
            }
        }
        let n_slow = (n0 as f64 / rate) as usize;
        let slow = axis(n_slow, 0);
        channels.insert("GPS_SpeedKmh".to_string(), (column(n_slow, 25_000, 10_000), 1.0, slow.clone()));
        channels.insert("GPS_Altitude".to_string(), (column(n_slow, 20_000, 2_000), 1.0, slow.clone()));
        channels.insert("HR_BPM".to_string(), (column(n_slow, 150, 20), 1.0, slow));
        RawSession { channels, time_base: (n0, rate) }
    }

    /// The comparison workbook's full-rate cells (`c0000003`–`c0000007` of
    /// `mx-compare-2026-09-19-vs-2026-10-24.idl1wb`), trimmed to the
    /// definitions that work without detected laps, in their order and with
    /// their text; `n_laps` is a constant here. A `js` cell charts `lat_g`,
    /// `long_g`, `frame_g`, `f_bump` and `r_bump`, as the workbook's do.
    fn mx_workbook() -> WorkbookDoc {
        let lines: &[(&str, &str)] = &[
            ("cl", "current_lap() + 0 * [IMU0_AccelX]"),
            ("on_t", "where(current_lap() >= 2, 1, 0)"),
            ("on_g", "where(resample([cl], [GPS_SpeedKmh]) >= 1.5, 1, 0)"),
            ("on_h", "where(resample([cl], [HR_BPM]) >= 1.5, 1, 0)"),
            ("n_laps", "3"),
            ("v_top", "max(where([on_g] > 0.5, [GPS_SpeedKmh], sqrt(-1)))"),
            ("v_p90", "percentile(where([on_g] > 0.5, [GPS_SpeedKmh], sqrt(-1)), 90)"),
            ("alt_lp", "butter(2, 0.1, \"low\", [GPS_Altitude])"),
            ("hr_mean", "mean(where([on_h] > 0.5, [HR_BPM], sqrt(-1)))"),
            ("speed_lp", "butter(2, 0.4, \"low\", [GPS_SpeedKmh])"),
            ("long_g", "differentiate([speed_lp]) / 3.6 / 9.81"),
            ("ux", "mean([IMU0_AccelX])"),
            ("uy", "mean([IMU0_AccelY])"),
            ("uz", "mean([IMU0_AccelZ])"),
            ("w_up", "butter(2, 2, \"low\", ([IMU0_GyroX] * mean([IMU0_AccelX]) + [IMU0_GyroY] * mean([IMU0_AccelY]) + [IMU0_GyroZ] * mean([IMU0_AccelZ])) / sqrt(mean([IMU0_AccelX]) * mean([IMU0_AccelX]) + mean([IMU0_AccelY]) * mean([IMU0_AccelY]) + mean([IMU0_AccelZ]) * mean([IMU0_AccelZ])))"),
            ("v_ms", "resample([GPS_SpeedKmh], [IMU0_AccelX]) / 3.6"),
            ("k_lean", "[v_ms] * deg2rad([w_up]) / 9.81"),
            ("yaw", "deg2rad([w_up]) / sqrt(1 - [k_lean] * [k_lean])"),
            ("lat_g", "[v_ms] * [yaw] / 9.81"),
            ("on_fast", "where(current_lap() >= 2, where([v_ms] > 4.17, 1, 0), 0)"),
            ("lat_p95", "percentile(where([on_fast] > 0.5, abs([lat_g]), sqrt(-1)), 95)"),
            ("long_t", "resample([long_g], [IMU0_AccelX])"),
            ("comb_p95", "percentile(where([on_fast] > 0.5, sqrt([lat_g] * [lat_g] + [long_t] * [long_t]), sqrt(-1)), 95)"),
            ("rate0", "count([IMU0_AccelX]) / (last([IMU0_AccelX]) - first([IMU0_AccelX]))"),
            ("ax_lp", "butter(2, 12, \"low\", [IMU0_AccelX])"),
            ("ay_lp", "butter(2, 12, \"low\", [IMU0_AccelY])"),
            ("az_lp", "butter(2, 12, \"low\", [IMU0_AccelZ])"),
            ("frame_g", "sqrt([ax_lp] * [ax_lp] + [ay_lp] * [ay_lp] + [az_lp] * [az_lp])"),
            ("air", "where([frame_g] < 0.45, 1, 0)"),
            ("air_lap", "sum([on_t] * [air]) / [rate0] / [n_laps]"),
            ("f25", "where(mean([air], 202) > 0.999, 1, 0)"),
            ("f50", "where(mean([air], 403) > 0.999, 1, 0)"),
            ("e25", "[on_t] * where(differentiate([f25]) > 0, 1, 0)"),
            ("e50", "[on_t] * where(differentiate([f50]) > 0, 1, 0)"),
            ("n25", "sum([e25]) / [n_laps]"),
            ("v_to50", "3.6 * sum([e50] * [v_ms]) / sum([e50])"),
            ("pitch_lp", "butter(2, 5, \"low\", [IMU0_GyroY])"),
            ("pitch_air", "sum([on_t] * [f25] * [pitch_lp]) / [rate0] / sum([e25])"),
            ("on1", "where(resample([cl], [IMU1_AccelX]) >= 1.5, 1, 0)"),
            ("on2", "where(resample([cl], [IMU2_AccelX]) >= 1.5, 1, 0)"),
            ("f_vert", "([IMU1_AccelX] * mean([IMU1_AccelX]) + [IMU1_AccelY] * mean([IMU1_AccelY]) + [IMU1_AccelZ] * mean([IMU1_AccelZ])) / sqrt(mean([IMU1_AccelX]) * mean([IMU1_AccelX]) + mean([IMU1_AccelY]) * mean([IMU1_AccelY]) + mean([IMU1_AccelZ]) * mean([IMU1_AccelZ]))"),
            ("r_vert", "([IMU2_AccelX] * mean([IMU2_AccelX]) + [IMU2_AccelY] * mean([IMU2_AccelY]) + [IMU2_AccelZ] * mean([IMU2_AccelZ])) / sqrt(mean([IMU2_AccelX]) * mean([IMU2_AccelX]) + mean([IMU2_AccelY]) * mean([IMU2_AccelY]) + mean([IMU2_AccelZ]) * mean([IMU2_AccelZ]))"),
            ("c_vert", "([IMU0_AccelX] * mean([IMU0_AccelX]) + [IMU0_AccelY] * mean([IMU0_AccelY]) + [IMU0_AccelZ] * mean([IMU0_AccelZ])) / sqrt(mean([IMU0_AccelX]) * mean([IMU0_AccelX]) + mean([IMU0_AccelY]) * mean([IMU0_AccelY]) + mean([IMU0_AccelZ]) * mean([IMU0_AccelZ]))"),
            ("f_bump", "butter(2, 100, \"low\", butter(2, 5, \"high\", [f_vert]))"),
            ("r_bump", "butter(2, 100, \"low\", butter(2, 5, \"high\", [r_vert]))"),
            ("c_bump", "butter(2, 60, \"low\", butter(2, 2, \"high\", [c_vert]))"),
            ("f_rms", "sqrt(sum([on1] * [f_bump] * [f_bump]) / sum([on1]))"),
            ("r_rms", "sqrt(sum([on2] * [r_bump] * [r_bump]) / sum([on2]))"),
            ("c_rms", "sqrt(sum([on_t] * [c_bump] * [c_bump]) / sum([on_t]))"),
            ("f_mag", "sqrt([IMU1_AccelX] * [IMU1_AccelX] + [IMU1_AccelY] * [IMU1_AccelY] + [IMU1_AccelZ] * [IMU1_AccelZ])"),
            ("f_p999", "percentile(where([on1] > 0.5, [f_mag], sqrt(-1)), 99.9)"),
            ("f1", "resample([f25], [IMU1_AccelX])"),
            ("cl1", "resample([cl], [IMU1_AccelX])"),
            ("f_lp", "sqrt(butter(2, 12, \"low\", [IMU1_AccelX]) * butter(2, 12, \"low\", [IMU1_AccelX]) + butter(2, 12, \"low\", [IMU1_AccelY]) * butter(2, 12, \"low\", [IMU1_AccelY]) + butter(2, 12, \"low\", [IMU1_AccelZ]) * butter(2, 12, \"low\", [IMU1_AccelZ]))"),
            ("f_air_early", "median(where([f1] > 0.5, where([cl1] < 4.5, [f_lp], sqrt(-1)), sqrt(-1)))"),
        ];
        let defs: Vec<MathCellDef> =
            lines.iter().enumerate().map(|(i, (name, expr))| MathCellDef { order: i, ..def(name, expr) }).collect();
        let cells = vec![
            cell("aaaaaaaa", CellKindToken::Math, "", None),
            cell("bbbbbbbb", CellKindToken::Js, "plot([lat_g, long_g, frame_g]); plot(channel(\"f_bump\"), r_bump)", None),
        ];
        doc(cells, defs)
    }

    /// The counted peak of evaluating `doc` over `session` under `plan`, and
    /// the cells.
    fn counted(doc: &WorkbookDoc, session: &RawSession, plan: &EvalPlan) -> (Vec<super::super::CellEvalResult>, usize) {
        // Five 30 s laps over the session's 150 s, so lap 2+ windows are real.
        let lap_ctx = MathLapContext {
            main_lap_bounds: (0..5).map(|i| (i as f64 * 30.0, (i + 1) as f64 * 30.0)).collect(),
            ..MathLapContext::empty()
        };
        let (cells, peak) = crate::counting_alloc::peak_of(|| {
            super::super::eval_cells_planned(doc, &[], session, &lap_ctx, plan, None).unwrap()
        });
        (cells, peak)
    }

    #[test]
    fn plan_mx_compare_shaped_fixture_estimate_within_25_percent_of_the_counted_peak() {
        // Arrange — the session is decoded before the evaluation starts, so
        // no input is missing and the estimate is the evaluation's own.
        let session = mx_session(120_000);
        let wb = mx_workbook();
        let plan = EvalPlan::new(&wb.defs, &wb.constants, &session.plan_inputs(), &Demand::document(&wb));

        // Act
        let (cells, peak) = counted(&wb, &session, &plan);

        // Assert
        let estimate = plan.estimate_with_missing_inputs(|_| false) as f64 / 1.25;
        let off = (estimate - peak as f64) / peak as f64;
        eprintln!("mx fixture: estimate {estimate:.0} B, counted peak {peak} B, off {:+.1} %", off * 100.0);
        assert!(off.abs() <= 0.25, "estimate {estimate} B is {:+.1} % off the counted peak {peak} B", off * 100.0);
        assert!(cells[0].defs.iter().all(|d| d.error.is_none()), "{:?}", cells[0].defs.iter().find(|d| d.error.is_some()));
    }

    #[test]
    fn plan_mx_compare_shaped_fixture_counted_peak_at_most_i_plus_f_plus_t_times_1_1() {
        // Arrange
        let session = mx_session(120_000);
        let wb = mx_workbook();
        let plan = EvalPlan::new(&wb.defs, &wb.constants, &session.plan_inputs(), &Demand::document(&wb));

        // Act
        let (_, peak) = counted(&wb, &session, &plan);

        // Assert — inputs are resident before the evaluation, so I = 0 here.
        let bound = (plan.estimate.live_peak_bytes + plan.estimate.transient_bytes) as f64 * 1.1;
        assert!((peak as f64) <= bound, "counted peak {peak} B is over (I + F + T) × 1.1 = {bound:.0} B");
    }

    #[test]
    fn plan_mx_compare_shaped_fixture_liveness_and_retain_cut_the_peak_below_keeping_every_value() {
        // Arrange
        let session = mx_session(120_000);
        let wb = mx_workbook();
        let keep_all = EvalPlan::new(&wb.defs, &wb.constants, &session.plan_inputs(), &Demand::everything());
        let planned = EvalPlan::new(&wb.defs, &wb.constants, &session.plan_inputs(), &Demand::document(&wb));

        // Act
        let (all_cells, before) = counted(&wb, &session, &keep_all);
        let (cells, after) = counted(&wb, &session, &planned);

        // Assert — same answers; under 60 % of the memory; the five charted
        // values kept, the rest released with their lengths.
        eprintln!("mx fixture: keep-everything peak {before} B, planned peak {after} B");
        assert!(after * 10 < before * 6, "planned peak {after} B is not under 60 % of {before} B");
        for (a, b) in all_cells[0].defs.iter().zip(cells[0].defs.iter()) {
            let (a, b) = (a.value.as_ref().unwrap(), b.value.as_ref().unwrap());
            assert_eq!(a.length, b.length);
            if !b.released {
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a.v), bits(&b.v));
            }
        }
        let kept: Vec<&str> =
            cells[0].defs.iter().filter(|d| d.value.as_ref().is_some_and(|v| !v.released && v.length > 1)).map(|d| d.name.as_str()).collect();
        assert_eq!(kept, vec!["long_g", "lat_g", "frame_g", "f_bump", "r_bump"]);
    }

    #[test]
    fn plan_cost_model_each_builtin_family_counted_peak_within_25_percent_of_the_model() {
        // Arrange — one definition per allocation family, over 100,000 samples.
        let session = mx_session(100_000);
        let families = [
            "[IMU0_AccelX] * 2",
            "abs([IMU0_AccelX])",
            "[IMU0_AccelX] * [IMU0_AccelY] + [IMU0_AccelZ]",
            "butter(2, 12, \"low\", [IMU0_AccelX])",
            "mean([IMU0_AccelX], 8)",
            "differentiate([IMU0_AccelX])",
            "where([IMU0_AccelX] > 0, 1, 0)",
            "resample([GPS_SpeedKmh], [IMU0_AccelX])",
            "percentile([IMU0_AccelX], 90)",
            "sum([IMU0_AccelX])",
            "current_lap() + 0 * [IMU0_AccelX]",
        ];

        for expr in families {
            let wb = doc(vec![cell("aaaaaaaa", CellKindToken::Math, "", None)], vec![def("x", expr)]);
            let plan = EvalPlan::new(&wb.defs, &wb.constants, &session.plan_inputs(), &Demand::everything());

            // Act
            let (_, peak) = counted(&wb, &session, &plan);

            // Assert
            let model = plan.estimate.step_peak_bytes as f64;
            let off = (model - peak as f64) / peak as f64;
            eprintln!("{expr}: model {model:.0} B, counted {peak} B, off {:+.1} %", off * 100.0);
            assert!(off.abs() <= 0.25, "{expr}: model {model} B is {:+.1} % off the counted {peak} B", off * 100.0);
        }
    }
}
