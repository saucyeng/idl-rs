## Windows and laps

Everything a workbook evaluates is evaluated over a **selection**, and a
selection is a list of *windows*. A window is one span of one session. Two
windows over the same channel is the ordinary case, not the exotic one: it is
how you compare one lap against another.

### What a window changes

An aggregate (`mean`, `rms`, `percentile`, `count`) is computed over the
selected span, not the whole session. A per-sample function is computed over
the samples inside it. Nothing about a definition's text changes when the
selection changes — a host variable is keyed by the definition alone, never by
which windows are selected, so a cell's code never depends on what is currently
clicked.

### Several windows in one payload

When more than one window is selected, `channel(name)` returns *every* selected
window's samples in one payload rather than one payload per window:

- the records are each window's own samples, window after window, in window
  order.
- `w[i]` is the index into `windows` naming which window produced sample `i`.
- `windows[j]` is that window's `{ sessionId, span, colour, label }`. The
  `colour` is a `--chart-1`…`--chart-8` token, which is how a per-window colour
  reaches a mark.

Exactly one `NaN` record (`t = v = w = NaN`) is inserted between each adjacent
pair of windows. Observable Plot breaks a line mark at a `NaN`, so a cell that
destructures only `{t, v}` and knows nothing about `w` still draws *n* separate
segments instead of one line vaulting from one window's last sample to the
next window's first.

A single selected window is byte-identical to having no window concept at all:
`w` is all zeros and `windows` has one entry.

### Laps

`laps` is the active session's lap table — `{ number, startT, endT }[]`, with
`number` 1-based.

The lap functions divide into three groups:

- **Where am I?** `current_lap()` gives the 1-based lap number at each sample
  and `0` outside any lap; `sector_number()` gives the 0-based sector index and
  NaN outside any sector. Both are per-sample channels at the session's base
  rate, so both are usable as a `where(...)` condition over a channel at that
  same rate. A slower channel — GPS at 1 Hz, say — has to be brought onto the
  base rate first, because nothing resamples implicitly:

  ```
  where(sector_number() == 3, resample([GPS_SpeedKmh], [Time]), 0)
  ```

  `[Time]` is the synthesized base-rate channel, so the resampled speed lands
  on the axis `sector_number()` counts along. Written without the `resample`,
  the same expression is a typed error naming the two rates — never a wrong
  answer.
- **Where does a lap start?** `lap_start_time(n)` and `lap_start_distance(n)`
  are scalars, NaN when `n` is out of range (and `lap_start_distance` is also
  NaN when the session has no `[Distance]` channel).
- **How does this lap differ?** `lap_delta_time(ch)` and `lap_delta_dist(ch)`
  subtract an overlay lap from the main lap — time-matched and
  arc-length-matched respectively — and take the mean across every overlay when
  more than one is selected. Both carry `ch`'s own units: they are a difference
  of two `[ch]` series, not a time or a distance.

`channel(name, { lap: n })` fixes a lookup to one lap, and
`channel(name, { from: a, to: b })` to a stretch of session seconds; see
[The data a chart gets](#the-data-a-chart-gets). A different session is
reached by selecting its window, not from cell code.

### The lap-relative time axis

A time chart's x binding is either `"t"` (session time) or `"tr"` (time
relative to the start of each window). Use `"tr"` when comparing laps: session
time puts two laps a minute apart on the x axis, which is never what a lap
comparison wants. The binding is stored per mark and edited per plot.
