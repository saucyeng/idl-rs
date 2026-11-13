## JavaScript host variables

A `js` cell is an Observable Runtime cell. The host injects the variables
below into the module scope before any cell runs. None of them crosses IPC
from inside cell code — the host resolves every one ahead of time and feeds
the results in, which is why a cell never awaits the engine.

| Variable | Shape |
|---|---|
| one per `math` definition, **by name** | For a time-axis definition, the same records `channel(name)` returns, over the whole selected window and decimated to a few thousand points. A rank-0 value binds a bare number; a rank-1 value on some other axis keys its records by that axis (`lap`, `f`) instead of `t`; a rank ≥ 2 value binds `{ shape, axes, v }`. |
| `channel(name, opts?)` | An array of `{ t, v, tr, w }` records, one per sample, plus a `windows` descriptor. `opts` is a scope: `{ lap: n }` or `{ from: a, to: b }` (session seconds, number literals). See [The data a chart gets](#the-data-a-chart-gets). |
| `spectrum(name, params)` | The FFT payload for a channel, keyed by `spectrumKey(channelId, fftParams)`. |
| `histogram(name, params)` | The binned payload for a channel. |
| `scatter(xChannel, yChannel, params)` | The point cloud for a channel pair. |
| `laps` | `{ number, startT, endT }[]` for the active session. |
| `session` | `{ id, name?, timestampUtcMs }`. |
| `constants` | `{ [name]: number }` — front matter and every `const` line, flattened. Keys may contain spaces. (Before 2026-11-05 this was always empty.) |
| `Plot` | `@observablehq/plot` 0.6.17, bundled. |
| `d3` | `d3` 7.9.0, bundled. |
| `Inputs` | `@observablehq/inputs` 0.12.0, bundled. |
| `html` | The standard tagged-template helper. |

Every library is bundled with the app. There is no CDN, ever, so a cell that
reaches for a URL will not load.

### The data a chart gets

`channel(name)` returns an array of records, one per sample:

| Field | Meaning |
|---|---|
| `t` | Seconds since the session's first sample. |
| `v` | The value, in its own unit. |
| `tr` | Seconds since this sample's own window began. |
| `w` | Index into `windows`: which selected window the sample came from. |

A mark addresses the fields by name:

```js
Plot.plot({
  marks: [Plot.lineY(channel("front_travel"), { x: "t", y: "v" })]
})
```

**It is decimated for the chart.** A chart never gets every sample of a long
window: the data is planned for the chart's span and width, a min/max envelope
of about two points per pixel column, so every spike survives but a 0.1 s event
in a whole ride is a few pixels wide. Zooming the chart (or narrowing the
selected window on the timeline) fetches the finer detail for the new span.

**A scope fixes a call's span.** `channel(name, { from: a, to: b })` covers
session seconds `a` to `b`, and `channel(name, { lap: n })` covers lap `n`.
That data is planned for that span at the chart's width, so a short range
arrives at full detail, and it stays put when the chart is panned or zoomed.
The numbers must be literals: the host reads them before the cell runs. Two
calls over different ranges of one channel are two separate pieces of data.

```js
const zone = channel("front_v", { from: 5827.3, to: 5829.8 });
return Plot.plot({ marks: [Plot.lineY(zone, { x: "t", y: "v" })] });
```

A definition is still **evaluated** over the whole selected window; a scope
only chooses which part of the result is sent. A definition that reads data
outside the range (a calibration stretch elsewhere in the ride) is unaffected.

**Event lists come from the maths, not the chart.** A list of moments (every
pulse gap over a threshold, every stop) is a number result, so compute it with
`keep`, which returns only the matching samples as a short channel; it arrives
exact, whatever the decimation:

```
breaks = keep([gap_ratio], [gap_ratio] > 1.8)
```

The records are built from the bytes the engine transferred; treat them as
read-only.

### Units on a channel

A `channel(...)` result carries its unit as two **non-enumerable** properties,
so neither appears in a `for...of` over the samples nor in Plot's column
inference:

| Property | Meaning |
|---|---|
| `.unit` | The unit as text, ready to splice into prose. Empty when the value is dimensionless. |
| `.unitState` | `"known"`, `"dimensionless"` or `"unknown"`. |

The unit comes from whichever source resolved the value — a definition's own
result, or a raw channel's recorded unit. It never comes from the transferred
sample bytes, which carry samples and framing only.

### Colouring by window

`w` is an index into `windows`, so a multi-window chart colours itself by
stroking on `"w"` and building the range from each window's own colour token:

```js
Plot.plot({
  color: { legend: true },
  marks: [Plot.lineY(channel("front_travel"), { x: "tr", y: "v", stroke: "w" })]
})
```
