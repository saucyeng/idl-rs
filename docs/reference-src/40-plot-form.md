## The chart grammar (`plotForm`)

The Properties panel does not own a chart's configuration — the `js` cell's own
code does. The panel reads the code, and writing a control writes the code
back. That round-trip works over one narrow subset of `Plot.plot(...)`, and a
cell outside the subset is shown as **Custom**: it still renders, it simply
stops being editable by control.

Staying inside the subset is worth it while a chart is one of the four kinds
below; step outside it deliberately, for something the panel could never offer.

### The shape of an editable cell

```js
Plot.plot({
  title: "Fork travel",
  x: { label: "Time (s)", domain: [0, 60], type: "linear" },
  y: { label: "Travel (mm)", domain: [0, 160], type: "linear" },
  color: { legend: true },
  marks: [ /* one of the four mark sets below */ ]
})
```

Every option is optional, but each one that is present must take exactly one of
the forms listed. `x.type` is `"linear"` or `"log"`. `y.type` adds `"sqrt"` and
`"pow"`; `"pow"` alone may carry an `exponent`.

### The four mark sets

A cell's `marks` array is one of these four, never a mixture.

**Time.** One or more marks over `channel(...)`, optionally led by a zero rule.

```js
marks: [
  Plot.ruleY([0]),
  Plot.lineY(channel("front_travel"), { x: "tr", y: "v", stroke: "#e07a3f", strokeWidth: 1.5 })
]
```

Mark names: `lineY`, `dot`, `areaY`, `rectY`, `ruleY`. The x binding is `"t"`
(session time) or `"tr"` (relative to each window's start); `y` is always
`"v"`. A `channel(...)` call may carry `{ lap: n }`.

There is no distance binding, by design.

**Spectrum.** Exactly one mark over `spectrum(...)`, with all six parameters
present in this order:

```js
marks: [
  Plot.lineY(spectrum("front_travel", {
    windowSize: 1024, hopSize: 512, window: "hann",
    detrend: "mean", scaling: "density", averaging: "mean"
  }), { x: "f", y: "m" })
]
```

`windowSize` and `hopSize` are integers or `"all"`. `window` is
`"rectangular" | "hann" | "hamming"`. `detrend` is `"none" | "mean" | "linear"`.
`scaling` is `"density" | "spectrum" | "raw_magnitude"` — the maths language's
own three names. `averaging` is `"none" | "mean" | "median" | "max"`.

The older spelling `"magnitude"` is still read, so a cell written before the
rename parses and round-trips; it is never offered by a picker and never newly
written.

**Histogram.** Exactly one `Plot.rectY` over `histogram(...)`, four parameters,
fixed order:

```js
marks: [
  Plot.rectY(histogram("fork_velocity", {
    binMode: "width", binValue: 25, symmetric: true, normalise: "fraction"
  }), { x1: "v0", x2: "v1", y: "n" })
]
```

`binMode` is `"count"` (a bin count) or `"width"` (a bin width in the channel's
units). `normalise` is `"counts"` or `"fraction"`.

**Scatter.** Exactly one `Plot.dot` over `scatter(...)`, two parameters:

```js
marks: [
  Plot.dot(scatter("accel_lat", "accel_long", {
    pointBudget: 20000, equalAspect: true
  }), { x: "x", y: "y" })
]
```

`equalAspect` squares the domain, which is what a G-G plot wants. When several
windows are shown, they share one squared domain rather than each computing its
own.

### What makes a cell Custom

Anything the grammar above does not list: a mark name outside the set, a mixed
`marks` array, an option the subset has no production for, a computed value
where a literal is required, or any code around the `Plot.plot(...)` call. The
round-trip is byte-identical for code the panel itself wrote, so a cell that
goes Custom did so because of an edit, not because of a reformat.
