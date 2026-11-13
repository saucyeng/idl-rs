## Lap tables and progression

Two pictures answer "how did each lap compare?": a **lap table**, one row per
lap and one column per measurement, and a **lap progression**, the same
measurement drawn against lap number. Both are built on one idea — a value
with one entry per lap rather than one per sample.

### Per-lap values

A definition's value carries a shape. Most are a series over time, written
`[t]` — one number per sample, each keeping its own recorded timestamp. A
per-lap value is written `[lap]`: one number per lap, and its coordinate is
the 1-based lap number, not a time.

Three builtins read the lap a value belongs to:

| Function | In a table row | In a `math` cell |
|---|---|---|
| `lap_time()` | that row's lap time, seconds | a `[lap]` value — every lap's time |
| `sector_time(i)` | that row's time in sector `i`, **0-based** | a `[lap]` value — every lap's sector `i` |
| `lap_number()` | that row's 1-based lap number | a `[lap]` value of the lap numbers |

`lap_time()` reads the recorded lap time, so a neutralised zone that was
subtracted when the session was recorded stays subtracted here.

A `[t]` series becomes a `[lap]` one by reducing it per lap:

```
peak_freq_by_lap = mean([peak_freq], "t:lap")   # shape: [lap]
lap_time_s       = lap_time()                   # shape: [lap]
```

Combining a `[lap]` value with a `[t]` one is an error naming both shapes —
lap 3 and second 3 are not the same coordinate, and the alternative to the
error is a plausible-looking wrong answer.

### Drawing a lap progression

A lap chart is its own cell kind. Its single mark takes a **bare identifier**
— a `[lap]` definition — never a `channel(...)` call, because a per-lap value
is always something a definition computed:

```js
Plot.plot({
  x: { label: "Lap" },
  y: { label: "Lap time (s)" },
  marks: [ Plot.lineY(lap_time_s, { x: "lap", y: "v", z: "w" }) ]
})
```

`x` binds `"lap"`, never `"t"`: a `[lap]` value reaches JavaScript keyed by
its own axis, as `{ lap, v, w }`. A cell that asked for `"t"` would find
nothing rather than plot lap 3 at three seconds.

`z: "w"` separates the series by window, so selecting three laps of two
sessions draws one line per window in that window's own colour. Lap numbers
repeating across sessions is correct — x is an ordinal, and `w` keeps the
lines apart. There is no decimation and no cap: laps per session are tens.

Naming a definition that is not `[lap]`-shaped is reported on the cell, with
both shapes named and the reduction to write instead.

### The lap table

A lap table is a `table` cell, not a chart. Its rows follow the selection:

```json
{
  "rowSource": "windowLaps",
  "mainRowId": "fastest",
  "columns": [
    { "id": "c0", "name": "Lap time", "template": "lap_time()" },
    { "id": "c1", "name": "Sector 1", "template": "sector_time(0)" },
    { "id": "c2", "name": "Fork max", "template": "max([Fork travel])" }
  ],
  "rows": [],
  "cells": []
}
```

`rowSource: "windowLaps"` derives one row per lap of each selected window, in
window order then lap order, and each row evaluates **in its own lap** — so
`max([Fork travel])` in row 3 is lap 3's maximum, not the session's. Authored
`rows` are ignored while it is set, and every cell of a derived row evaluates
its column's `template`; a column with no template renders empty.

`mainRowId` names the row `main({col[]})` compares against. The literal
`"fastest"` is reserved for derived rows: the row with the smallest recorded
`lap_time()`, skipping any lap with no recorded time, ties going to the first.
It is highlighted in the grid. Under `rowSource: "authored"` the word is a
validation error rather than a silently unset baseline — an authored table's
rows are named, so a magic id there would shadow a real row id.

Column headers show the column's name and, where it can be stated, its unit:
`lap_time()` and `sector_time(i)` are seconds. A column whose template is an
arbitrary expression shows no unit rather than a guessed one.

A cell that fails shows its own error and nothing else in the grid changes —
one lap with no samples never blanks the other laps' numbers.

### What is not here yet

Cross-session comparison is a selection capability, not a grammar one:
nothing in either form above changes when it lands. Rank-2 values (a
spectrogram matrix) have no lap form — reduce them to a series or a per-lap
value first.
