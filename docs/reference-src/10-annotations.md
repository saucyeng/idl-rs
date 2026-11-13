## Definitions and annotations

A `math` cell's body is a sequence of lines. Blank lines and `#` comments are
ignored; everything else is either a constant or a definition.

```
const sag_target = 0.3          # a workbook-wide constant
fork_velocity = differentiate([Fork])   # a definition
```

A definition's name on the left of `=` must be a valid identifier —
`[A-Za-z_][A-Za-z0-9_]*` — because every definition is also bound as a
JavaScript variable of that name inside `js` cells. `const` is reserved and
cannot be a definition name.

A `[Channel Name]` reference on the right of `=` is captured verbatim, spaces
included, so a raw session channel whose name has a space is still reachable.
A `[Name]` naming another definition must match that definition's identifier
exactly.

### The three annotations

A trailing `#` comment is an ordinary comment with three exceptions. Each is
the literal word, a colon, then free text to the end of the line.

| Annotation | What it does |
|---|---|
| `# label: <text>` | The display name. What the channel list, chart legends and Properties controls show instead of the identifier. Without one, the UI falls back to the identifier itself. |
| `# unit: <text>` | Overrides the inferred unit for this definition. Use it when the engine's inference cannot reach a unit it should have — never to relabel a value whose real unit is different, because a wrong unit is worse than none. |
| `# shape: <text>` | Declares the value's shape when the expression's own shape is ambiguous. |

```
fork_velocity = differentiate([Fork])   # label: Fork velocity
compression = -fork_velocity            # label: Compression  # unit: mm/s
```

### Constants

Constants live in one flat, workbook-wide namespace fed from two places: the
front matter's `constants:` map, and `const name = value` lines in any `math`
cell. A name declared in both is a per-cell error, not a silent win for either.

A `const` line's name must be an identifier. A front-matter constant's name
need not be — it is only ever read from JavaScript as `constants["rider mass"]`,
never as a bare variable — so a spaced or free-text name is expressible in YAML
only.

A front-matter constant may carry a unit as a string: `rider_mass_kg: "82 kg"`
evaluates to `82`, and the `kg` is display metadata. It is never dimensionally
checked and never converted.

### Units, and the three states

Every value the engine produces carries one of three unit states, and the
distinction is deliberate: a value that *has* no unit and a value whose unit
could not be worked out are different claims.

| State | What it means |
|---|---|
| `known` | The unit was inferred (or declared with `# unit:`) and is shown. |
| `dimensionless` | The value genuinely has no unit — a count, a ratio, a sign. |
| `unknown` | The unit could not be inferred. A CSV channel with no declared unit, or `pow(x, [n])` with a non-literal exponent. |

Nothing is ever labelled with a guess. A prose `${…}` span does **not**
auto-append a unit either, because the span is an arbitrary expression whose
value may not be the quantity the unit describes — write
`${peak_travel} ${channel("fork").unit}` when you want it.
