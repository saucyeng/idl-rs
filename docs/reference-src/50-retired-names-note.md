## Naming policy

Where a function is semantically equivalent to a scipy or numpy one, it takes
that name. Where it is **not** equivalent, it takes a deliberately different
name. A familiar name with unfamiliar behaviour is worse than an invented one,
because an invented name makes a reader check and a false friend does not.

That is why the retired-name table above exists, and why two known false
friends were left in place rather than renamed:

- **`butter`** designs *and* applies its filter, zero-phase, in one call.
  scipy's `butter` only designs one; applying it is a separate call. Splitting
  the two here is a real behaviour change with its own migration, not a naming
  fix, so it is recorded rather than done.
- **`round`** rounds half away from zero (`2.5` gives `3`), where `numpy.round`
  rounds half to even (`2.5` gives `2`). Changing it would silently move every
  existing value sitting exactly on a halfway point.

Neither is a promise that either name changes soon. Both are documented in
their own entries above.
