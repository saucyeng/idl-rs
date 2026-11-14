# idl-library

idl-library is the ride library behind idl-rs. It keeps a searchable catalog of every ride, stores each original log file once, and handles import jobs, the bike log, and sync between devices. The idl-rs engine works on one session at a time, so reach for idl-library if you are building your own app or logger software that needs to manage hundreds of rides.

## What's inside

- `catalog`: the SQLite index of sessions, laps and tracks. It is rebuildable from the files on disk, so it can be deleted at any time.
- `blob`: the content-addressed store of original log files.
- `import`, `index_job`, `set_job`: the import, re-index and metadata-edit jobs.
- `bikelog`: one file per bike holding its notes and setup sheets, which merge across devices without losing an edit.
- `sync`: the LAN-sync model between devices.

The `idl-rs` command ([idl-rs-cli](https://crates.io/crates/idl-rs-cli)) is built on this crate and is the quickest way to try it.

Part of the [idl-rs repository](https://github.com/saucyeng/idl-rs). AGPL-3.0-or-later.
