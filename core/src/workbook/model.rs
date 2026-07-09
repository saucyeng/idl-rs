//! The portable analysis workbook as the engine sees it. `math_channels` and
//! table blocks (via [`Workbook::tables`]) are surfaced; charts / axes / layout
//! are display state and are not modeled (serde drops the unknown fields). See
//! the shipped app format.

use crate::config::VersionedConfig;
use crate::math::channel_def::MathChannelDef;
use crate::overlay::model::OverlayLayout;
use crate::table::model::TableModel;

/// Highest `workbook_version` this engine can read. v2 added the additive
/// `overlay_layouts` field (SPEC §33.1).
pub const SUPPORTED_WORKBOOK_VERSION: u32 = 2;

fn default_workbook_version() -> u32 {
    1
}

/// A workbook's engine-relevant contents: identity, version, the math channels
/// to evaluate, and (via [`Workbook::tables`]) its table blocks.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Workbook {
    /// Stable UUIDv4 from the `.idl0wb`.
    pub workbook_id: String,
    /// Display name.
    pub name: String,
    /// Math-channel definitions; empty when the field is absent.
    #[serde(default)]
    pub math_channels: Vec<MathChannelDef>,
    /// Schema version; defaults to 1 when absent (mirrors the Dart loader).
    #[serde(default = "default_workbook_version")]
    pub workbook_version: u32,
    /// Worksheets — parsed only far enough to extract table blocks (charts and
    /// layout are ignored). Crate-internal; tables are surfaced via
    /// [`Workbook::tables`].
    #[serde(default)]
    pub(crate) worksheets: Vec<WorksheetRaw>,
    /// Overlay layouts (SPEC §33.1); empty when absent (v1 files).
    #[serde(default)]
    pub overlay_layouts: Vec<OverlayLayout>,
}

impl VersionedConfig for Workbook {
    const SUPPORTED_VERSION: u32 = SUPPORTED_WORKBOOK_VERSION;
    const LABEL: &'static str = "workbook";
    fn version(&self) -> u32 {
        self.workbook_version
    }
}

/// A worksheet as the engine reads it for table extraction — only `name` and
/// `blocks` matter; everything else is ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct WorksheetRaw {
    #[serde(default)]
    name: String,
    #[serde(default)]
    blocks: Vec<BlockRaw>,
}

/// A worksheet block. `placement` / `overlay_target_id` are passthrough layout
/// metadata; `content` is dispatched on its `kind`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockRaw {
    #[serde(default)]
    id: String,
    #[serde(default)]
    placement: String,
    #[serde(default)]
    overlay_target_id: Option<String>,
    content: BlockContentRaw,
}

/// Block content; only `table` is modelled — chart/unknown fall to `Other`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum BlockContentRaw {
    Table {
        table: TableModel,
    },
    #[serde(other)]
    Other,
}

/// A table extracted from a workbook: its addressing handle (`block_id`), label
/// (`worksheet`), passthrough layout metadata, and the portable model. The
/// engine never renders layout — `placement` / `overlay_target_id` are surfaced
/// only so a consumer knows the author's intended arrangement.
#[derive(Debug, Clone)]
pub struct WorkbookTable {
    /// Stable block UUID — the addressing handle.
    pub block_id: String,
    /// Owning worksheet's display name.
    pub worksheet: String,
    /// Author's intended layout (`inFlow` / `sideBySide` / `overlay`).
    pub placement: String,
    /// Overlay target block id, when `placement == "overlay"`.
    pub overlay_target_id: Option<String>,
    /// The portable table model.
    pub table: TableModel,
}

impl Workbook {
    /// Every table block across all worksheets, in document order.
    pub fn tables(&self) -> Vec<WorkbookTable> {
        self.worksheets
            .iter()
            .flat_map(|ws| {
                ws.blocks.iter().filter_map(move |b| match &b.content {
                    BlockContentRaw::Table { table } => Some(WorkbookTable {
                        block_id: b.id.clone(),
                        worksheet: ws.name.clone(),
                        placement: b.placement.clone(),
                        overlay_target_id: b.overlay_target_id.clone(),
                        table: table.clone(),
                    }),
                    BlockContentRaw::Other => None,
                })
            })
            .collect()
    }

    /// Select an overlay layout by `name`, or the sole layout when `name` is
    /// `None`. `Err` carries a user-facing message listing available names
    /// (surfaced verbatim by the CLI's `--layout` handling).
    pub fn overlay_layout(&self, name: Option<&str>) -> Result<&OverlayLayout, String> {
        let names = || {
            self.overlay_layouts
                .iter()
                .map(|l| l.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match name {
            Some(n) => self
                .overlay_layouts
                .iter()
                .find(|l| l.name == n)
                .ok_or_else(|| format!("no overlay layout named '{n}'; available: {}", names())),
            None => match self.overlay_layouts.len() {
                0 => Err("workbook has no overlay layouts".to_string()),
                1 => Ok(&self.overlay_layouts[0]),
                _ => Err(format!(
                    "workbook has multiple overlay layouts, pass --layout; available: {}",
                    names()
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_extracts_table_blocks_and_skips_charts() {
        // Arrange — one worksheet, a chart block then a table block.
        let json = r#"{
            "workbook_id": "wb-1",
            "name": "WB",
            "worksheets": [{
                "name": "Lap Times",
                "blocks": [
                    { "id": "blk-chart", "placement": "inFlow",
                      "content": { "kind": "chart", "slot": {} } },
                    { "id": "blk-table", "placement": "inFlow",
                      "content": { "kind": "table", "table": {
                        "columns": [{"id":"c0","name":"fmax","template":"max([Fork])"}],
                        "rows": [{"id":"r0"}],
                        "cells": [[{}]]
                      } } }
                ]
            }]
        }"#;

        // Act
        let wb: Workbook = serde_json::from_str(json).unwrap();
        let tables = wb.tables();

        // Assert — only the table block surfaces, with its handle + label.
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].block_id, "blk-table");
        assert_eq!(tables[0].worksheet, "Lap Times");
        assert_eq!(tables[0].placement, "inFlow");
        assert_eq!(tables[0].table.columns.len(), 1);
        assert_eq!(tables[0].table.columns[0].name.as_deref(), Some("fmax"));
    }

    #[test]
    fn tables_empty_when_no_worksheets() {
        // Act — a workbook with no worksheets field.
        let wb: Workbook = serde_json::from_str(r#"{ "workbook_id":"w", "name":"n" }"#).unwrap();

        // Assert
        assert!(wb.tables().is_empty());
    }

    #[test]
    fn workbook_v2_with_overlay_layouts_parses_layout_list() {
        // Arrange
        let json = r#"{
          "workbook_id": "w1", "name": "wb", "workbook_version": 2,
          "overlay_layouts": [
            { "id": "L1", "name": "A", "canvas": "1920x1080",
              "elements": [ { "type": "track_map", "rect": [0.8, 0.0, 0.2, 0.3] } ] }
          ]
        }"#;

        // Act
        let wb = crate::workbook::read::parse_workbook(json.as_bytes()).unwrap();

        // Assert
        assert_eq!(wb.overlay_layouts.len(), 1);
        assert_eq!(wb.overlay_layouts[0].name, "A");
    }

    #[test]
    fn workbook_v1_without_field_has_empty_layouts_and_still_parses() {
        // Arrange
        let json = r#"{ "workbook_id": "w1", "name": "wb", "workbook_version": 1 }"#;

        // Act
        let wb = crate::workbook::read::parse_workbook(json.as_bytes()).unwrap();

        // Assert
        assert!(wb.overlay_layouts.is_empty());
    }

    #[test]
    fn overlay_layout_none_with_two_layouts_errs_listing_names() {
        // Arrange
        let json = r#"{ "workbook_id": "w1", "name": "wb", "workbook_version": 2,
          "overlay_layouts": [
            { "id": "L1", "name": "A", "canvas": "1920x1080", "elements": [] },
            { "id": "L2", "name": "B", "canvas": "1920x1080", "elements": [] } ] }"#;
        let wb = crate::workbook::read::parse_workbook(json.as_bytes()).unwrap();

        // Act
        let err = wb.overlay_layout(None).unwrap_err();
        let ok = wb.overlay_layout(Some("B")).unwrap();
        let missing = wb.overlay_layout(Some("C")).unwrap_err();

        // Assert
        assert!(err.contains("A") && err.contains("B"));
        assert_eq!(ok.name, "B");
        assert!(missing.contains("'C'") && missing.contains("A, B"));
    }
}
