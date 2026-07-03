//! The engine's minimal notion of a math-channel definition: a registry name
//! and an expression. The output sample rate is derived at eval time from the
//! referenced channels (`crate::math::evaluate` takes no rate argument), so the
//! workbook's declared `sample_rate_hz` is not modeled here; nor are the display
//! fields (`quantity`, `units`, `decimal_places`, `color`). serde ignores those
//! unknown JSON fields, so a real `.idl0wb` `math_channels` entry deserializes
//! cleanly into this struct.

/// A math-channel definition the engine can evaluate.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct MathChannelDef {
    /// Registry name, e.g. `ForkVelocity`.
    pub name: String,
    /// Expression text, e.g. `differentiate([ForkTravel])`.
    pub expression: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_name_and_expression_ignoring_display_fields() {
        // Arrange — a workbook math_channels entry with display metadata.
        let json = r##"{
            "name": "ForkVelocity",
            "expression": "differentiate([ForkTravel])",
            "quantity": "velocity",
            "units": "m/s",
            "sample_rate_hz": 0.0,
            "decimal_places": 2,
            "color": "#FF5722"
        }"##;

        // Act
        let def: MathChannelDef = serde_json::from_str(json).unwrap();

        // Assert — only name + expression are modeled; the rest are dropped.
        assert_eq!(def.name, "ForkVelocity");
        assert_eq!(def.expression, "differentiate([ForkTravel])");
    }
}
