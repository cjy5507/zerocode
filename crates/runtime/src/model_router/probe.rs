//! Routing probe: a model-verbalized `{complexity, risk, confidence}`
//! self-assessment fused over the deterministic keyword classifier.
//!
//! The jacobian-lens principle behind this module: a cheap early readout of
//! what a model is "prepared to say" predicts the expensive final outcome —
//! here, a ~200-token Fast-tier classification call predicts whether a task
//! deserves a stronger tier, before any tier commits. This module is the pure
//! core only (parsing, fusion arithmetic, prompt text); the provider call
//! itself lives in the tools crate next to the other blocking client users.
//!
//! Fusion is deliberately bounded: the deterministic classifier always runs
//! and its verdict is the anchor. A probe may move complexity at most one
//! band, may only *raise* risk, and is discarded entirely below its
//! confidence gate — a hallucinated probe can therefore never swing a
//! trivial task to the Deep tier (or a repo-wide migration to a Fast model)
//! on its own.

use super::policy::{RouteConfidence, RouteTaskComplexity, RouteTaskRisk};

/// A parsed probe self-assessment. `confidence` is the probe's own stated
/// confidence in its verdict, not the router's signal-weight confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeAssessment {
    pub complexity: RouteTaskComplexity,
    pub risk: RouteTaskRisk,
    pub confidence: RouteConfidence,
}

/// What fusion did with a probe, for audit notes and outcome attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeFusionEffect {
    /// Probe agreed with (or was ignored in favor of) the deterministic verdict.
    Unchanged,
    /// Probe moved complexity up one band (or supplied it for `Unknown`).
    RaisedComplexity,
    /// Probe moved complexity down one band.
    LoweredComplexity,
}

/// Fused verdict plus the effect that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeFusion {
    pub complexity: RouteTaskComplexity,
    pub risk: RouteTaskRisk,
    pub effect: ProbeFusionEffect,
}

const fn complexity_rank(complexity: RouteTaskComplexity) -> Option<i8> {
    match complexity {
        RouteTaskComplexity::Trivial => Some(0),
        RouteTaskComplexity::Small => Some(1),
        RouteTaskComplexity::Medium => Some(2),
        RouteTaskComplexity::Large => Some(3),
        RouteTaskComplexity::Unknown => None,
    }
}

const fn complexity_from_rank(rank: i8) -> RouteTaskComplexity {
    match rank {
        i8::MIN..=0 => RouteTaskComplexity::Trivial,
        1 => RouteTaskComplexity::Small,
        2 => RouteTaskComplexity::Medium,
        _ => RouteTaskComplexity::Large,
    }
}

const fn risk_rank(risk: RouteTaskRisk) -> Option<i8> {
    match risk {
        RouteTaskRisk::Low => Some(0),
        RouteTaskRisk::Medium => Some(1),
        RouteTaskRisk::High => Some(2),
        RouteTaskRisk::Critical => Some(3),
        RouteTaskRisk::Unknown => None,
    }
}

/// Fuse a probe over the deterministic verdict.
///
/// Complexity rules, in order:
/// - probe `Low` confidence, or probe `Unknown` complexity: ignored.
/// - deterministic `Unknown`: probe supplies its value outright (there was
///   nothing to anchor to).
/// - probe `High` confidence: move one band toward the probe's value.
/// - probe `Medium` confidence: move one band toward the probe's value only
///   when that direction is *up* — an under-modeled task costs quality, an
///   over-modeled one only costs tokens, so the downgrade bar is higher.
///
/// Risk: a probe may only raise risk (any confidence above the gate); the
/// deterministic keyword tables stay the floor. Risk gates hard safety
/// behavior downstream (exploration gates, diversity), so it is never
/// lowered on a model's say-so.
#[must_use]
pub fn fuse_probe_assessment(
    deterministic_complexity: RouteTaskComplexity,
    deterministic_risk: RouteTaskRisk,
    probe: &ProbeAssessment,
) -> ProbeFusion {
    let mut fusion = ProbeFusion {
        complexity: deterministic_complexity,
        risk: deterministic_risk,
        effect: ProbeFusionEffect::Unchanged,
    };
    if !matches!(probe.confidence, RouteConfidence::Low) {
        if let Some(probe_risk) = risk_rank(probe.risk) {
            if risk_rank(deterministic_risk).is_none_or(|current| probe_risk > current) {
                fusion.risk = probe.risk;
            }
        }
        fusion = fuse_complexity(deterministic_complexity, *probe, fusion);
    }
    fusion
}

fn fuse_complexity(
    deterministic: RouteTaskComplexity,
    probe: ProbeAssessment,
    mut fusion: ProbeFusion,
) -> ProbeFusion {
    let Some(probe_rank) = complexity_rank(probe.complexity) else {
        return fusion;
    };
    let Some(current) = complexity_rank(deterministic) else {
        fusion.complexity = probe.complexity;
        fusion.effect = ProbeFusionEffect::RaisedComplexity;
        return fusion;
    };
    let step = (probe_rank - current).signum();
    let allowed = match probe.confidence {
        RouteConfidence::High => step != 0,
        RouteConfidence::Medium => step > 0,
        RouteConfidence::Low => false,
    };
    if allowed {
        fusion.complexity = complexity_from_rank(current + step);
        fusion.effect = if step > 0 {
            ProbeFusionEffect::RaisedComplexity
        } else {
            ProbeFusionEffect::LoweredComplexity
        };
    }
    fusion
}

/// The probe's instruction text. Kept terse on purpose: the probe runs on a
/// Fast-tier model with a hard output budget, and the JSON contract below is
/// the entire interface — [`parse_probe_response`] rejects anything else.
#[must_use]
pub fn probe_prompt(description: &str, prompt: &str) -> String {
    // Truncate on a char boundary so a long CJK brief cannot split a code
    // point; the probe only needs the head of the task text to band it.
    const PROBE_INPUT_CHAR_CAP: usize = 2_000;
    let mut task = format!("{description}\n{prompt}");
    if task.chars().count() > PROBE_INPUT_CHAR_CAP {
        task = task.chars().take(PROBE_INPUT_CHAR_CAP).collect();
    }
    format!(
        "You are a routing classifier. Assess the software task below and \
         reply with ONLY this JSON object, no prose:\n\
         {{\"complexity\":\"trivial|small|medium|large\",\
         \"risk\":\"low|medium|high|critical\",\
         \"confidence\":\"low|medium|high\"}}\n\
         complexity = how much reasoning/context the task needs end to end \
         (large: repo-wide, multi-subsystem, or architecturally hard; \
         trivial: a label/typo-class edit). risk = blast radius of a wrong \
         edit (credentials, deletion, security = high+). confidence = your \
         confidence in this assessment.\n\nTASK:\n{task}"
    )
}

/// Parse the probe's reply. Strict by design: the first `{{…}}` JSON object
/// found is parsed, unknown enum tokens reject the whole probe (fail closed
/// to the deterministic verdict rather than guess what a malformed probe
/// meant).
#[must_use]
pub fn parse_probe_response(raw: &str) -> Option<ProbeAssessment> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(&raw[start..=end]).ok()?;
    let field = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(|token| token.trim().to_ascii_lowercase())
    };
    let complexity = match field("complexity")?.as_str() {
        "trivial" => RouteTaskComplexity::Trivial,
        "small" => RouteTaskComplexity::Small,
        "medium" => RouteTaskComplexity::Medium,
        "large" => RouteTaskComplexity::Large,
        _ => return None,
    };
    let risk = match field("risk")?.as_str() {
        "low" => RouteTaskRisk::Low,
        "medium" => RouteTaskRisk::Medium,
        "high" => RouteTaskRisk::High,
        "critical" => RouteTaskRisk::Critical,
        _ => return None,
    };
    let confidence = match field("confidence")?.as_str() {
        "low" => RouteConfidence::Low,
        "medium" => RouteConfidence::Medium,
        "high" => RouteConfidence::High,
        _ => return None,
    };
    Some(ProbeAssessment { complexity, risk, confidence })
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn probe(
        complexity: RouteTaskComplexity,
        risk: RouteTaskRisk,
        confidence: RouteConfidence,
    ) -> ProbeAssessment {
        ProbeAssessment { complexity, risk, confidence }
    }

    #[test]
    fn high_confidence_probe_moves_one_band_not_more() {
        let fusion = fuse_probe_assessment(
            RouteTaskComplexity::Small,
            RouteTaskRisk::Low,
            &probe(RouteTaskComplexity::Large, RouteTaskRisk::Low, RouteConfidence::High),
        );
        assert_eq!(fusion.complexity, RouteTaskComplexity::Medium);
        assert_eq!(fusion.effect, ProbeFusionEffect::RaisedComplexity);
    }

    #[test]
    fn medium_confidence_upgrades_but_never_downgrades() {
        let up = fuse_probe_assessment(
            RouteTaskComplexity::Small,
            RouteTaskRisk::Low,
            &probe(RouteTaskComplexity::Medium, RouteTaskRisk::Low, RouteConfidence::Medium),
        );
        assert_eq!(up.complexity, RouteTaskComplexity::Medium);
        let down = fuse_probe_assessment(
            RouteTaskComplexity::Large,
            RouteTaskRisk::Low,
            &probe(RouteTaskComplexity::Small, RouteTaskRisk::Low, RouteConfidence::Medium),
        );
        assert_eq!(down.complexity, RouteTaskComplexity::Large);
        assert_eq!(down.effect, ProbeFusionEffect::Unchanged);
    }

    #[test]
    fn low_confidence_probe_is_ignored_entirely() {
        let fusion = fuse_probe_assessment(
            RouteTaskComplexity::Small,
            RouteTaskRisk::Low,
            &probe(RouteTaskComplexity::Large, RouteTaskRisk::Critical, RouteConfidence::Low),
        );
        assert_eq!(fusion.complexity, RouteTaskComplexity::Small);
        assert_eq!(fusion.risk, RouteTaskRisk::Low);
        assert_eq!(fusion.effect, ProbeFusionEffect::Unchanged);
    }

    #[test]
    fn unknown_deterministic_complexity_takes_probe_value() {
        let fusion = fuse_probe_assessment(
            RouteTaskComplexity::Unknown,
            RouteTaskRisk::Unknown,
            &probe(RouteTaskComplexity::Large, RouteTaskRisk::Medium, RouteConfidence::High),
        );
        assert_eq!(fusion.complexity, RouteTaskComplexity::Large);
        assert_eq!(fusion.risk, RouteTaskRisk::Medium);
        assert_eq!(fusion.effect, ProbeFusionEffect::RaisedComplexity);
    }

    #[test]
    fn risk_only_ever_rises() {
        let fusion = fuse_probe_assessment(
            RouteTaskComplexity::Medium,
            RouteTaskRisk::High,
            &probe(RouteTaskComplexity::Medium, RouteTaskRisk::Low, RouteConfidence::High),
        );
        assert_eq!(fusion.risk, RouteTaskRisk::High);
        let raised = fuse_probe_assessment(
            RouteTaskComplexity::Medium,
            RouteTaskRisk::Low,
            &probe(RouteTaskComplexity::Medium, RouteTaskRisk::High, RouteConfidence::Medium),
        );
        assert_eq!(raised.risk, RouteTaskRisk::High);
    }

    #[test]
    fn parse_accepts_fenced_json_and_rejects_unknown_tokens() {
        let parsed = parse_probe_response(
            "```json\n{\"complexity\":\"large\",\"risk\":\"medium\",\"confidence\":\"high\"}\n```",
        )
        .expect("fenced JSON parses");
        assert_eq!(parsed.complexity, RouteTaskComplexity::Large);
        assert_eq!(parsed.risk, RouteTaskRisk::Medium);
        assert_eq!(parsed.confidence, RouteConfidence::High);
        assert!(parse_probe_response(
            "{\"complexity\":\"gigantic\",\"risk\":\"medium\",\"confidence\":\"high\"}"
        )
        .is_none());
        assert!(parse_probe_response("no json here").is_none());
    }

    #[test]
    fn probe_prompt_caps_input_on_char_boundary() {
        let long_task = "레포".repeat(3_000);
        let prompt = probe_prompt("설명", &long_task);
        assert!(prompt.chars().count() < 2_600);
        assert!(prompt.contains("routing classifier"));
    }
}
