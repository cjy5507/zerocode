//! Paired-arm ablation comparison — the cross-run judgment `fairness.rs`
//! deliberately defers.
//!
//! A single run records `fairness_level: "unknown"` on purpose: no one run can
//! say whether it is comparable to another. This module is the missing half.
//! Given the two contracts of an ablation pair it decides whether they differ
//! in EXACTLY the held-out feature set, and only then reports the outcome
//! delta.
//!
//! The order matters: conditions are checked BEFORE any number is computed, and
//! a rejected pair yields no delta at all rather than a delta with a caveat
//! attached. A caveat is something a reader can skip; a missing number is not.
//! This is the same reason `AblationSet::parse` refuses an unknown feature key
//! instead of ignoring it — an experiment that quietly measured the wrong thing
//! is worse than one that refused to report.

use serde::Serialize;

use crate::fairness::FairnessContract;
use crate::runner::RunResult;

/// One side of an ablation pair: what was run, and how it went.
#[derive(Debug, Clone, Copy)]
pub struct Arm<'a> {
    pub contract: &'a FairnessContract,
    pub result: &'a RunResult,
    /// Which repetition of this cell the run was (0-based).
    ///
    /// Carried for traceability into the emitted delta and deliberately NOT
    /// compared between arms: the k-th run of one arm has no privileged
    /// correspondence to the k-th run of the other. Pairing by index is only a
    /// convention that keeps the two arms' sample counts equal, and enforcing
    /// it would dress that convention up as a paired design it is not.
    pub rep: usize,
}

impl<'a> Arm<'a> {
    #[must_use]
    pub fn new(contract: &'a FairnessContract, result: &'a RunResult, rep: usize) -> Self {
        Self {
            contract,
            result,
            rep,
        }
    }
}

/// Why a pair cannot be read as a controlled comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PairRejection {
    /// A run whose own recorded conditions were already untrustworthy. Nothing
    /// downstream can repair that, so the pair is refused before any
    /// cross-arm check runs.
    ArmInvalid {
        arm: &'static str,
        reason: String,
    },
    /// A condition that had to be held constant differed between the arms, so
    /// any measured difference has at least two candidate causes.
    ConditionDiffers {
        condition: &'static str,
        full: String,
        ablated: String,
    },
    /// The baseline arm held features out. Then it is not a baseline, and the
    /// "delta" would be between two treatments.
    BaselineIsAblated { held_out: Vec<String> },
    /// Neither arm held anything out: there is no treatment to attribute a
    /// difference to. Two identical configurations measure run-to-run noise,
    /// which is a useful thing to measure and NOT an ablation.
    NoHoldout,
    /// A contract names a holdout the runtime has no such feature for. The
    /// `ZO_ABLATE` that produced this run therefore suppressed nothing, so the
    /// "ablated" arm is a second full-feature run wearing an experiment's
    /// label — the exact failure `AblationSet::parse` refuses at spec time,
    /// caught again here because a contract is data and can arrive from
    /// anywhere.
    UnknownHoldout { arm: &'static str, token: String },
    /// A condition that must be held constant is blank on BOTH arms. Equality
    /// then proves nothing: two runs that never recorded their runner version
    /// compare equal whether they came from the same build or not.
    ConditionUndeclared { condition: &'static str },
    /// The result handed in does not belong to the contract beside it. Pairing
    /// one arm's contract with another arm's result would attribute a
    /// difference the arms never produced.
    ResultDoesNotMatchContract {
        arm: &'static str,
        field: &'static str,
        contract: String,
        result: String,
    },
}

impl std::fmt::Display for PairRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArmInvalid { arm, reason } => {
                write!(formatter, "{arm} arm is not a valid run: {reason}")
            }
            Self::ConditionDiffers {
                condition,
                full,
                ablated,
            } => write!(
                formatter,
                "{condition} differs between arms (full: {full}, ablated: {ablated}) — \
                 the arms must differ only in the holdout"
            ),
            Self::BaselineIsAblated { held_out } => write!(
                formatter,
                "the baseline arm itself holds out {} — it is not a baseline",
                held_out.join(", ")
            ),
            Self::NoHoldout => formatter.write_str(
                "neither arm holds any feature out — this pair measures run-to-run \
                 variance, not a feature's contribution",
            ),
            Self::UnknownHoldout { arm, token } => write!(
                formatter,
                "{arm} arm records a holdout '{token}' that names no harness feature — \
                 whatever ran, it did not suppress that"
            ),
            Self::ConditionUndeclared { condition } => write!(
                formatter,
                "{condition} is blank on both arms, so holding it constant proves nothing"
            ),
            Self::ResultDoesNotMatchContract {
                arm,
                field,
                contract,
                result,
            } => write!(
                formatter,
                "{arm} arm's result does not belong to its contract ({field}: contract says \
                 {contract}, result says {result})"
            ),
        }
    }
}

/// What ONE pair of runs was observed to do — deliberately named for the
/// observation, not for a cause.
///
/// An agent run is stochastic, so a single pass/fail flip is one sample, not a
/// demonstration that the feature carried or hurt the task. Naming a variant
/// `FeatureCarried` invited exactly that over-read: a reader (or a report
/// generator) would lift the word straight into a causal claim built on n=1.
/// Attribution belongs to an aggregate over repeats; this type only records
/// what happened, and [`Self::is_correctness_informative`] says whether it can
/// contribute to such an aggregate at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeShift {
    /// Passed only in the full-feature arm. Evidence FOR the feature on this
    /// sample — one observation, not a verdict.
    PassedOnlyWithFeature,
    /// Passed only in the ablated arm. Evidence AGAINST the feature on this
    /// sample, and the direction a harness that only looked for improvements
    /// would never surface.
    PassedOnlyWithoutFeature,
    /// Both arms passed: this task did not separate the arms on correctness.
    /// Cost differences are still meaningful.
    BothPassed,
    /// Both arms failed: the task is out of reach either way, so it carries no
    /// signal about this feature and must not be averaged in as if it did.
    BothFailed,
}

impl OutcomeShift {
    /// Whether this pair can contribute to a CORRECTNESS aggregate at all.
    /// `BothFailed` cannot: the task never got far enough for the feature to
    /// matter, so counting it would dilute the rate with a non-observation.
    ///
    /// Named for correctness specifically because a `BothFailed` pair can
    /// still carry usable COST evidence — both arms ran and billed — and a
    /// bare `is_informative` would invite discarding that too.
    #[must_use]
    pub const fn is_correctness_informative(self) -> bool {
        !matches!(self, Self::BothFailed)
    }

    /// Whether the two arms disagreed. A single disagreement is a sample worth
    /// recording and NOT a causal finding — see the type's own doc.
    #[must_use]
    pub const fn arms_disagreed(self) -> bool {
        matches!(
            self,
            Self::PassedOnlyWithFeature | Self::PassedOnlyWithoutFeature
        )
    }

    #[must_use]
    const fn of(full_pass: bool, ablated_pass: bool) -> Self {
        match (full_pass, ablated_pass) {
            (true, false) => Self::PassedOnlyWithFeature,
            (false, true) => Self::PassedOnlyWithoutFeature,
            (true, true) => Self::BothPassed,
            (false, false) => Self::BothFailed,
        }
    }
}

/// One controlled comparison: what was held out, and what it cost or bought.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AblationDelta {
    /// The feature keys the ablated arm held out.
    pub held_out: Vec<String>,
    /// The runner under test — the same on both arms, since that is what the
    /// pair holds constant.
    pub runner_kind: String,
    /// The two arms' suite names. These necessarily DIFFER (each arm owns its
    /// own output directory and its own `<NAME>_ABLATE`), so they are recorded
    /// rather than compared — a reader needs to know which two directories
    /// produced the number.
    pub runner_full: String,
    pub runner_ablated: String,
    pub fixture_id: String,
    pub lane: String,
    /// Which repetition of the cell these two runs were. See [`Arm::rep`].
    pub rep: usize,
    pub full_pass: bool,
    pub ablated_pass: bool,
    pub outcome: OutcomeShift,
    pub wall_seconds_full: u64,
    pub wall_seconds_ablated: u64,
    /// Billed totals, when both arms reported them.
    pub token_total_full: Option<u64>,
    pub token_total_ablated: Option<u64>,
    /// False when either arm's token accounting was incomplete — a bare
    /// `input` headline hides cached context, so a delta computed from it
    /// would understate the feature's real cost. A reader must be able to tell
    /// "the feature saved nothing" from "we could not see what it saved".
    pub token_accounting_complete: bool,
}

impl AblationDelta {
    /// Tokens the feature COST: positive when running the feature billed more
    /// than holding it out.
    ///
    /// `None` unless BOTH arms reported totals AND both accounted completely.
    /// Incomplete accounting means the cache counters were absent and `total`
    /// summed only the classes present, so the difference of two such totals
    /// is a lower bound of unknown tightness — returning it as "the cost"
    /// would publish that lower bound as a measurement. The raw per-arm totals
    /// stay on the struct for a caller that wants to say so explicitly.
    #[must_use]
    pub fn token_cost_of_feature(&self) -> Option<i64> {
        if !self.token_accounting_complete {
            return None;
        }
        let full = i64::try_from(self.token_total_full?).ok()?;
        let ablated = i64::try_from(self.token_total_ablated?).ok()?;
        Some(full - ablated)
    }

    /// Wall seconds the feature cost: positive when running it was slower.
    #[must_use]
    pub fn wall_cost_of_feature(&self) -> i64 {
        let full = i64::try_from(self.wall_seconds_full).unwrap_or(i64::MAX);
        let ablated = i64::try_from(self.wall_seconds_ablated).unwrap_or(i64::MAX);
        full - ablated
    }
}

/// One condition that must match for a difference to be attributable to the
/// holdout.
struct HeldConstant {
    name: &'static str,
    read: fn(&FairnessContract) -> String,
    /// When true, a value blank on BOTH arms is itself a rejection.
    ///
    /// Equality between two blanks proves nothing: two runs that never
    /// recorded their runner version compare equal whether or not they came
    /// from the same build. `judge` files exactly these omissions as `partial`
    /// rather than `invalid` (they are only a *cross-runner* gap there), so the
    /// pair check is the only thing standing between an undeclared condition
    /// and a confident delta.
    must_be_declared: bool,
}

/// Every condition held constant. Adding one is a single entry, and the entry
/// carries its own reported name, so a new condition cannot be compared without
/// being nameable in the rejection.
const HELD_CONSTANT: &[HeldConstant] = &[
    // The runner KIND, never the runner NAME. The two arms of an ablation are
    // two suite runners over one binary, so their names are guaranteed to
    // differ — each owns an output directory and a `<NAME>_ABLATE` — while the
    // thing under test is identical. Comparing the name here would reject every
    // pair the suite is able to produce, which is how a measurement apparatus
    // ends up unable to measure anything; comparing the kind rejects the case
    // that actually matters (a `zo` arm paired against a `claude` arm).
    HeldConstant { name: "runner_kind", read: |c| c.runner_kind.clone(), must_be_declared: true },
    HeldConstant { name: "lane", read: |c| c.lane.clone(), must_be_declared: true },
    HeldConstant { name: "fixture_id", read: |c| c.fixture_id.clone(), must_be_declared: true },
    // Either identifier alone pins the fixture, and the suite records only the
    // tree hash — so neither may demand a value on its own. A pair with NO
    // pinned base is already `invalid` by `judge`, which the arm check catches.
    HeldConstant {
        name: "fixture_tree_hash",
        read: |c| c.fixture_tree_hash.clone(),
        must_be_declared: false,
    },
    HeldConstant {
        name: "fixture_commit",
        read: |c| c.fixture_commit.clone(),
        must_be_declared: false,
    },
    HeldConstant { name: "prompt_sha256", read: |c| c.prompt_sha256.clone(), must_be_declared: true },
    HeldConstant {
        name: "test_command_sha256",
        read: |c| c.test_command_sha256.clone(),
        must_be_declared: true,
    },
    HeldConstant {
        name: "intended_path_set_sha256",
        read: |c| c.intended_path_set_sha256.clone(),
        must_be_declared: true,
    },
    // Exact, not the normalized family. Normalization exists for CROSS-runner
    // comparison, where `claude-opus-5` and `claude-opus-4-8` both reading
    // "opus" is the point. An ablation pair is same-runner, so two different
    // Opus releases are a confound the coarser check would wave through.
    HeldConstant { name: "declared_model", read: |c| c.declared_model.clone(), must_be_declared: true },
    HeldConstant {
        name: "declared_effort",
        read: |c| c.declared_effort.clone(),
        must_be_declared: true,
    },
    HeldConstant {
        name: "permission_mode",
        read: |c| c.permission_mode.clone(),
        must_be_declared: true,
    },
    HeldConstant {
        name: "timeout_seconds",
        read: |c| c.timeout_seconds.to_string(),
        must_be_declared: false,
    },
    // Two arms run from different builds measure the build as much as the
    // holdout. Arms are usually run minutes apart, which is exactly long
    // enough for a rebuild to land between them. A version STRING is a weak
    // proxy — two local rebuilds both print `zo 0.1.13` — so this rejects the
    // detectable case and does not pretend to prove binary identity; that
    // needs an executable digest the contract does not yet record.
    HeldConstant { name: "runner_version", read: |c| c.runner_version.clone(), must_be_declared: true },
    HeldConstant {
        name: "harness_version",
        read: |c| c.harness_version.clone(),
        must_be_declared: true,
    },
    HeldConstant {
        name: "benchmark_suite_version",
        read: |c| c.benchmark_suite_version.clone(),
        must_be_declared: true,
    },
    // The schema the other fields are read under. Two contracts written by
    // different schema versions may not mean the same thing field for field.
    HeldConstant {
        name: "fairness_contract_version",
        read: |c| c.fairness_contract_version.clone(),
        must_be_declared: true,
    },
];

/// Every reason this pair is not a controlled comparison. Empty means it is
/// one.
///
/// All reasons are collected rather than short-circuiting on the first: an
/// operator fixing a misconfigured sweep wants the whole list, not one
/// round-trip per divergent field. That includes continuing past an invalid
/// arm — an invalid run is often invalid *and* misconfigured, and reporting
/// only the first defect would hide the second until the next round.
///
/// Note this checks only the two CONTRACTS. Binding each contract to the
/// result recorded beside it is a separate internal check that [`measure`]
/// also runs.
#[must_use]
pub fn reject_reasons(full: &FairnessContract, ablated: &FairnessContract) -> Vec<PairRejection> {
    let mut rejections = Vec::new();

    for (arm, contract) in [("full", full), ("ablated", ablated)] {
        if contract.status == "invalid" {
            rejections.push(PairRejection::ArmInvalid {
                arm,
                reason: contract
                    .status_reason
                    .clone()
                    .unwrap_or_else(|| "recorded as invalid".to_string()),
            });
        }
    }

    // A contract is data and can arrive from any producer, so the keys are
    // re-checked here even though the runner refuses an unknown holdout before
    // spawning. A holdout that names nothing suppressed nothing.
    for (arm, contract) in [("full", full), ("ablated", ablated)] {
        for token in &contract.ablation {
            if telemetry::HarnessFeature::from_key(token).is_none() {
                rejections.push(PairRejection::UnknownHoldout {
                    arm,
                    token: token.clone(),
                });
            }
        }
    }

    if !full.ablation.is_empty() {
        rejections.push(PairRejection::BaselineIsAblated {
            held_out: full.ablation.clone(),
        });
    } else if ablated.ablation.is_empty() {
        rejections.push(PairRejection::NoHoldout);
    }

    for condition in HELD_CONSTANT {
        let (left, right) = ((condition.read)(full), (condition.read)(ablated));
        if left != right {
            rejections.push(PairRejection::ConditionDiffers {
                condition: condition.name,
                full: left,
                ablated: right,
            });
        } else if condition.must_be_declared && left.trim().is_empty() {
            rejections.push(PairRejection::ConditionUndeclared {
                condition: condition.name,
            });
        }
    }

    rejections
}

/// Reasons this arm's result does not belong to the contract beside it.
///
/// The delta reads outcome from the RESULT and identity from the CONTRACT, so
/// nothing else notices if the two describe different runs. `model` and
/// `effort` are compared through the same normalization the spec applied when
/// the run was launched, which is what makes the contract's declared labels and
/// the result's recorded labels comparable at all.
fn result_binding_rejections(arm: &'static str, side: Arm<'_>) -> Vec<PairRejection> {
    // `runner` IS compared here, unlike in `HELD_CONSTANT`: across the arms the
    // name must differ, but within one arm the contract and the result describe
    // the same run and must agree on it. This is the check that catches a
    // result file read out of the wrong arm's directory.
    let expected: [(&'static str, String, String); 4] = [
        (
            "runner",
            side.contract.runner.clone(),
            side.result.runner.clone(),
        ),
        ("lane", side.contract.lane.clone(), side.result.lane.clone()),
        (
            "model",
            crate::runner::normalize_model_label(&side.contract.declared_model),
            side.result.model.clone(),
        ),
        (
            "effort",
            crate::runner::normalize_effort_label(&side.contract.declared_effort),
            side.result.effort.clone(),
        ),
    ];
    expected
        .into_iter()
        .filter(|(_, contract, result)| contract != result)
        .map(
            |(field, contract, result)| PairRejection::ResultDoesNotMatchContract {
                arm,
                field,
                contract,
                result,
            },
        )
        .collect()
}

/// Measure one ablation pair, or refuse with every reason it is not a
/// controlled comparison.
///
/// # Errors
///
/// Returns the non-empty rejection list when the arms are not comparable.
pub fn measure(full: Arm<'_>, ablated: Arm<'_>) -> Result<AblationDelta, Vec<PairRejection>> {
    let mut rejections = reject_reasons(full.contract, ablated.contract);
    rejections.extend(result_binding_rejections("full", full));
    rejections.extend(result_binding_rejections("ablated", ablated));
    if !rejections.is_empty() {
        return Err(rejections);
    }

    let token_total_full = full.result.tokens.as_ref().map(|tokens| tokens.total);
    let token_total_ablated = ablated.result.tokens.as_ref().map(|tokens| tokens.total);
    let token_accounting_complete = full
        .result
        .tokens
        .as_ref()
        .is_some_and(|tokens| tokens.complete)
        && ablated
            .result
            .tokens
            .as_ref()
            .is_some_and(|tokens| tokens.complete);

    Ok(AblationDelta {
        held_out: ablated.contract.ablation.clone(),
        runner_kind: full.contract.runner_kind.clone(),
        runner_full: full.contract.runner.clone(),
        runner_ablated: ablated.contract.runner.clone(),
        fixture_id: full.contract.fixture_id.clone(),
        lane: full.contract.lane.clone(),
        rep: ablated.rep,
        full_pass: full.result.pass,
        ablated_pass: ablated.result.pass,
        outcome: OutcomeShift::of(full.result.pass, ablated.result.pass),
        wall_seconds_full: full.result.wall_seconds,
        wall_seconds_ablated: ablated.result.wall_seconds,
        token_total_full,
        token_total_ablated,
        token_accounting_complete,
    })
}

/// A cell that declared a treatment arm but could not be turned into a pair.
///
/// Reported rather than dropped, on the same principle as the rejections: an
/// experiment arm that quietly produced no comparison looks, in the output,
/// exactly like an experiment that found no difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnpairedArm {
    pub runner_kind: String,
    pub runner_ablated: String,
    pub lane: String,
    pub fixture_id: String,
    pub rep: usize,
    pub reason: String,
}

/// A pair that was formed but refused, with every reason it was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RejectedPair {
    pub runner_full: String,
    pub runner_ablated: String,
    pub lane: String,
    pub fixture_id: String,
    pub rep: usize,
    pub reasons: Vec<PairRejection>,
    /// The rendered reasons, so the JSON is readable without re-implementing
    /// [`PairRejection`]'s `Display` in whatever reads it.
    pub messages: Vec<String>,
}

/// Every pair a suite run produced, plus everything it could not pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AblationReport {
    /// Per-lane rollups — the only level at which a difference should be read.
    /// A single pair is one sample of a stochastic process; see
    /// [`OutcomeShift`]'s own doc.
    pub aggregates: Vec<AblationAggregate>,
    pub pairs: Vec<AblationDelta>,
    pub rejected: Vec<RejectedPair>,
    pub unpaired: Vec<UnpairedArm>,
    /// The level every `evidence` verdict in this report was stated at.
    /// Serialized because a threshold that lives only in a Rust const makes two
    /// reports written under different constants silently incomparable.
    pub evidence_alpha: f64,
    /// Always `"none"` today. Each rollup is tested independently at
    /// `evidence_alpha`, so across many rollups the chance that at least one
    /// reads `difference_favours_*` by luck is far above alpha — twenty null
    /// rollups give roughly 64%. A consumer that promotes the one significant
    /// lane out of many is over-reading, and this field is what tells it so.
    pub multiple_comparison_adjustment: &'static str,
}

impl Default for AblationReport {
    /// Hand-written rather than derived: `f64::default()` would stamp an
    /// alpha of 0.0 on every report, which is not a level any verdict here was
    /// stated at.
    fn default() -> Self {
        Self {
            aggregates: Vec::new(),
            pairs: Vec::new(),
            rejected: Vec::new(),
            unpaired: Vec::new(),
            evidence_alpha: EVIDENCE_ALPHA,
            multiple_comparison_adjustment: "none",
        }
    }
}

impl AblationReport {
    /// True when nothing about this suite run was an experiment: no runner
    /// declared a holdout, so there is nothing to write.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty() && self.rejected.is_empty() && self.unpaired.is_empty()
    }
}

/// What a whole lane's worth of pairs observed about one holdout.
///
/// Named — again — for the observation. Even here nothing is called a "win": a
/// rollup over a handful of stochastic tasks is evidence whose strength is
/// stated by [`Self::discordant_two_sided_p`], not by the sign of a difference.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AblationAggregate {
    pub runner_kind: String,
    pub held_out: Vec<String>,
    pub lane: String,
    pub pairs: usize,
    /// Pairs that can speak to correctness at all — every pair except the ones
    /// both arms failed. This, not `pairs`, is the denominator of the counts
    /// below.
    pub correctness_informative_pairs: usize,
    pub passed_only_with_feature: usize,
    pub passed_only_without_feature: usize,
    pub both_passed: usize,
    pub both_failed: usize,
    /// Exact two-sided sign test over the DISCORDANT pairs only — the standard
    /// read of a paired binary outcome, and the number that stops a 2-vs-0 flip
    /// from being reported as a result (its p is 0.5).
    ///
    /// `None` when no pair disagreed: with nothing discordant there is no test
    /// to run, and printing `1.0` would suggest one was run and came back null.
    pub discordant_two_sided_p: Option<f64>,
    /// What the pairs above actually support at [`EVIDENCE_ALPHA`]. Computed
    /// beside the counts so an under-powered rollup cannot be read as a result
    /// by skipping the p-value.
    pub evidence: EvidenceVerdict,
    /// How many pairs contributed to the token figures. Both arms must have
    /// reported COMPLETE accounting, so this is usually smaller than `pairs`
    /// and a reader needs it to know the coverage behind the totals.
    pub token_cost_pairs: usize,
    /// Tokens the feature cost across the lane: positive means running it
    /// billed more than holding it out.
    pub token_cost_of_feature_total: Option<i64>,
    /// Per-task median of the same quantity — the figure to quote when one
    /// runaway task dominates the sum.
    pub token_cost_of_feature_median: Option<i64>,
    pub wall_cost_of_feature_total: i64,
}

/// The significance level every rollup's verdict is stated at.
pub const EVIDENCE_ALPHA: f64 = 0.05;

/// What a rollup's evidence actually supports, computed rather than left to
/// the reader.
///
/// The counts and the p-value already sit side by side, and a reader who
/// quotes `passed_only_with_feature: 2, passed_only_without_feature: 0` while
/// skipping the `0.5` beside it has published a result the data does not
/// contain. That is the same shape of defect as a verdict field nothing
/// derives or a judge budget nothing enforces: a guarantee stated in prose
/// that no type holds. This enum holds it — the refusal is computed from the
/// same inputs the counts come from.
///
/// It does NOT make the refusal unskippable. `AblationAggregate` is a plain
/// `Serialize` struct, so a consumer can still read only the counts; what this
/// buys is that the supported reading is stated rather than left to be
/// inferred. Enforcing it needs a renderer that derives directional prose only
/// from this field.
///
/// Every rollup is judged at the same [`EVIDENCE_ALPHA`] with NO
/// multiple-comparison correction, so one significant lane among many is
/// weaker than its verdict word — see [`AblationReport::multiple_comparison_adjustment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum EvidenceVerdict {
    /// The arms never disagreed on any pair.
    ///
    /// Deliberately not [`Self::InsufficientEvidence`]: fifty pairs that all
    /// agreed is a substantive observation about the feature, not a sample-size
    /// problem, and no number of further fixtures from the same population is
    /// promised to produce a flip. `discordant_two_sided_p` is `None` here for
    /// the same reason — there is no test to run.
    NoDiscordantPairs { pairs: usize },
    /// Some pairs disagreed, but too few for ANY split of them to reach
    /// [`EVIDENCE_ALPHA`].
    InsufficientEvidence {
        discordant_pairs: usize,
        /// The floor a discordant count must clear before the test can reject
        /// at all, reached only by an ALL-ONE-DIRECTION split — a 3-3 split at
        /// this count still concludes nothing.
        ///
        /// A necessary condition, not an actionable remedy: what governs how
        /// many fixtures produce this many flips is the discordance RATE, so
        /// this is not "add N more fixtures".
        min_discordant_pairs_all_one_direction: usize,
    },
    /// Enough discordant pairs that the most extreme split would have reached
    /// alpha, and this split did not.
    ///
    /// Weak evidence of absence, never evidence of no effect: near the floor
    /// only a near-sweep is detectable, so a 5-1 split lands here while saying
    /// very little. An equivalence claim needs a test this module does not run.
    NoDifferenceDetected,
    /// The arms differed at alpha, with the full-feature arm passing more.
    DifferenceFavoursFeature,
    /// The arms differed at alpha, with the ablated arm passing more — the
    /// direction a harness that only looked for improvements would never see.
    DifferenceFavoursHoldout,
}

/// How far the search for a discordant-count floor runs. Every positive `f64`
/// alpha is reached well inside this: `p(n, 0)` is `2^(1-n)`, which reaches
/// the smallest subnormal by n ≈ 1075.
const MAX_DISCORDANT_FLOOR_SEARCH: usize = 2048;

/// The smallest number of discordant pairs that could ever reach `alpha`.
///
/// Derived from [`discordant_two_sided_p`] itself rather than from a closed
/// form, so the threshold and the test can never disagree about what the test
/// does. At the default alpha this is 6: a suite whose tasks flip fewer than
/// six times cannot conclude anything, however lopsided the flips are.
///
/// `None` when `alpha` is not a probability a test can be stated at — zero,
/// negative, above one, or NaN — rather than a sentinel that would serialize
/// as an astronomically large "pairs needed".
#[must_use]
pub fn min_discordant_pairs_for(alpha: f64) -> Option<usize> {
    if !(alpha > 0.0 && alpha <= 1.0) {
        return None;
    }
    (1..=MAX_DISCORDANT_FLOOR_SEARCH)
        .find(|n| discordant_two_sided_p(*n, 0).is_some_and(|p| p <= alpha))
}

fn evidence_verdict(with: usize, without: usize, pairs: usize, alpha: f64) -> EvidenceVerdict {
    let discordant_pairs = with + without;
    if discordant_pairs == 0 {
        return EvidenceVerdict::NoDiscordantPairs { pairs };
    }
    // An alpha no test can be stated at leaves every rollup unjudged rather
    // than silently judged at some other level.
    let Some(needed) = min_discordant_pairs_for(alpha) else {
        return EvidenceVerdict::InsufficientEvidence {
            discordant_pairs,
            min_discordant_pairs_all_one_direction: usize::MAX,
        };
    };
    if discordant_pairs < needed {
        return EvidenceVerdict::InsufficientEvidence {
            discordant_pairs,
            min_discordant_pairs_all_one_direction: needed,
        };
    }
    // `discordant_pairs >= needed >= 1`, so the tail is defined and this is
    // `Some`. A computation that somehow failed is refused, never reported as
    // a null result.
    let Some(p) = discordant_two_sided_p(with, without) else {
        return EvidenceVerdict::InsufficientEvidence {
            discordant_pairs,
            min_discordant_pairs_all_one_direction: needed,
        };
    };
    match (p <= alpha, with.cmp(&without)) {
        (true, std::cmp::Ordering::Greater) => EvidenceVerdict::DifferenceFavoursFeature,
        (true, std::cmp::Ordering::Less) => EvidenceVerdict::DifferenceFavoursHoldout,
        _ => EvidenceVerdict::NoDifferenceDetected,
    }
}

/// Exact two-sided sign test on `with`-vs-`without` discordant pairs.
///
/// p = min(1, 2·P[X ≤ min(a,b)]) for X ~ Binomial(a+b, ½). Computed by the
/// term-to-term recurrence rather than from factorials: the binomial
/// coefficients overflow `u64` around n = 68, which a suite with a few hundred
/// pairs would reach, and an overflowed p-value would silently read as
/// significant.
fn discordant_two_sided_p(with: usize, without: usize) -> Option<f64> {
    let n = with.checked_add(without)?;
    if n == 0 {
        return None;
    }
    let n_f = u32::try_from(n).ok().map(f64::from)?;
    let smaller = with.min(without);

    // Accumulated in log2 space and combined by log-sum-exp.
    //
    // The obvious form starts the recurrence at the endpoint term 2^-n. That
    // endpoint underflows to zero for n ≳ 1075, and once it does every later
    // term stays zero and the tail comes back as exactly 0 — so a near-balanced
    // large sample like 1001-vs-999, whose true p is ≈ 0.97, was reported as
    // p = 0 and read as a confident directional difference. Scaling every term
    // against the largest one keeps the sum meaningful no matter how small the
    // individual terms are.
    let mut log2_binomial = 0.0_f64; // log2 C(n, 0)
    let mut max_log2 = -n_f; // log2 of the i = 0 term
    let mut scaled_sum = 1.0_f64; // Σ 2^(log2 term_i − max_log2)
    for i in 1..=smaller {
        let i_f = u32::try_from(i).ok().map(f64::from)?;
        log2_binomial += (n_f - i_f + 1.0).log2() - i_f.log2();
        let log2_term = log2_binomial - n_f;
        if log2_term > max_log2 {
            scaled_sum = scaled_sum * (max_log2 - log2_term).exp2() + 1.0;
            max_log2 = log2_term;
        } else {
            scaled_sum += (log2_term - max_log2).exp2();
        }
    }
    // The two-sided doubling rides in the exponent rather than multiplying the
    // result: `2.0 * exp2(-1075)` is 0.0 because the inner term rounds away
    // below the smallest subnormal, while `exp2(1 - 1075)` is that subnormal
    // exactly — which is the true p for a 1075-0 sweep.
    Some((1.0 + max_log2 + scaled_sum.log2()).exp2().min(1.0))
}

/// Middle value, averaging the two middle values on an even count. Truncating
/// division is fine here: the quantity is a token count, so half a token is not
/// a meaningful distinction.
fn median(mut values: Vec<i64>) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        i64::midpoint(values[middle - 1], values[middle])
    })
}

/// The cell an arm belongs to. Arms pair within a cell and never across one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    runner_kind: String,
    lane: String,
    fixture_id: String,
    rep: usize,
}

/// Pair every ablated arm in a suite run against its full-feature baseline and
/// roll the results up per lane.
///
/// The baseline is identified by CONFIGURATION, not by name: within a cell, the
/// arm whose contract holds nothing out is the control. That is the only
/// definition that cannot drift from what actually ran, because the contract's
/// holdout is the same value that set the child's `ZO_ABLATE`.
///
/// A suite with no ablated runner returns an empty report — an ordinary
/// benchmark run is not a failed experiment.
#[must_use]
pub fn pair_suite_runs(runs: &[Arm<'_>]) -> AblationReport {
    let mut cells: std::collections::BTreeMap<CellKey, Vec<&Arm<'_>>> =
        std::collections::BTreeMap::new();
    for run in runs {
        cells
            .entry(CellKey {
                runner_kind: run.contract.runner_kind.clone(),
                lane: run.contract.lane.clone(),
                fixture_id: run.contract.fixture_id.clone(),
                rep: run.rep,
            })
            .or_default()
            .push(run);
    }

    let mut report = AblationReport::default();
    for (cell, arms) in cells {
        let (baselines, treatments): (Vec<&Arm<'_>>, Vec<&Arm<'_>>) = arms
            .into_iter()
            .partition(|arm| arm.contract.ablation.is_empty());
        if treatments.is_empty() {
            continue;
        }
        // Ambiguity is refused rather than resolved by picking the first: two
        // full-feature runners in one cell may differ in something the contract
        // does not record (their `args`), so choosing between them would make
        // the delta depend on runner ordering.
        let baseline = match baselines.as_slice() {
            [only] => *only,
            other => {
                let reason = if other.is_empty() {
                    "no full-feature arm ran this cell, so there is nothing to compare against"
                        .to_string()
                } else {
                    format!(
                        "{} full-feature arms ran this cell ({}) — the baseline is ambiguous",
                        other.len(),
                        other
                            .iter()
                            .map(|arm| arm.contract.runner.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                report
                    .unpaired
                    .extend(treatments.iter().map(|treatment| UnpairedArm {
                        runner_kind: cell.runner_kind.clone(),
                        runner_ablated: treatment.contract.runner.clone(),
                        lane: cell.lane.clone(),
                        fixture_id: cell.fixture_id.clone(),
                        rep: cell.rep,
                        reason: reason.clone(),
                    }));
                continue;
            }
        };

        for treatment in treatments {
            match measure(*baseline, *treatment) {
                Ok(delta) => report.pairs.push(delta),
                Err(reasons) => report.rejected.push(RejectedPair {
                    runner_full: baseline.contract.runner.clone(),
                    runner_ablated: treatment.contract.runner.clone(),
                    lane: cell.lane.clone(),
                    fixture_id: cell.fixture_id.clone(),
                    rep: cell.rep,
                    messages: reasons.iter().map(ToString::to_string).collect(),
                    reasons,
                }),
            }
        }
    }

    report.aggregates = aggregate(&report.pairs);
    report
}

/// Roll pairs up per (runner kind, holdout, lane). Lanes are never merged —
/// the same precedent `summarize_ledger` follows, and for the same reason: a
/// fast-lane task and a deep-lane task are not samples of one population.
fn aggregate(pairs: &[AblationDelta]) -> Vec<AblationAggregate> {
    let mut grouped: std::collections::BTreeMap<(String, Vec<String>, String), Vec<&AblationDelta>> =
        std::collections::BTreeMap::new();
    for pair in pairs {
        grouped
            .entry((
                pair.runner_kind.clone(),
                pair.held_out.clone(),
                pair.lane.clone(),
            ))
            .or_default()
            .push(pair);
    }

    grouped
        .into_iter()
        .map(|((runner_kind, held_out, lane), group)| {
            let count = |shift: OutcomeShift| {
                group.iter().filter(|pair| pair.outcome == shift).count()
            };
            let passed_only_with_feature = count(OutcomeShift::PassedOnlyWithFeature);
            let passed_only_without_feature = count(OutcomeShift::PassedOnlyWithoutFeature);
            let token_costs: Vec<i64> = group
                .iter()
                .filter_map(|pair| pair.token_cost_of_feature())
                .collect();
            AblationAggregate {
                runner_kind,
                held_out,
                lane,
                pairs: group.len(),
                correctness_informative_pairs: group
                    .iter()
                    .filter(|pair| pair.outcome.is_correctness_informative())
                    .count(),
                passed_only_with_feature,
                passed_only_without_feature,
                both_passed: count(OutcomeShift::BothPassed),
                both_failed: count(OutcomeShift::BothFailed),
                discordant_two_sided_p: discordant_two_sided_p(
                    passed_only_with_feature,
                    passed_only_without_feature,
                ),
                evidence: evidence_verdict(
                    passed_only_with_feature,
                    passed_only_without_feature,
                    group.len(),
                    EVIDENCE_ALPHA,
                ),
                token_cost_pairs: token_costs.len(),
                token_cost_of_feature_total: (!token_costs.is_empty())
                    .then(|| token_costs.iter().sum()),
                token_cost_of_feature_median: median(token_costs),
                wall_cost_of_feature_total: group
                    .iter()
                    .map(|pair| pair.wall_cost_of_feature())
                    .sum(),
            }
        })
        .collect()
}

/// Validate and normalize a `ZO_ABLATE` spec into the keys a contract records.
///
/// Routed through [`telemetry::AblationSet`] so the harness and the runtime
/// share ONE vocabulary of feature names. A harness that accepted a name the
/// runtime does not know would set an environment variable that suppresses
/// nothing and then file the result as a control arm.
///
/// # Errors
///
/// Returns the parse error when a token names no harness feature.
pub fn holdout_keys(spec: &str) -> Result<Vec<String>, telemetry::AblationSpecError> {
    Ok(telemetry::AblationSet::parse(spec)?
        .keys()
        .into_iter()
        .map(str::to_string)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fairness::{build_contract, FairnessInput};
    use crate::runner::{normalize_effort_label, normalize_model_label, Tokens};
    use crate::TestStatus;

    fn base_input(ablation: &[&str]) -> FairnessInput {
        FairnessInput {
            runner: "zo".to_string(),
            runner_kind: "zo".to_string(),
            lane: "deep".to_string(),
            fixture_id: "wide-rename".to_string(),
            fixture_tree_hash: "abc123treehash".to_string(),
            prompt: "Rename fetch to load.".to_string(),
            test_command: "node --test".to_string(),
            intended_path_set: vec!["src/a.js".to_string()],
            declared_model: "claude-opus-5".to_string(),
            declared_effort: "high".to_string(),
            permission_mode: "danger-full-access".to_string(),
            timeout_seconds: 300,
            runner_version: "zo 0.1.13".to_string(),
            harness_version: "1.0".to_string(),
            benchmark_suite_version: "1.0".to_string(),
            ablation: ablation.iter().map(|key| (*key).to_string()).collect(),
            ..FairnessInput::default()
        }
    }

    fn contract_with(ablation: &[&str], mutate: impl FnOnce(&mut FairnessInput)) -> FairnessContract {
        let mut input = base_input(ablation);
        mutate(&mut input);
        build_contract(&input)
    }

    /// A result that belongs to `contract` — same runner, lane, and the
    /// normalized labels the spec would have carried into the run.
    fn result_for(contract: &FairnessContract, pass: bool, wall_seconds: u64, tokens: Option<Tokens>) -> RunResult {
        RunResult {
            runner: contract.runner.clone(),
            model: normalize_model_label(&contract.declared_model),
            effort: normalize_effort_label(&contract.declared_effort),
            lane: contract.lane.clone(),
            exit_code: 0,
            wall_seconds,
            startup_seconds: None,
            test: if pass { TestStatus::Pass } else { TestStatus::Fail },
            intended_changed: 1,
            permission_denials: 0,
            pollution: Vec::new(),
            unexpected: Vec::new(),
            clean_diff: true,
            pass,
            tokens,
            iterations: None,
            fail_reasons: Vec::new(),
            warnings: Vec::new(),
            artifact_dir: None,
            deep: None,
        }
    }

    fn tokens(total: u64, complete: bool) -> Tokens {
        Tokens {
            input: total / 2,
            output: total / 2,
            cache_creation: complete.then_some(0),
            cache_read: complete.then_some(0),
            total,
            complete,
        }
    }

    fn condition_names(rejections: &[PairRejection]) -> Vec<&'static str> {
        rejections
            .iter()
            .filter_map(|rejection| match rejection {
                PairRejection::ConditionDiffers { condition, .. } => Some(*condition),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn an_ablation_arm_records_sorted_deduped_keys_and_a_full_run_omits_the_field() {
        let ablated = contract_with(&["info_topology", "routing_probe", "info_topology"], |_| {});
        assert_eq!(ablated.ablation, vec!["info_topology", "routing_probe"]);

        let full = contract_with(&[], |_| {});
        assert!(full.ablation.is_empty());
        let json = serde_json::to_value(&full).expect("serialize");
        assert!(
            json.get("ablation").is_none(),
            "a full-feature contract must stay byte-identical to a pre-ablation one: {json}"
        );
    }

    #[test]
    fn a_pair_differing_only_in_the_holdout_measures_the_feature() {
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["info_topology"], |_| {});
        let full_result = result_for(&full, true, 90, Some(tokens(30_000, true)));
        let ablated_result = result_for(&ablated, false, 70, Some(tokens(24_000, true)));

        let delta = measure(
            Arm::new(&full, &full_result, 0),
            Arm::new(&ablated, &ablated_result, 0),
        )
        .expect("arms differ only in the holdout");

        assert_eq!(delta.held_out, vec!["info_topology"]);
        assert_eq!(delta.outcome, OutcomeShift::PassedOnlyWithFeature);
        assert!(delta.outcome.arms_disagreed());
        assert!(delta.outcome.is_correctness_informative());
        assert!(delta.token_accounting_complete);
        assert_eq!(delta.token_cost_of_feature(), Some(6_000));
        assert_eq!(delta.wall_cost_of_feature(), 20);
    }

    #[test]
    fn a_pair_where_only_the_ablated_arm_passed_is_reported_not_swallowed() {
        // The direction a harness that only looked for improvements would miss.
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["design_guidance"], |_| {});
        let delta = measure(
            Arm::new(&full, &result_for(&full, false, 120, None), 0),
            Arm::new(&ablated, &result_for(&ablated, true, 60, None), 0),
        )
        .expect("comparable");

        assert_eq!(delta.outcome, OutcomeShift::PassedOnlyWithoutFeature);
        assert!(delta.outcome.arms_disagreed());
        assert_eq!(delta.wall_cost_of_feature(), 60);
    }

    #[test]
    fn a_task_both_arms_fail_carries_no_correctness_signal() {
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["routing_probe"], |_| {});
        let delta = measure(
            Arm::new(&full, &result_for(&full, false, 30, None), 0),
            Arm::new(&ablated, &result_for(&ablated, false, 30, None), 0),
        )
        .expect("comparable");

        assert_eq!(delta.outcome, OutcomeShift::BothFailed);
        assert!(!delta.outcome.arms_disagreed());
        assert!(
            !delta.outcome.is_correctness_informative(),
            "a task neither arm can do says nothing about the feature"
        );
    }

    #[test]
    fn incomplete_token_accounting_withholds_the_cost_instead_of_publishing_a_lower_bound() {
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["routing_probe"], |_| {});
        let delta = measure(
            Arm::new(&full, &result_for(&full, true, 10, Some(tokens(100, true))), 0),
            Arm::new(
                &ablated,
                &result_for(&ablated, true, 10, Some(tokens(80, false))),
                0,
            ),
        )
        .expect("comparable");

        assert!(!delta.token_accounting_complete);
        assert_eq!(
            delta.token_cost_of_feature(),
            None,
            "an incomplete total omits cache classes, so their difference is a lower bound \
             of unknown tightness and must not be reported as the cost"
        );
        // The raw per-arm totals stay available for a caller that says so.
        assert_eq!(delta.token_total_full, Some(100));
        assert_eq!(delta.token_total_ablated, Some(80));
    }

    #[test]
    fn every_held_constant_condition_actually_refuses_a_divergent_pair() {
        // Table-driven so a condition cannot be dropped from `HELD_CONSTANT`
        // while the suite stays green: each mutation must be caught by name.
        type InputMutation = (&'static str, fn(&mut FairnessInput));
        let mutations: Vec<InputMutation> = vec![
            ("runner_kind", |input| {
                input.runner_kind = "claude".to_string();
            }),
            ("lane", |input| input.lane = "fast".to_string()),
            ("fixture_id", |input| input.fixture_id = "other".to_string()),
            ("fixture_tree_hash", |input| {
                input.fixture_tree_hash = "deadbeef".to_string();
            }),
            ("fixture_commit", |input| {
                input.fixture_commit = "c0ffee".to_string();
            }),
            ("prompt_sha256", |input| {
                input.prompt = "a different task entirely".to_string();
            }),
            ("test_command_sha256", |input| {
                input.test_command = "pytest".to_string();
            }),
            ("intended_path_set_sha256", |input| {
                input.intended_path_set = vec!["src/z.js".to_string()];
            }),
            ("declared_model", |input| {
                input.declared_model = "claude-sonnet-5".to_string();
            }),
            ("declared_effort", |input| {
                input.declared_effort = "low".to_string();
            }),
            ("permission_mode", |input| {
                input.permission_mode = "acceptEdits".to_string();
            }),
            ("timeout_seconds", |input| input.timeout_seconds = 60),
            ("runner_version", |input| {
                input.runner_version = "zo 0.1.14".to_string();
            }),
            ("harness_version", |input| {
                input.harness_version = "2.0".to_string();
            }),
            ("benchmark_suite_version", |input| {
                input.benchmark_suite_version = "2.0".to_string();
            }),
        ];
        assert_eq!(
            mutations.len(),
            HELD_CONSTANT.len() - 1,
            "every held-constant condition needs a mutation case. The one condition this \
             table cannot carry is `fairness_contract_version`, which no `FairnessInput` \
             can express because `build_contract` stamps it — see \
             `a_contract_written_under_a_different_schema_version_is_refused`"
        );

        let full = contract_with(&[], |_| {});
        for (condition, mutate) in mutations {
            let ablated = contract_with(&["routing_probe"], mutate);
            let names = condition_names(&reject_reasons(&full, &ablated));
            assert!(
                names.contains(&condition),
                "diverging {condition} must refuse the pair, got {names:?}"
            );
        }
    }

    #[test]
    fn a_contract_written_under_a_different_schema_version_is_refused() {
        // The one held-constant condition a `FairnessInput` cannot reach:
        // `build_contract` stamps the version, so a divergence can only arrive
        // from a contract some OTHER writer produced — which is exactly the
        // case worth refusing, because two schema versions need not mean the
        // same thing field for field.
        let full = contract_with(&[], |_| {});
        let mut ablated = contract_with(&["routing_probe"], |_| {});
        ablated.fairness_contract_version = "2.0".to_string();

        let names = condition_names(&reject_reasons(&full, &ablated));
        assert!(
            names.contains(&"fairness_contract_version"),
            "a schema-version divergence must refuse the pair, got {names:?}"
        );
    }

    #[test]
    fn every_divergent_condition_is_reported_in_one_pass() {
        // An operator repairing a misconfigured sweep needs the whole list,
        // not one rejection per round-trip.
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["routing_probe"], |input| {
            input.declared_model = "claude-sonnet-5".to_string();
            input.prompt = "a different task entirely".to_string();
            input.timeout_seconds = 60;
        });

        let names = condition_names(&reject_reasons(&full, &ablated));
        assert!(names.contains(&"prompt_sha256"), "{names:?}");
        assert!(names.contains(&"declared_model"), "{names:?}");
        assert!(names.contains(&"timeout_seconds"), "{names:?}");
    }

    #[test]
    fn two_releases_of_one_model_family_are_a_confound_not_a_match() {
        // `normalize_model_family` maps both of these to "opus" — correct for
        // the cross-runner comparison it was built for, and wrong here.
        assert_eq!(
            crate::fairness::normalize_model_family("claude-opus-5"),
            crate::fairness::normalize_model_family("claude-opus-4-8"),
            "the premise: the coarse family check cannot tell these apart"
        );
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["routing_probe"], |input| {
            input.declared_model = "claude-opus-4-8".to_string();
        });
        assert!(condition_names(&reject_reasons(&full, &ablated)).contains(&"declared_model"));
    }

    #[test]
    fn a_condition_blank_on_both_arms_is_refused_rather_than_counted_as_equal() {
        // Two blanks compare equal, which would silently certify that the arms
        // shared a runner build they never recorded. `judge` files this as
        // `partial`, not `invalid`, so nothing upstream catches it.
        let full = contract_with(&[], |input| input.runner_version = String::new());
        let ablated = contract_with(&["routing_probe"], |input| {
            input.runner_version = String::new();
        });
        assert_eq!(full.status, "partial", "premise: not invalid, so only this check stands");

        let rejections = reject_reasons(&full, &ablated);
        assert!(
            rejections.contains(&PairRejection::ConditionUndeclared {
                condition: "runner_version"
            }),
            "{rejections:?}"
        );
    }

    #[test]
    fn a_holdout_that_names_no_feature_is_refused_on_either_arm() {
        // The runner refuses this before spawning, but a contract is data and
        // can arrive from any producer.
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["desgin_guidance"], |_| {});
        let rejections = reject_reasons(&full, &ablated);
        assert!(
            rejections.contains(&PairRejection::UnknownHoldout {
                arm: "ablated",
                token: "desgin_guidance".to_string(),
            }),
            "{rejections:?}"
        );
        assert!(rejections[0].to_string().contains("desgin_guidance"));
    }

    #[test]
    fn a_result_from_another_run_cannot_be_paired_with_this_contract() {
        // The delta takes outcome from the result and identity from the
        // contract, so without this check a mismatched pair yields a
        // valid-looking number for a comparison that never happened.
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["routing_probe"], |_| {});
        let mut foreign = result_for(&ablated, false, 70, None);
        foreign.lane = "fast".to_string();

        let rejections = measure(
            Arm::new(&full, &result_for(&full, true, 90, None), 0),
            Arm::new(&ablated, &foreign, 0),
        )
        .expect_err("a result from another lane does not belong to this contract");

        assert!(
            rejections.contains(&PairRejection::ResultDoesNotMatchContract {
                arm: "ablated",
                field: "lane",
                contract: "deep".to_string(),
                result: "fast".to_string(),
            }),
            "{rejections:?}"
        );
    }

    #[test]
    fn two_unablated_arms_are_a_variance_measurement_not_an_experiment() {
        let full = contract_with(&[], |_| {});
        let also_full = contract_with(&[], |_| {});
        assert_eq!(
            reject_reasons(&full, &also_full),
            vec![PairRejection::NoHoldout]
        );
    }

    #[test]
    fn an_ablated_baseline_is_refused_because_the_delta_would_be_treatment_to_treatment() {
        let full = contract_with(&["routing_probe"], |_| {});
        let ablated = contract_with(&["routing_probe", "info_topology"], |_| {});
        assert_eq!(
            reject_reasons(&full, &ablated),
            vec![PairRejection::BaselineIsAblated {
                held_out: vec!["routing_probe".to_string()]
            }]
        );
    }

    #[test]
    fn an_invalid_arm_is_reported_alongside_every_other_defect_it_has() {
        // Collecting rather than short-circuiting is the documented contract:
        // an invalid run is often invalid AND misconfigured, and stopping at
        // the first finding would hide the second until the next round.
        let full = contract_with(&[], |input| input.fixture_dirty_before = true);
        let ablated = contract_with(&["routing_probe"], |input| {
            input.timeout_seconds = 60;
            input.declared_effort = "low".to_string();
        });
        let rejections = reject_reasons(&full, &ablated);

        assert!(
            matches!(
                rejections.first(),
                Some(PairRejection::ArmInvalid { arm: "full", .. })
            ),
            "the invalid arm leads the list: {rejections:?}"
        );
        let names = condition_names(&rejections);
        assert!(names.contains(&"timeout_seconds"), "{rejections:?}");
        assert!(names.contains(&"declared_effort"), "{rejections:?}");
    }

    #[test]
    fn the_two_arms_of_one_experiment_have_different_names_and_still_pair() {
        // The defect this pins: the arms of a suite ablation are two RUNNERS
        // over one binary, so their names cannot be equal — each owns an output
        // directory and its own `<NAME>_ABLATE`. Holding the name constant made
        // every pair the suite is capable of producing reject on `runner`,
        // i.e. the measurement apparatus could not produce a measurement.
        let full = contract_with(&[], |input| input.runner = "zo".to_string());
        let ablated = contract_with(&["routing_probe"], |input| {
            input.runner = "zo_noprobe".to_string();
        });
        assert_ne!(full.runner, ablated.runner, "premise: the names differ");
        assert_eq!(full.runner_kind, ablated.runner_kind, "premise: one runner");

        let delta = measure(
            Arm::new(&full, &result_for(&full, true, 90, None), 0),
            Arm::new(&ablated, &result_for(&ablated, false, 80, None), 0),
        )
        .expect("differing arm names are the design, not a confound");
        assert_eq!(delta.runner_full, "zo");
        assert_eq!(delta.runner_ablated, "zo_noprobe");
        assert_eq!(delta.runner_kind, "zo");
    }

    #[test]
    fn two_different_runners_never_pair_even_when_one_declares_a_holdout() {
        // The case dropping the name check must NOT let through: the holdout
        // would be confounded with the entire runner.
        let full = contract_with(&[], |input| {
            input.runner = "claude".to_string();
            input.runner_kind = "claude".to_string();
        });
        let ablated = contract_with(&["routing_probe"], |_| {});
        assert!(condition_names(&reject_reasons(&full, &ablated)).contains(&"runner_kind"));
    }

    #[test]
    fn a_pair_whose_arms_both_omit_the_runner_kind_is_refused_not_certified() {
        let full = contract_with(&[], |input| input.runner_kind = String::new());
        let ablated = contract_with(&["routing_probe"], |input| {
            input.runner_kind = String::new();
        });
        assert!(reject_reasons(&full, &ablated).contains(&PairRejection::ConditionUndeclared {
            condition: "runner_kind"
        }));
    }

    /// A cell's worth of arms for the pairing tests: one baseline plus one
    /// treatment, with the outcomes the caller wants.
    struct Cell {
        full: FairnessContract,
        ablated: FairnessContract,
        full_result: RunResult,
        ablated_result: RunResult,
    }

    fn cell(
        fixture: &str,
        held_out: &[&str],
        full_pass: bool,
        ablated_pass: bool,
        tokens_full: u64,
        tokens_ablated: u64,
    ) -> Cell {
        let full = contract_with(&[], |input| {
            input.runner = "zo".to_string();
            input.fixture_id = fixture.to_string();
        });
        let ablated = contract_with(held_out, |input| {
            input.runner = "zo_ablated".to_string();
            input.fixture_id = fixture.to_string();
        });
        let full_result = result_for(&full, full_pass, 100, Some(tokens(tokens_full, true)));
        let ablated_result = result_for(&ablated, ablated_pass, 80, Some(tokens(tokens_ablated, true)));
        Cell {
            full,
            ablated,
            full_result,
            ablated_result,
        }
    }

    fn arms_of(cells: &[Cell]) -> Vec<Arm<'_>> {
        cells
            .iter()
            .flat_map(|c| {
                [
                    Arm::new(&c.full, &c.full_result, 0),
                    Arm::new(&c.ablated, &c.ablated_result, 0),
                ]
            })
            .collect()
    }

    #[test]
    fn a_suite_with_no_ablated_runner_produces_no_experiment_at_all() {
        // An ordinary benchmark run must not emit a report full of `NoHoldout`
        // rejections: that would read as an experiment that failed rather than
        // as a suite that was never one.
        let full = contract_with(&[], |_| {});
        let also_full = contract_with(&[], |input| input.runner = "claude".to_string());
        let (r1, r2) = (
            result_for(&full, true, 10, None),
            result_for(&also_full, true, 10, None),
        );
        let report = pair_suite_runs(&[Arm::new(&full, &r1, 0), Arm::new(&also_full, &r2, 0)]);
        assert!(report.is_empty(), "{report:?}");
        assert!(report.aggregates.is_empty());
    }

    #[test]
    fn a_suite_pairs_every_cell_and_rolls_the_lane_up() {
        let cells = vec![
            cell("a", &["routing_probe"], true, false, 30_000, 24_000),
            cell("b", &["routing_probe"], true, false, 20_000, 15_000),
            cell("c", &["routing_probe"], true, true, 10_000, 9_000),
            cell("d", &["routing_probe"], false, false, 5_000, 5_000),
            cell("e", &["routing_probe"], false, true, 8_000, 6_000),
        ];
        let report = pair_suite_runs(&arms_of(&cells));

        assert_eq!(report.pairs.len(), 5, "{:?}", report.rejected);
        assert!(report.rejected.is_empty() && report.unpaired.is_empty());
        let rollup = &report.aggregates[0];
        assert_eq!(rollup.runner_kind, "zo");
        assert_eq!(rollup.held_out, vec!["routing_probe"]);
        assert_eq!(rollup.lane, "deep");
        assert_eq!(rollup.pairs, 5);
        assert_eq!(
            rollup.correctness_informative_pairs, 4,
            "the pair both arms failed carries no correctness signal"
        );
        assert_eq!(rollup.passed_only_with_feature, 2);
        assert_eq!(rollup.passed_only_without_feature, 1);
        assert_eq!(rollup.both_passed, 1);
        assert_eq!(rollup.both_failed, 1);
        // 2-vs-1 discordant is nowhere near evidence, and the p-value is what
        // stops the report's reader from treating it as such.
        assert_eq!(rollup.discordant_two_sided_p, Some(1.0));
        assert_eq!(rollup.token_cost_pairs, 5);
        assert_eq!(
            rollup.token_cost_of_feature_total,
            // 6k + 5k + 1k from the three where the arms differed, nothing
            // from the pair that billed the same, 2k from the last.
            Some(6_000 + 5_000 + 1_000 + 2_000)
        );
        assert_eq!(rollup.token_cost_of_feature_median, Some(2_000));
        assert_eq!(rollup.wall_cost_of_feature_total, 5 * 20);
    }

    #[test]
    fn lanes_are_never_merged_into_one_rollup() {
        // Same precedent as `summarize_ledger`: a fast task and a deep task are
        // not samples of one population.
        let fast_full = contract_with(&[], |input| input.lane = "fast".to_string());
        let fast_ablated = contract_with(&["routing_probe"], |input| {
            input.lane = "fast".to_string();
            input.runner = "zo_ablated".to_string();
        });
        let deep = cell("a", &["routing_probe"], true, true, 10, 10);
        let (fr, ar) = (
            result_for(&fast_full, true, 10, None),
            result_for(&fast_ablated, true, 10, None),
        );
        let mut arms = arms_of(std::slice::from_ref(&deep));
        arms.push(Arm::new(&fast_full, &fr, 0));
        arms.push(Arm::new(&fast_ablated, &ar, 0));

        let report = pair_suite_runs(&arms);
        let lanes: Vec<&str> = report.aggregates.iter().map(|a| a.lane.as_str()).collect();
        assert_eq!(lanes, vec!["deep", "fast"]);
    }

    #[test]
    fn an_ablated_arm_with_no_baseline_is_reported_rather_than_dropped() {
        // Silently dropping it would make a cell nobody measured look exactly
        // like a cell where the feature made no difference.
        let ablated = contract_with(&["routing_probe"], |input| {
            input.runner = "zo_ablated".to_string();
        });
        let result = result_for(&ablated, true, 10, None);
        let report = pair_suite_runs(&[Arm::new(&ablated, &result, 0)]);

        assert!(report.pairs.is_empty());
        assert_eq!(report.unpaired.len(), 1);
        assert_eq!(report.unpaired[0].runner_ablated, "zo_ablated");
        assert!(report.unpaired[0].reason.contains("no full-feature arm"));
        assert!(!report.is_empty(), "an unpaired arm is still an experiment");
    }

    #[test]
    fn two_baselines_in_one_cell_are_ambiguous_rather_than_resolved_by_order() {
        // Picking the first would make the delta depend on runner ordering,
        // and the two baselines can differ in `args`, which no contract records.
        let full = contract_with(&[], |input| input.runner = "zo".to_string());
        let other_full = contract_with(&[], |input| input.runner = "zo_b".to_string());
        let ablated = contract_with(&["routing_probe"], |input| {
            input.runner = "zo_ablated".to_string();
        });
        let (r1, r2, r3) = (
            result_for(&full, true, 10, None),
            result_for(&other_full, true, 10, None),
            result_for(&ablated, false, 10, None),
        );
        let report = pair_suite_runs(&[
            Arm::new(&full, &r1, 0),
            Arm::new(&other_full, &r2, 0),
            Arm::new(&ablated, &r3, 0),
        ]);

        assert!(report.pairs.is_empty());
        assert_eq!(report.unpaired.len(), 1);
        assert!(report.unpaired[0].reason.contains("ambiguous"), "{report:?}");
        assert!(report.unpaired[0].reason.contains("zo_b"));
    }

    #[test]
    fn a_cell_whose_pair_is_refused_reports_the_reasons_in_readable_form() {
        let full = contract_with(&[], |_| {});
        let ablated = contract_with(&["routing_probe"], |input| {
            input.runner = "zo_ablated".to_string();
            input.declared_effort = "low".to_string();
        });
        let (r1, r2) = (
            result_for(&full, true, 10, None),
            result_for(&ablated, true, 10, None),
        );
        let report = pair_suite_runs(&[Arm::new(&full, &r1, 0), Arm::new(&ablated, &r2, 0)]);

        assert!(report.pairs.is_empty());
        assert_eq!(report.rejected.len(), 1);
        let rejected = &report.rejected[0];
        assert_eq!(rejected.runner_ablated, "zo_ablated");
        assert!(
            rejected.messages.iter().any(|m| m.contains("declared_effort")),
            "{:?}",
            rejected.messages
        );
        assert_eq!(rejected.reasons.len(), rejected.messages.len());
    }

    #[test]
    fn reps_of_one_cell_pair_index_for_index_and_all_count_toward_the_rollup() {
        let rep0 = cell("a", &["routing_probe"], true, false, 100, 80);
        let rep1 = cell("a", &["routing_probe"], false, false, 100, 80);
        let arms = vec![
            Arm::new(&rep0.full, &rep0.full_result, 0),
            Arm::new(&rep0.ablated, &rep0.ablated_result, 0),
            Arm::new(&rep1.full, &rep1.full_result, 1),
            Arm::new(&rep1.ablated, &rep1.ablated_result, 1),
        ];
        let report = pair_suite_runs(&arms);

        assert_eq!(report.pairs.len(), 2, "{:?}", report.rejected);
        let paired_reps: Vec<usize> = report.pairs.iter().map(|p| p.rep).collect();
        assert_eq!(
            paired_reps,
            vec![0, 1],
            "each rep is its own pair, not one merged"
        );
        assert_eq!(report.aggregates[0].pairs, 2);
    }

    #[test]
    fn the_sign_test_calls_a_small_streak_inconclusive_and_a_long_one_evidence() {
        // The whole reason this number is in the rollup: a 2–0 flip reads as a
        // clean sweep and is worth p = 0.5.
        assert_eq!(discordant_two_sided_p(0, 0), None, "no test without discord");
        assert_eq!(discordant_two_sided_p(1, 0), Some(1.0));
        assert_eq!(discordant_two_sided_p(2, 0), Some(0.5));
        assert_eq!(discordant_two_sided_p(3, 0), Some(0.25));
        let ten = discordant_two_sided_p(10, 0).expect("ten discordant pairs");
        assert!((ten - 2.0 / 1024.0).abs() < 1e-12, "{ten}");
        // Symmetric: the direction does not change the strength of evidence.
        assert_eq!(discordant_two_sided_p(2, 7), discordant_two_sided_p(7, 2));
        // Balanced discord is the null itself.
        assert_eq!(discordant_two_sided_p(5, 5), Some(1.0));
        // Well past where a factorial-based binomial coefficient overflows u64.
        let big = discordant_two_sided_p(80, 0).expect("80 discordant pairs");
        assert!(big > 0.0 && big < 1e-20, "{big}");
    }

    #[test]
    fn the_smallest_p_the_test_can_state_is_returned_rather_than_flushed_to_zero() {
        // The exact boundary of the doubling: p(1075, 0) is 2^-1074, the
        // smallest subnormal f64 and still a representable answer. Computing
        // the tail first and doubling after rounded it away to 0.0, which
        // reports a stronger claim than a sign test can ever make.
        assert_eq!(
            discordant_two_sided_p(1075, 0),
            Some(f64::from_bits(1)),
            "2^-1074 is representable and must not round to zero"
        );
    }

    #[test]
    fn the_median_token_cost_averages_the_middle_of_an_even_count() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![7]), Some(7));
        assert_eq!(median(vec![9, 1, 5]), Some(5));
        assert_eq!(median(vec![10, 2, 8, 4]), Some(6));
        assert_eq!(median(vec![-100, 4]), Some(-48), "a saving is a negative cost");
    }

    #[test]
    fn a_lopsided_flip_too_small_to_test_reports_insufficient_evidence() {
        // The failure this prevents: quoting "2 with, 0 without" as a result.
        // The p-value beside it is 0.5, but a reader can skip a number; a
        // verdict field that says `insufficient_evidence` has to be rendered.
        assert_eq!(
            evidence_verdict(2, 0, 4, EVIDENCE_ALPHA),
            EvidenceVerdict::InsufficientEvidence {
                discordant_pairs: 2,
                min_discordant_pairs_all_one_direction: 6,
            }
        );
    }

    #[test]
    fn six_discordant_pairs_is_the_smallest_suite_that_can_conclude_anything() {
        // Quantifies the sample-size requirement instead of leaving it as
        // folklore: below six flips NO split reaches alpha, and at six an
        // all-one-direction split just does (p = 2·2^-6 = 0.03125).
        assert_eq!(min_discordant_pairs_for(EVIDENCE_ALPHA), Some(6));
        assert!(discordant_two_sided_p(5, 0).is_some_and(|p| p > EVIDENCE_ALPHA));
        assert!(discordant_two_sided_p(6, 0).is_some_and(|p| p <= EVIDENCE_ALPHA));
        assert_eq!(
            evidence_verdict(6, 0, 6, EVIDENCE_ALPHA),
            EvidenceVerdict::DifferenceFavoursFeature
        );
    }

    #[test]
    fn a_powered_but_balanced_split_is_no_difference_not_insufficient() {
        // Enough pairs to have seen a difference; none was there. That is a
        // real finding and must not be filed as a sample-size problem.
        assert_eq!(
            evidence_verdict(4, 4, 8, EVIDENCE_ALPHA),
            EvidenceVerdict::NoDifferenceDetected
        );
    }

    #[test]
    fn a_difference_favouring_the_holdout_is_reported_with_the_same_force() {
        // The direction a harness looking only for improvements would miss.
        assert_eq!(
            evidence_verdict(0, 6, 6, EVIDENCE_ALPHA),
            EvidenceVerdict::DifferenceFavoursHoldout
        );
    }

    #[test]
    fn a_rollup_carries_its_evidence_verdict() {
        let cells = [
            cell("wide-rename", &["routing_probe"], true, false, 100, 80),
            cell("narrow-fix", &["routing_probe"], true, false, 100, 80),
        ];
        let report = pair_suite_runs(&arms_of(&cells));
        let rollup = &report.aggregates[0];
        assert_eq!(rollup.passed_only_with_feature, 2);
        assert_eq!(
            rollup.evidence,
            EvidenceVerdict::InsufficientEvidence {
                discordant_pairs: 2,
                min_discordant_pairs_all_one_direction: 6,
            },
            "two fixtures cannot conclude, and the rollup has to say so"
        );
    }

    #[test]
    fn a_near_balanced_large_sample_is_not_significant() {
        // Regression for a false-significant defect: the tail used to start at
        // 2^-n, which underflows to zero for n above ~1075. Every later term
        // then stayed zero, so this split came back p = 0 and the verdict read
        // `DifferenceFavoursFeature` on a sample that shows nothing.
        let p = discordant_two_sided_p(1001, 999).expect("a tail exists for n > 0");
        assert!(p > 0.9, "1001-vs-999 is essentially balanced, got p = {p}");
        assert_eq!(
            evidence_verdict(1001, 999, 2000, EVIDENCE_ALPHA),
            EvidenceVerdict::NoDifferenceDetected
        );
    }

    #[test]
    fn a_lopsided_split_over_the_floor_is_still_not_a_difference() {
        // 6 discordant pairs clears the floor, but only a 6-0 sweep reaches
        // alpha there. 5-1 is the likeliest misquote, so it is pinned.
        let p = discordant_two_sided_p(5, 1).expect("tail");
        assert!(p > EVIDENCE_ALPHA, "got p = {p}");
        assert_eq!(
            evidence_verdict(5, 1, 6, EVIDENCE_ALPHA),
            EvidenceVerdict::NoDifferenceDetected
        );
    }

    #[test]
    fn a_significant_split_need_not_be_unanimous() {
        // Guards against a verdict that keys off the floor instead of the
        // actual p-value: 9-1 is not a sweep and is still significant.
        let p = discordant_two_sided_p(9, 1).expect("tail");
        assert!(p <= EVIDENCE_ALPHA, "got p = {p}");
        assert_eq!(
            evidence_verdict(9, 1, 10, EVIDENCE_ALPHA),
            EvidenceVerdict::DifferenceFavoursFeature
        );
    }

    #[test]
    fn arms_that_never_disagreed_are_not_an_under_powered_suite() {
        // Fifty pairs that all agreed is an observation about the feature, not
        // a sample-size problem, and no fixture count is promised to produce a
        // flip. Filing it as `InsufficientEvidence` would prescribe a remedy
        // that may never apply.
        assert_eq!(
            evidence_verdict(0, 0, 50, EVIDENCE_ALPHA),
            EvidenceVerdict::NoDiscordantPairs { pairs: 50 }
        );
    }

    #[test]
    fn an_alpha_no_test_can_be_stated_at_yields_no_floor() {
        // A sentinel here would serialize as an astronomical "pairs needed".
        assert_eq!(min_discordant_pairs_for(0.0), None);
        assert_eq!(min_discordant_pairs_for(-0.1), None);
        assert_eq!(min_discordant_pairs_for(1.5), None);
        assert_eq!(min_discordant_pairs_for(f64::NAN), None);
        assert_eq!(min_discordant_pairs_for(1.0), Some(1));
        // Far below the default, and still a real answer rather than a cap.
        assert_eq!(min_discordant_pairs_for(1e-30), Some(101));
    }

    #[test]
    fn a_report_records_the_level_its_verdicts_were_stated_at() {
        // A threshold that lived only in a Rust const would make two reports
        // written under different constants silently incomparable.
        let report = AblationReport::default();
        assert!((report.evidence_alpha - EVIDENCE_ALPHA).abs() < f64::EPSILON);
        assert_eq!(report.multiple_comparison_adjustment, "none");
        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(json["multiple_comparison_adjustment"], "none");
    }

    #[test]
    fn holdout_keys_share_the_runtime_vocabulary_and_reject_a_typo() {
        assert_eq!(
            holdout_keys(" routing_probe , info_topology ").expect("valid spec"),
            vec!["routing_probe", "info_topology"],
            "keys come back in HarnessFeature display order, not input order"
        );
        let error = holdout_keys("routing_probe,desgin_guidance")
            .expect_err("a name the runtime does not know must not reach a contract");
        assert_eq!(error.token, "desgin_guidance");
    }
}
