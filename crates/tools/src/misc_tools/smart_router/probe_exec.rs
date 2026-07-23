//! Blocking executor for the routing probe (`smart.autoClassifier:
//! "probed"`): one bounded Fast-tier `send_message` call whose parsed
//! `{complexity, risk, confidence}` self-assessment feeds
//! `runtime::fuse_probe_assessment` on top of the deterministic classifier.
//!
//! Everything here fails open to `None` — a missing credential, a timeout, a
//! malformed reply, or a panic-free provider error all leave the caller on
//! the deterministic verdict. The probe is a refinement, never a dependency.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use api::{InputMessage, MessageRequest, OutputContentBlock};
use runtime::{
    parse_probe_response, probe_prompt, route_model, ModelInventory, ProbeAssessment,
    RouteRequest, RouteRole, RoutingTarget,
};

use crate::misc_tools::agent_tools::{build_provider_client_for_agent, shared_agent_runtime};

/// Hard wall for one probe call. A Fast-tier model answers the ~200-token
/// classification prompt well inside this; anything slower forfeits the
/// probe rather than stall a spawn batch.
const PROBE_TIMEOUT: Duration = Duration::from_secs(8);
/// The reply is one small JSON object; this cap is the cost ceiling per
/// probe, not a target.
const PROBE_MAX_TOKENS: u32 = 128;
/// Process-wide probe memo cap. Entries are keyed by task-text fingerprint,
/// so a fan-out retry or a re-spawned identical task never pays (or waits
/// for) a second probe. Negative results are memoized too — a provider that
/// just failed a probe should not be re-probed per fan-out member.
const PROBE_CACHE_CAP: usize = 256;

fn probe_cache() -> &'static Mutex<HashMap<u64, Option<ProbeAssessment>>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<u64, Option<ProbeAssessment>>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// FNV-1a over both text fields with a length-prefixed separator, so
/// (`"ab"`, `"c"`) and (`"a"`, `"bc"`) cannot collide by concatenation.
fn task_fingerprint(description: &str, prompt: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for chunk in [description.len().to_le_bytes().as_slice(), description.as_bytes(), prompt.as_bytes()] {
        for byte in chunk {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Resolve the model the probe itself runs on: the router's own Fast-role
/// pick from the already-loaded inventory — the probe reuses the engine it
/// serves instead of hand-rolling a second "cheap model" table.
fn probe_model(inventory: &ModelInventory, parent_model: &str) -> String {
    let request = RouteRequest::for_target(
        RoutingTarget::RoleFallback(RouteRole::Fast),
        RouteRole::Fast,
        parent_model,
    );
    route_model(&request, inventory).resolved_model
}

/// Run (or recall) the routing probe for one task. Returns `None` on any
/// failure — the caller stays on the deterministic classification.
pub(super) fn route_probe_assessment(
    inventory: &ModelInventory,
    parent_model: &str,
    description: &str,
    prompt: &str,
) -> Option<ProbeAssessment> {
    route_probe_assessments(inventory, parent_model, &[(description, prompt)])
        .first()
        .copied()
        .flatten()
}

/// Batch form for a fan-out: all cache misses fire CONCURRENTLY inside one
/// `block_on` (per-probe timeout each), so an N-member spawn pays one probe
/// round-trip of wall-clock, not N sequential ones. Results align with
/// `tasks` by index; every failure is a `None` slot (fail open).
pub(super) fn route_probe_assessments(
    inventory: &ModelInventory,
    parent_model: &str,
    tasks: &[(&str, &str)],
) -> Vec<Option<ProbeAssessment>> {
    type ProbeJoin = (usize, u64, tokio::task::JoinHandle<Option<api::MessageResponse>>);
    let mut results: Vec<Option<ProbeAssessment>> = vec![None; tasks.len()];
    let mut misses: Vec<(usize, u64)> = Vec::new();
    {
        let Ok(cache) = probe_cache().lock() else {
            return results;
        };
        for (index, (description, prompt)) in tasks.iter().enumerate() {
            if description.trim().is_empty() && prompt.trim().is_empty() {
                continue;
            }
            let fingerprint = task_fingerprint(description, prompt);
            match cache.get(&fingerprint) {
                Some(memoized) => results[index] = *memoized,
                None => misses.push((index, fingerprint)),
            }
        }
    }
    if misses.is_empty() {
        return results;
    }
    let model = probe_model(inventory, parent_model);
    // Alias-normalized like every sibling `build_provider_client_for_agent`
    // caller (`provider_client.rs`), so a pinned/aliased Fast-role model id
    // cannot misroute provider detection or the wire model.
    let model = api::resolve_model_alias(&model);
    let Ok(client) = build_provider_client_for_agent(&model) else {
        return results;
    };
    let client = std::sync::Arc::new(client);
    let handle = shared_agent_runtime().handle().clone();
    // Each miss is SPAWNED onto the shared agent runtime, so all probes run
    // concurrently and the batch pays roughly one probe of wall-clock (each
    // still individually timeout-bounded), instead of N sequential calls
    // blocking the spawn path. The fingerprint rides the tuple so the
    // result↔cache pairing never depends on collection order.
    let probes: Vec<ProbeJoin> = misses
        .iter()
        .map(|(index, fingerprint)| {
            let (description, prompt) = tasks[*index];
            let request = probe_request(&model, description, prompt);
            let client = std::sync::Arc::clone(&client);
            let task = handle.spawn(async move {
                tokio::time::timeout(PROBE_TIMEOUT, client.send_message(&request))
                    .await
                    .ok()
                    .and_then(Result::ok)
            });
            (*index, *fingerprint, task)
        })
        .collect();
    let collect_all = async move {
        let mut collected = Vec::with_capacity(probes.len());
        for (index, fingerprint, task) in probes {
            collected.push((index, fingerprint, task.await.ok().flatten()));
        }
        collected
    };
    // Sync→async bridge: `run_blocking` uses `block_in_place` only on a
    // multi-thread ambient runtime and falls back to a dedicated runtime
    // otherwise — the hand-rolled `Handle::try_current().is_ok()` guard this
    // replaces panicked on a `current_thread` ambient runtime (the main TUI
    // and headless hosts are exactly that). The probe tasks themselves
    // already run on the shared agent runtime; this only drives the awaits.
    let responses = api::sync_bridge::run_blocking(collect_all);
    let mut fresh: Vec<(u64, Option<ProbeAssessment>)> = Vec::with_capacity(responses.len());
    for (index, fingerprint, response) in responses {
        // Transport failures (timeout, provider error, join error) are NOT
        // memoized — a transient outage must not disable the probe for this
        // task for the rest of the process. A response that ARRIVED is
        // memoized whichever way it parses: a model that answers garbage for
        // this task will keep answering garbage, so re-probing it per spawn
        // would only burn tokens.
        let Some(response) = response else {
            continue;
        };
        let text: String = response
            .content
            .iter()
            .filter_map(|block| match block {
                OutputContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let assessment = parse_probe_response(&text);
        results[index] = assessment;
        fresh.push((fingerprint, assessment));
    }
    if let Ok(mut cache) = probe_cache().lock() {
        if cache.len() + fresh.len() > PROBE_CACHE_CAP {
            cache.clear();
        }
        cache.extend(fresh);
    }
    results
}

fn probe_request(model: &str, description: &str, prompt: &str) -> MessageRequest {
    MessageRequest {
        model: model.to_string(),
        max_tokens: PROBE_MAX_TOKENS,
        messages: vec![InputMessage::user_text(probe_prompt(description, prompt))],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_separates_field_boundaries() {
        assert_ne!(task_fingerprint("ab", "c"), task_fingerprint("a", "bc"));
        assert_eq!(task_fingerprint("a", "b"), task_fingerprint("a", "b"));
    }
}
