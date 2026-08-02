//! Blocking executor for the routing probe (`smart.autoClassifier:
//! "probed"`): one bounded Fast-tier `send_message` call whose parsed
//! `{complexity, risk, confidence}` self-assessment feeds
//! `runtime::fuse_probe_assessment` on top of the deterministic classifier.
//!
//! Everything here fails open to `None` — a missing credential, a timeout, a
//! malformed reply, or a panic-free provider error all leave the caller on
//! the deterministic verdict. Failures remain visible under `ZO_ROUTE_DEBUG`;
//! the probe is a refinement, never a user-facing dependency.

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
const PROBE_MAX_TOKENS: u32 = 512;
/// Process-wide probe memo cap. Entries are keyed by task-text fingerprint,
/// so a fan-out retry or a re-spawned identical task never pays (or waits
/// for) a second probe. Arrived-but-malformed responses are memoized too;
/// transient transport/provider/join failures are not, so a later turn can
/// recover.
const PROBE_CACHE_CAP: usize = 256;
const ROUTE_DEBUG_ENV: &str = "ZO_ROUTE_DEBUG";

/// Every way a probe CALL can go wrong, as a fixed token shared VERBATIM by
/// the `ZO_ROUTE_DEBUG` console echo and the always-on attestation counter —
/// so the two observability surfaces can never disagree about what happened.
///
/// All of these are `attest_failed`, never `attest_declined`: reaching this
/// file already means a gate decided the probe was worth running (the decline
/// gates live in `turn.rs`), so anything that goes wrong from here is a real
/// failure and must escalate as one.
///
/// The counter is the load-bearing surface. `ZO_ROUTE_DEBUG` is off by
/// default, so before the ledger existed a probe that failed on EVERY call
/// emitted nothing at all — which is precisely how a 100%-failing probe
/// survived for weeks (see [`PROBE_BASE_URL_ENV`]'s post-mortem below). The
/// counter makes that state readable from `/smart doctor` with no env var set.
const FAIL_TIMEOUT: &str = "timeout";
const FAIL_PROVIDER_FAILURE: &str = "provider_failure";
const FAIL_JOIN_FAILURE: &str = "join_failure";
const FAIL_MALFORMED: &str = "malformed";
const FAIL_CACHED_MALFORMED: &str = "cached_malformed";
const FAIL_CLIENT_UNAVAILABLE: &str = "client_unavailable";
const FAIL_CACHE_UNAVAILABLE: &str = "cache_unavailable";

enum ProbeCallOutcome {
    Response(Box<api::MessageResponse>),
    TimedOut,
    ProviderFailure,
}

fn route_debug_enabled() -> bool {
    std::env::var(ROUTE_DEBUG_ENV).is_ok()
}

/// Attest one probe firing — a verdict that actually reached the router.
fn attest_probe_fired() {
    telemetry::attest_fired(telemetry::HarnessFeature::RoutingProbe);
}

/// Record one per-task failure on both surfaces: the attestation counter
/// (always) and the `ZO_ROUTE_DEBUG` echo (only when enabled).
fn probe_failed(fingerprint: u64, model: &str, reason: &'static str) {
    telemetry::attest_failed(telemetry::HarnessFeature::RoutingProbe, reason);
    if route_debug_enabled() {
        eprintln!("[ROUTE_PROBE] task={fingerprint:016x} model={model} result={reason}");
    }
}

fn record_probe_outcome(
    results: &mut [Option<ProbeAssessment>],
    fresh: &mut Vec<(u64, Option<ProbeAssessment>)>,
    index: usize,
    fingerprint: u64,
    model: &str,
    outcome: Result<ProbeCallOutcome, tokio::task::JoinError>,
) {
    // Transport failures are not memoized: a later turn may recover.
    let response = match outcome {
        Ok(ProbeCallOutcome::Response(response)) => response,
        Ok(ProbeCallOutcome::TimedOut) => {
            probe_failed(fingerprint, model, FAIL_TIMEOUT);
            return;
        }
        Ok(ProbeCallOutcome::ProviderFailure) => {
            probe_failed(fingerprint, model, FAIL_PROVIDER_FAILURE);
            return;
        }
        Err(_) => {
            probe_failed(fingerprint, model, FAIL_JOIN_FAILURE);
            return;
        }
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
    match assessment {
        Some(assessment) => {
            attest_probe_fired();
            if route_debug_enabled() {
                eprintln!(
                    "[ROUTE_PROBE] task={fingerprint:016x} model={model} result=success \
                     complexity={:?} confidence={:?} intent={:?}",
                    assessment.complexity, assessment.confidence, assessment.intent
                );
            }
        }
        None => probe_failed(fingerprint, model, FAIL_MALFORMED),
    }
    results[index] = assessment;
    // An arrived malformed response is stable enough to memoize: retrying the
    // same task would spend tokens on the same invalid answer.
    fresh.push((fingerprint, assessment));
}

/// Test/diagnostic override for the probe's endpoint (`http://host:port`).
///
/// The probe's LIVE request path had no coverage at all, and it failed 100% of
/// the time for weeks: first `stream: false` → Codex `400 Stream must be set to
/// true`, then `store: false` → a terminal frame carrying `"output": []`. Both
/// are wire failures, invisible to the parse-level tests, and the fail-open
/// design below turned each one into silence rather than an error. Only an
/// end-to-end test against a real HTTP endpoint can catch that class of bug,
/// and the single thing preventing one was credential resolution inside
/// [`build_provider_client_for_agent`].
///
/// When set, the probe — and only the probe — talks to this endpoint through a
/// ChatGPT-backend (Codex Responses) client with a placeholder bearer. Every
/// other client build in the process, and every other credential path, is
/// untouched.
const PROBE_BASE_URL_ENV: &str = "ZO_PROBE_BASE_URL";

/// The client one probe batch sends on: the override endpoint when
/// [`PROBE_BASE_URL_ENV`] names one, else the normally-resolved provider client
/// for `model`.
fn probe_client(model: &str) -> Option<api::ProviderClient> {
    if let Some(base_url) = std::env::var(PROBE_BASE_URL_ENV)
        .ok()
        .map(|url| url.trim().to_string())
        .filter(|url| !url.is_empty())
    {
        return Some(api::ProviderClient::chatgpt_backend_at(
            &base_url,
            "zo-probe-diagnostic-token",
        ));
    }
    build_provider_client_for_agent(model).ok()
}

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
    type ProbeJoin = (usize, u64, tokio::task::JoinHandle<ProbeCallOutcome>);
    // The ablation arm bails out HERE rather than discarding the verdict
    // further down, so the control arm pays none of the probe's latency or
    // tokens either — that cost is part of what an ablation measures. This is
    // also the one entry both the single and batch forms funnel through, so
    // no caller can reach a probe firing around the gate.
    if telemetry::attest_ablated(telemetry::HarnessFeature::RoutingProbe) {
        return vec![None; tasks.len()];
    }
    let mut results: Vec<Option<ProbeAssessment>> = vec![None; tasks.len()];
    let mut misses: Vec<(usize, u64)> = Vec::new();
    {
        let Ok(cache) = probe_cache().lock() else {
            // Batch-wide bail-out: no fingerprint is resolved yet, so this is
            // the one failure attributed to the batch rather than to a task.
            telemetry::attest_failed(
                telemetry::HarnessFeature::RoutingProbe,
                FAIL_CACHE_UNAVAILABLE,
            );
            if route_debug_enabled() {
                eprintln!("[ROUTE_PROBE] result={FAIL_CACHE_UNAVAILABLE}");
            }
            return results;
        };
        for (index, (description, prompt)) in tasks.iter().enumerate() {
            if description.trim().is_empty() && prompt.trim().is_empty() {
                continue;
            }
            let fingerprint = task_fingerprint(description, prompt);
            match cache.get(&fingerprint) {
                Some(memoized) => {
                    results[index] = *memoized;
                    // A memo hit counts as a firing: the question this ledger
                    // answers is whether a probe verdict routed a turn, and a
                    // recalled verdict did exactly that. A memoized MALFORMED
                    // reply stays a failure, since no verdict reached the
                    // router.
                    let result = if memoized.is_some() {
                        attest_probe_fired();
                        "cached_success"
                    } else {
                        telemetry::attest_failed(
                            telemetry::HarnessFeature::RoutingProbe,
                            FAIL_CACHED_MALFORMED,
                        );
                        FAIL_CACHED_MALFORMED
                    };
                    if route_debug_enabled() {
                        eprintln!("[ROUTE_PROBE] task={fingerprint:016x} result={result}");
                    }
                }
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
    let Some(client) = probe_client(&model) else {
        // One failure per task that lost its probe, not one per batch — a
        // credential-less environment should read as "every probe gave up",
        // which is what the count then says.
        for (_, fingerprint) in &misses {
            probe_failed(*fingerprint, &model, FAIL_CLIENT_UNAVAILABLE);
        }
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
                match tokio::time::timeout(PROBE_TIMEOUT, client.send_message(&request)).await {
                    Ok(Ok(response)) => ProbeCallOutcome::Response(Box::new(response)),
                    Ok(Err(_)) => ProbeCallOutcome::ProviderFailure,
                    Err(_) => ProbeCallOutcome::TimedOut,
                }
            });
            (*index, *fingerprint, task)
        })
        .collect();
    let collect_all = async move {
        let mut collected = Vec::with_capacity(probes.len());
        for (index, fingerprint, task) in probes {
            collected.push((index, fingerprint, task.await));
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
    for (index, fingerprint, outcome) in responses {
        record_probe_outcome(
            &mut results,
            &mut fresh,
            index,
            fingerprint,
            &model,
            outcome,
        );
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
        effort: Some(api::EffortLevel::Low),
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

/// End-to-end coverage of the probe's LIVE request path, over a real socket.
///
/// This is the test that did not exist while the probe failed 100% of the time
/// for weeks. Both historical failures were wire-shaped and therefore invisible
/// to [`runtime::parse_probe_response`]'s unit tests:
///
/// 1. `stream: false` — the Codex Responses endpoint rejects it with `400
///    {"detail":"Stream must be set to true"}`.
/// 2. `store: false` — the terminal `response.completed` frame carries
///    `"output": []`; the answer exists only in the `response.output_text.delta`
///    events, so folding the terminal frame verbatim yielded a well-formed
///    response with empty content, which parsed as malformed and failed open.
///
/// The mocks below therefore speak the REAL contract, empty terminal `output`
/// and all: a regression on either fix turns these red instead of silent.
#[cfg(test)]
mod live_path_tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};

    use runtime::{
        ModelInventory, RouteConfidence, RouteTaskComplexity, RouteTaskIntent, RouteTaskRisk,
    };

    use super::{route_probe_assessment, PROBE_BASE_URL_ENV};

    /// `PROBE_BASE_URL_ENV` is process-global and the probe memo is too, so the
    /// live-path cases run one at a time.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Sets `ZO_PROBE_BASE_URL` for the duration of one case and restores the
    /// previous value (or absence) on drop, even if the case panics.
    struct ProbeEnv {
        previous: Option<String>,
        _guard: MutexGuard<'static, ()>,
    }

    impl ProbeEnv {
        fn pointing_at(addr: SocketAddr) -> Self {
            let guard = env_lock();
            let previous = std::env::var(PROBE_BASE_URL_ENV).ok();
            std::env::set_var(PROBE_BASE_URL_ENV, format!("http://{addr}"));
            Self { previous, _guard: guard }
        }
    }

    impl Drop for ProbeEnv {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var(PROBE_BASE_URL_ENV, value),
                None => std::env::remove_var(PROBE_BASE_URL_ENV),
            }
        }
    }

    /// A blocking HTTP/1.1 mock that answers every request with the same canned
    /// status + body and records each request body it saw. Recording *every*
    /// request (not just the first) is what makes the 400 case able to prove
    /// "no retry storm".
    struct MockCodex {
        addr: SocketAddr,
        requests: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
        join: Option<std::thread::JoinHandle<()>>,
    }

    impl MockCodex {
        fn serving(status_line: &'static str, content_type: &'static str, body: String) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock Codex endpoint");
            let addr = listener.local_addr().expect("mock endpoint address");
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let server_requests = Arc::clone(&requests);
            let server_stop = Arc::clone(&stop);
            let join = std::thread::spawn(move || {
                for connection in listener.incoming() {
                    if server_stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut connection) = connection else { break };
                    let Some(request_body) = read_http_request(&mut connection) else {
                        continue;
                    };
                    if let Ok(mut recorded) = server_requests.lock() {
                        recorded.push(request_body);
                    }
                    let head = format!(
                        "{status_line}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\n\r\n",
                        body.len()
                    );
                    let _ = connection.write_all(head.as_bytes());
                    let _ = connection.write_all(body.as_bytes());
                    let _ = connection.flush();
                    let _ = connection.shutdown(Shutdown::Write);
                }
            });
            Self { addr, requests, stop, join: Some(join) }
        }

        fn sse(body: String) -> Self {
            Self::serving("HTTP/1.1 200 OK", "text/event-stream", body)
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().map(|recorded| recorded.clone()).unwrap_or_default()
        }
    }

    impl Drop for MockCodex {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            // Unblock the parked `accept` so the thread observes the stop flag.
            let _ = TcpStream::connect(self.addr);
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }

    /// Read one HTTP request off `connection`, returning its body. Honors
    /// `content-length` so the body is fully drained before the mock replies.
    fn read_http_request(connection: &mut TcpStream) -> Option<String> {
        let mut reader = BufReader::new(connection.try_clone().ok()?);
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).ok()? == 0 {
                return None;
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(value) = trimmed
                .split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim())
            {
                content_length = value.parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).ok()?;
        String::from_utf8(body).ok()
    }

    /// The exact shape the live backend sends under `store: false`: the answer
    /// arrives only as text deltas, and the terminal frame's `output` is empty.
    fn responses_sse(answer: &str) -> String {
        let delta = serde_json::json!({
            "type": "response.output_text.delta",
            "delta": answer,
        });
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_probe",
                "status": "completed",
                "output": [],
                "usage": { "input_tokens": 120, "output_tokens": 24 },
            },
        });
        format!("data: {delta}\n\ndata: {completed}\n\ndata: [DONE]\n\n")
    }

    fn inventory() -> ModelInventory {
        ModelInventory::new("gpt-5.5-codex", Vec::new())
    }

    #[test]
    fn live_probe_fuses_a_four_field_assessment_including_intent() {
        let mock = MockCodex::sse(responses_sse(
            "{\"complexity\":\"large\",\"risk\":\"high\",\"confidence\":\"high\",\"intent\":\"design\"}",
        ));
        let _env = ProbeEnv::pointing_at(mock.addr);

        let assessment = route_probe_assessment(
            &inventory(),
            "gpt-5.5-codex",
            "e2e-intent",
            "redesign the settings screen",
        )
        .expect("a live SSE probe must produce an assessment");

        assert_eq!(assessment.complexity, RouteTaskComplexity::Large);
        assert_eq!(assessment.risk, RouteTaskRisk::High);
        assert_eq!(assessment.confidence, RouteConfidence::High);
        assert_eq!(
            assessment.intent,
            RouteTaskIntent::Design,
            "the intent axis must survive the wire, not just the parser"
        );

        let requests = mock.requests();
        assert_eq!(requests.len(), 1, "one probe, one request");
        let body: serde_json::Value =
            serde_json::from_str(&requests[0]).expect("probe request body is JSON");
        assert_eq!(
            body.get("stream").and_then(serde_json::Value::as_bool),
            Some(true),
            "regression pin: `stream: false` is rejected outright by Codex Responses"
        );
        assert!(
            requests[0].contains("routing classifier"),
            "the real probe prompt must be what goes on the wire"
        );
    }

    #[test]
    fn live_probe_without_the_intent_field_falls_open_to_other() {
        let mock = MockCodex::sse(responses_sse(
            "{\"complexity\":\"small\",\"risk\":\"low\",\"confidence\":\"medium\"}",
        ));
        let _env = ProbeEnv::pointing_at(mock.addr);

        let assessment = route_probe_assessment(
            &inventory(),
            "gpt-5.5-codex",
            "e2e-legacy",
            "rename one config key",
        )
        .expect("a legacy three-field probe must still parse end to end");

        assert_eq!(assessment.complexity, RouteTaskComplexity::Small);
        assert_eq!(assessment.risk, RouteTaskRisk::Low);
        assert_eq!(assessment.confidence, RouteConfidence::Medium);
        assert_eq!(
            assessment.intent,
            RouteTaskIntent::Other,
            "an older probe model that never heard of `intent` must keep working"
        );
        assert_eq!(mock.requests().len(), 1);
    }

    #[test]
    fn live_probe_fails_open_on_a_400_without_panicking_or_retrying() {
        let mock = MockCodex::serving(
            "HTTP/1.1 400 Bad Request",
            "application/json",
            "{\"detail\":\"Stream must be set to true\"}".to_string(),
        );
        let _env = ProbeEnv::pointing_at(mock.addr);

        let assessment = route_probe_assessment(
            &inventory(),
            "gpt-5.5-codex",
            "e2e-rejected",
            "anything at all",
        );

        assert!(assessment.is_none(), "a rejected probe must fail open to None");
        assert_eq!(
            mock.requests().len(),
            1,
            "a 400 is terminal for the probe: no retry storm"
        );
    }
}
