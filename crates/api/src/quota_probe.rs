//! Ask each signed-in provider what quota the account has left.
//!
//! Every other quota signal in this crate is a by-product of traffic: the
//! Anthropic headers ride on a response, and the 429 cool-downs only exist once
//! the account is already throttled. Neither can answer the question a session
//! opens with — *how much room do I have right now* — because both require
//! having already spent something to find out.
//!
//! These endpoints answer it directly. They are the same ones the vendors' own
//! desktop clients poll, so the figures match what a user sees elsewhere rather
//! than being a second opinion derived from our traffic alone.
//!
//! Everything here fails soft. A probe that cannot authenticate, cannot reach
//! the network, or gets a shape it does not recognise records nothing and lets
//! [`crate::quota`] fall back to the signals it already had — a missing row
//! reads as "unknown", which is honest, whereas a zero would read as "empty".

use std::time::Duration;

use serde::Deserialize;

use crate::quota::{record_measured_quota, MeasuredQuota, QuotaWindow};
use crate::ProviderKind;

/// Ceiling on a probe. These run on a background cadence and nothing waits on
/// them, so a hung endpoint must not pin a task for minutes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Refresh every provider we hold credentials for.
///
/// Probes run concurrently because they are independent and each one is a
/// network round trip; serially this would be three timeouts deep in the worst
/// case. Failures are per-provider — one signed-out account never suppresses
/// the others.
pub async fn refresh_measured_quotas() {
    let (anthropic, codex) = tokio::join!(probe_anthropic(), probe_codex());
    if let Some(quota) = anthropic {
        record_measured_quota(ProviderKind::Anthropic, quota);
    }
    if let Some(quota) = codex {
        record_measured_quota(ProviderKind::OpenAi, quota);
    }
}

/// How often a background refresh may run.
///
/// Quota windows move over hours, so this is not about freshness — it is about
/// not turning a per-frame HUD rebuild into a per-frame HTTP request.
const REFRESH_CADENCE: Duration = Duration::from_secs(60);

/// Unix millis of the last refresh kick, so the cadence survives across the
/// many places a HUD rebuild can originate.
static LAST_REFRESH_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Kick a refresh in the background if one is due, and return immediately.
///
/// Safe to call from a render path: it never blocks, never awaits, and drops
/// the request entirely when no async runtime is running. The cadence gate is
/// claimed before spawning, so two threads racing here produce one probe.
pub fn refresh_measured_quotas_soon() {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| u64::try_from(since.as_millis()).unwrap_or(u64::MAX));
    let last = LAST_REFRESH_MS.load(std::sync::atomic::Ordering::Relaxed);
    let cadence_ms = u64::try_from(REFRESH_CADENCE.as_millis()).unwrap_or(u64::MAX);
    if last != 0 && now.saturating_sub(last) < cadence_ms {
        return;
    }
    // Claim the slot before spawning so a burst of rebuilds yields one probe.
    if LAST_REFRESH_MS
        .compare_exchange(
            last,
            now,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        return;
    }
    handle.spawn(refresh_measured_quotas());
}

/// Anthropic's OAuth usage endpoint.
///
/// The beta header and the Claude Code user agent are both load-bearing: the
/// endpoint is part of the subscription OAuth surface, not the public API, and
/// it refuses the request without them.
async fn probe_anthropic() -> Option<MeasuredQuota> {
    // Off the async worker: this forks a `security` process to reach the
    // keychain, and blocking a current-thread runtime here would stall the very
    // UI the figure is for.
    let token = tokio::task::spawn_blocking(
        crate::providers::anthropic::keychain::read_claude_code_keychain_token,
    )
    .await
    .ok()
    .flatten()?;
    let response = crate::providers::shared_http_client()
        .get("https://api.anthropic.com/api/oauth/usage")
        .bearer_auth(token)
        .header("anthropic-beta", "oauth-2025-04-20")
        .header("User-Agent", "claude-code/2.1.0")
        .timeout(PROBE_TIMEOUT)
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let usage: AnthropicUsage = response.json().await.ok()?;
    Some(MeasuredQuota {
        windows: usage.windows(),
        // The usage response identifies the quota, not the person. The account
        // label comes from the credential itself elsewhere.
        account: None,
    })
}

/// The ChatGPT backend's usage endpoint, which Codex plans bill against.
async fn probe_codex() -> Option<MeasuredQuota> {
    // Same reasoning as the Anthropic probe: a filesystem read is short but it
    // is still blocking, and the cost of being wrong is a frozen frame.
    let auth = tokio::task::spawn_blocking(CodexAuth::load).await.ok().flatten()?;
    let mut request = crate::providers::shared_http_client()
        .get("https://chatgpt.com/backend-api/wham/usage")
        .bearer_auth(&auth.access_token)
        .header("User-Agent", "codex-cli")
        .header("OpenAI-Beta", "codex-1")
        .header("originator", "Codex Desktop")
        .timeout(PROBE_TIMEOUT);
    if let Some(account_id) = &auth.account_id {
        request = request.header("ChatGPT-Account-Id", account_id);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let usage: CodexUsage = response.json().await.ok()?;
    Some(MeasuredQuota {
        windows: usage.windows(),
        account: usage.email,
    })
}

/// Codex credentials as the Codex CLI writes them.
struct CodexAuth {
    access_token: String,
    account_id: Option<String>,
}

impl CodexAuth {
    /// Read `$CODEX_HOME/auth.json`, defaulting to `~/.codex`.
    fn load() -> Option<Self> {
        #[derive(Deserialize)]
        struct File {
            tokens: Tokens,
        }
        #[derive(Deserialize)]
        struct Tokens {
            access_token: String,
            account_id: Option<String>,
        }

        let home = std::env::var_os("CODEX_HOME").map_or_else(
            || std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".codex")),
            |home| Some(std::path::PathBuf::from(home)),
        )?;
        let raw = std::fs::read_to_string(home.join("auth.json")).ok()?;
        let file: File = serde_json::from_str(&raw).ok()?;
        Some(Self {
            access_token: file.tokens.access_token,
            account_id: file.tokens.account_id,
        })
    }
}

/// Anthropic's usage payload. Only the fields we render are named; the response
/// carries a dozen nullable per-model windows that are absent on most plans.
#[derive(Deserialize)]
struct AnthropicUsage {
    five_hour: Option<AnthropicWindow>,
    seven_day: Option<AnthropicWindow>,
}

#[derive(Deserialize)]
struct AnthropicWindow {
    /// Percent of the window consumed.
    utilization: f64,
    /// RFC 3339 instant the window rolls over.
    resets_at: Option<String>,
}

impl AnthropicUsage {
    fn windows(&self) -> Vec<QuotaWindow> {
        [("5h", self.five_hour.as_ref()), ("7d", self.seven_day.as_ref())]
            .into_iter()
            .filter_map(|(label, window)| window.map(|window| window.to_quota(label)))
            .collect()
    }
}

impl AnthropicWindow {
    fn to_quota(&self, label: &str) -> QuotaWindow {
        QuotaWindow {
            label: label.to_string(),
            remaining_percent: remaining_from_utilization(self.utilization),
            resets_at_unix: self
                .resets_at
                .as_deref()
                .and_then(core_types::date::unix_secs_from_rfc3339)
                .and_then(|secs| u64::try_from(secs).ok()),
        }
    }
}

/// The ChatGPT backend's usage payload.
#[derive(Deserialize)]
struct CodexUsage {
    email: Option<String>,
    rate_limit: Option<CodexRateLimit>,
}

#[derive(Deserialize)]
struct CodexRateLimit {
    primary_window: Option<CodexWindow>,
    secondary_window: Option<CodexWindow>,
}

#[derive(Deserialize)]
struct CodexWindow {
    used_percent: f64,
    /// Window length, which is what names the row — the payload has no label of
    /// its own and "primary"/"secondary" means nothing to a reader.
    limit_window_seconds: Option<u64>,
    /// Unix seconds, unlike Anthropic's RFC 3339 string.
    reset_at: Option<u64>,
}

impl CodexUsage {
    fn windows(&self) -> Vec<QuotaWindow> {
        let Some(limit) = self.rate_limit.as_ref() else {
            return Vec::new();
        };
        [
            limit.primary_window.as_ref(),
            limit.secondary_window.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(CodexWindow::to_quota)
        .collect()
    }
}

impl CodexWindow {
    fn to_quota(&self) -> QuotaWindow {
        QuotaWindow {
            label: self
                .limit_window_seconds
                .map_or_else(|| "limit".to_string(), window_label),
            remaining_percent: remaining_from_utilization(self.used_percent),
            resets_at_unix: self.reset_at,
        }
    }
}

/// Name a window by its length, the way the other quota rows read (`5h`, `7d`).
fn window_label(seconds: u64) -> String {
    let hours = seconds / 3_600;
    if hours >= 24 && hours.is_multiple_of(24) {
        return format!("{}d", hours / 24);
    }
    if hours > 0 {
        return format!("{hours}h");
    }
    format!("{}m", seconds / 60)
}

/// Convert a consumed percentage into remaining headroom.
///
/// Providers report utilization; every surface here reads headroom. Converting
/// once at the edge keeps the inversion out of the render path, where getting
/// it backwards would silently invert the meaning of every gauge.
fn remaining_from_utilization(utilization: f64) -> u8 {
    if !utilization.is_finite() {
        return 0;
    }
    let used = utilization.clamp(0.0, 100.0).round();
    // Safe: the clamp above bounds this to 0..=100.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let used = used as u8;
    100u8.saturating_sub(used)
}

#[cfg(test)]
mod tests {
    use super::{remaining_from_utilization, window_label, AnthropicUsage, CodexUsage};

    /// Verbatim from the live endpoint, so a schema drift shows up here rather
    /// than as a silently empty gauge.
    #[test]
    fn anthropic_payload_maps_to_remaining_headroom() {
        let usage: AnthropicUsage = serde_json::from_str(
            r#"{
                "five_hour": {"utilization": 31.0, "resets_at": "2026-08-03T09:20:00.447442+00:00"},
                "seven_day": {"utilization": 39.0, "resets_at": "2026-08-09T02:00:00.447469+00:00"},
                "seven_day_opus": null
            }"#,
        )
        .expect("the live shape parses");

        let windows = usage.windows();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].label, "5h");
        // 31% spent is 69% left — the inversion happens once, here.
        assert_eq!(windows[0].remaining_percent, 69);
        assert_eq!(windows[0].resets_at_unix, Some(1_785_748_800));
        assert_eq!(windows[1].label, "7d");
        assert_eq!(windows[1].remaining_percent, 61);
    }

    /// Also verbatim, including the null secondary window most plans return.
    #[test]
    fn codex_payload_maps_windows_and_account() {
        let usage: CodexUsage = serde_json::from_str(
            r#"{
                "email": "someone@example.com",
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 15,
                        "limit_window_seconds": 604800,
                        "reset_at": 1786169992
                    },
                    "secondary_window": null
                }
            }"#,
        )
        .expect("the live shape parses");

        assert_eq!(usage.email.as_deref(), Some("someone@example.com"));
        let windows = usage.windows();
        assert_eq!(windows.len(), 1);
        // 604800s is a week, and "7d" is what the other rows call that.
        assert_eq!(windows[0].label, "7d");
        assert_eq!(windows[0].remaining_percent, 85);
        assert_eq!(windows[0].resets_at_unix, Some(1_786_169_992));
    }

    #[test]
    fn window_labels_read_as_durations() {
        assert_eq!(window_label(604_800), "7d");
        assert_eq!(window_label(18_000), "5h");
        assert_eq!(window_label(900), "15m");
    }

    /// A provider that reports nonsense must not invent headroom out of it.
    #[test]
    fn utilization_outside_the_scale_clamps_rather_than_wrapping() {
        assert_eq!(remaining_from_utilization(0.0), 100);
        assert_eq!(remaining_from_utilization(100.0), 0);
        assert_eq!(remaining_from_utilization(140.0), 0);
        assert_eq!(remaining_from_utilization(-20.0), 100);
        assert_eq!(remaining_from_utilization(f64::NAN), 0);
    }
}
