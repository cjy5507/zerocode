//! Minimal calendar-date helpers, dependency-free by design.
//!
//! Single owner of the civil-from-days conversion (Howard Hinnant's
//! `civil_from_days` algorithm) and of "today as `YYYY-MM-DD`". The CLI
//! system prompt, sub-agent prompts, and the usage dashboard all format
//! dates from here so they cannot drift apart — and so nobody is tempted
//! to hardcode a "today" constant again (the prompt date was frozen at
//! `2026-03-31` for months because of exactly that).
//!
//! `build.rs` keeps its own private copy of the algorithm: build scripts
//! cannot depend on workspace crates.

/// Current **local** date as `YYYY-MM-DD` — what a prompt should call
/// "today" (Claude Code parity: the env block shows the user's local date;
/// a KST user's morning is still "yesterday" in UTC until 09:00).
///
/// `std` exposes no timezone database and the workspace forbids `unsafe`
/// (`libc::localtime_r`) and heavy date deps, so this consults the POSIX
/// `date` utility (`%F` = `YYYY-MM-DD`, locale-independent) and falls back
/// to the UTC date when the utility is unavailable (non-POSIX platform) or
/// prints something that is not a plausible date.
#[must_use]
pub fn current_local_date() -> String {
    local_date_from_date_utility().unwrap_or_else(current_utc_date)
}

fn local_date_from_date_utility() -> Option<String> {
    let output = std::process::Command::new("date").arg("+%F").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let candidate = text.trim();
    let shape_ok = candidate.len() == 10
        && candidate.bytes().enumerate().all(|(index, byte)| match index {
            4 | 7 => byte == b'-',
            _ => byte.is_ascii_digit(),
        });
    if shape_ok {
        Some(candidate.to_string())
    } else {
        None
    }
}

/// Current UTC date as `YYYY-MM-DD`, from the system clock.
///
/// Clock-before-epoch (or otherwise unreadable) degrades to the epoch date
/// rather than panicking: prompt assembly must never abort over a bad clock.
#[must_use]
pub fn current_utc_date() -> String {
    utc_date_from_unix_secs(current_unix_secs())
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// UTC date as `YYYY-MM-DD` for a Unix timestamp in seconds.
#[must_use]
pub fn utc_date_from_unix_secs(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX);
    let (year, month, day) = civil_from_unix_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Civil (proleptic Gregorian) `(year, month, day)` from days since the Unix
/// epoch (1970-01-01). Howard Hinnant's `civil_from_days`. Inputs beyond the
/// representable range saturate (the epoch-shift addition and the year both
/// clamp) instead of overflowing or wrapping.
#[must_use]
pub fn civil_from_unix_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch.saturating_add(719_468);
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (
        i32::try_from(year).unwrap_or(if year.is_negative() { i32::MIN } else { i32::MAX }),
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_conversion_matches_known_fixtures() {
        // Epoch, a leap day, and an ordinary modern date (fixtures
        // cross-checked against Python's datetime).
        assert_eq!(utc_date_from_unix_secs(0), "1970-01-01");
        assert_eq!(utc_date_from_unix_secs(951_782_400), "2000-02-29");
        assert_eq!(utc_date_from_unix_secs(1_751_702_400), "2025-07-05");
        // Day boundaries: last second of a day vs first of the next.
        assert_eq!(utc_date_from_unix_secs(86_399), "1970-01-01");
        assert_eq!(utc_date_from_unix_secs(86_400), "1970-01-02");
        // Extremes saturate instead of overflowing (documented contract).
        let _ = civil_from_unix_days(i64::MAX);
        let _ = civil_from_unix_days(i64::MIN);
    }

    /// The exact frozen-constant regression guard: "today" must equal the
    /// date derived from the live clock at the moment of the call — a
    /// hardcoded literal can only pass this on the single day it names.
    #[test]
    fn current_utc_date_tracks_the_live_clock() {
        let before = utc_date_from_unix_secs(current_unix_secs());
        let now = current_utc_date();
        let after = utc_date_from_unix_secs(current_unix_secs());
        // `before`/`after` bracket the call across a possible midnight tick.
        assert!(
            now == before || now == after,
            "current_utc_date must come from the live clock: {now} vs {before}/{after}"
        );
    }

    /// Local "today" is the UTC date shifted by at most one day in either
    /// direction (UTC-12 … UTC+14), and always well-formed. A frozen literal
    /// or a broken `date` invocation cannot satisfy this against a live
    /// clock outside its own day.
    #[test]
    fn current_local_date_stays_within_one_day_of_utc() {
        let local = current_local_date();
        let secs = current_unix_secs();
        let yesterday = utc_date_from_unix_secs(secs.saturating_sub(86_400));
        let today = utc_date_from_unix_secs(secs);
        let tomorrow = utc_date_from_unix_secs(secs + 86_400);
        assert!(
            local == yesterday || local == today || local == tomorrow,
            "local date must be within one day of UTC: {local} vs {yesterday}/{today}/{tomorrow}"
        );
    }
}
