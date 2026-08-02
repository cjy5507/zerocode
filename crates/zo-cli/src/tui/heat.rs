//! Pure Cold Steel / Hot Core state derivation.

use std::time::{Duration, Instant};

/// Duration of the post-turn cooling animation.
pub const COOLING_SECS: f32 = 3.0;

/// Duration of the full-width turn-start ignition wave.
pub const IGNITION_MS: u64 = 500;

/// Chrome temperature derived from the existing turn lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatState {
    /// No turn is active and the cooling window has elapsed.
    Cold,
    /// A turn is active.
    Hot,
    /// A turn ended recently; `ramp_idx` selects the precomputed color step.
    Cooling {
        /// Index into [`crate::tui::theme::HeatTokens::ramp`].
        ramp_idx: usize,
    },
}

impl HeatState {
    /// Derive temperature from the canonical turn-active flag and completion time.
    #[must_use]
    pub fn derive(turn_active: bool, cooled_since: Option<Instant>, now: Instant) -> Self {
        if turn_active {
            return Self::Hot;
        }
        let Some(cooled_since) = cooled_since else {
            return Self::Cold;
        };
        let elapsed = now.checked_duration_since(cooled_since).unwrap_or_default();
        if elapsed.as_secs_f64() >= f64::from(COOLING_SECS) {
            Self::Cold
        } else {
            Self::Cooling {
                ramp_idx: cooling_ramp_idx(elapsed),
            }
        }
    }

    /// Whether the cooling animation still needs render ticks.
    #[must_use]
    pub const fn is_cooling(self) -> bool {
        matches!(self, Self::Cooling { .. })
    }
}

/// Select one of eight cooling colors for `elapsed`, clamped to `0..=7`.
#[must_use]
pub fn cooling_ramp_idx(elapsed: Duration) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = ((elapsed.as_secs_f32() / COOLING_SECS) * 8.0).floor() as usize;
    index.min(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_covers_hot_cooling_cold_and_reheat_transitions() {
        let ended = Instant::now();

        assert_eq!(HeatState::derive(true, None, ended), HeatState::Hot);
        assert_eq!(
            HeatState::derive(false, Some(ended), ended),
            HeatState::Cooling { ramp_idx: 0 }
        );
        assert_eq!(
            HeatState::derive(
                false,
                Some(ended),
                (ended + Duration::from_secs(3))
                    .checked_sub(Duration::from_nanos(1))
                    .expect("three seconds exceeds one nanosecond")
            ),
            HeatState::Cooling { ramp_idx: 7 }
        );
        assert_eq!(
            HeatState::derive(false, Some(ended), ended + Duration::from_secs(3)),
            HeatState::Cold
        );
        assert_eq!(
            HeatState::derive(
                true,
                Some(ended),
                ended + Duration::from_millis(1_500)
            ),
            HeatState::Hot,
            "a new turn wins immediately over an in-progress cooling ramp"
        );
    }

    #[test]
    fn cooling_ramp_index_clamps_all_boundaries() {
        assert_eq!(cooling_ramp_idx(Duration::ZERO), 0);
        assert_eq!(cooling_ramp_idx(Duration::from_millis(2_999)), 7);
        assert_eq!(cooling_ramp_idx(Duration::from_secs(3)), 7);
        assert_eq!(cooling_ramp_idx(Duration::from_secs(30)), 7);
    }
}
