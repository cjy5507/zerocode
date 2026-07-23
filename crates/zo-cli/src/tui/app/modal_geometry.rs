//! Pure geometry helpers for centering and anchoring modal rectangles.
//!
//! All four functions are pure (no I/O, no global state) so they are
//! unit-tested in `app/tests.rs` against synthetic `Rect` inputs.
//! [`anchored_modal_rect`] reads private fields off [`super::App`],
//! which is allowed because this is a child module of `app/mod.rs`.

use ratatui::layout::Rect;

use crate::tui::layout::LayoutRegions;
use crate::tui::modals::ModalPlacement;

use super::{App, AppMode};

pub(super) fn modal_size_for_mode(app: &App, area: Rect) -> (u16, u16) {
    // An *anchored* slot modal computes its own size (already clamped); the
    // legacy per-mode match below is only for not-yet-migrated modals. Only
    // `Anchored` modals derive their rect from `anchored_modal_rect` — other
    // placements (effort banner, fullscreen viewers, palette) are sized by their
    // own `*_modal_rect` in `draw_modals`, so they fall through to the (unused)
    // default here rather than mis-anchoring above the input row.
    if let Some(modal) = &app.active_modal {
        if modal.placement() == ModalPlacement::Anchored {
            return modal.desired_size(area, &app.theme);
        }
    }
    let width = area
        .width
        .clamp(36, 64)
        .min(area.width.saturating_sub(4).max(24));

    // List modals now close with a blank spacer + key-hint footer (+2 rows on
    // top of the option rows and borders), so each arm budgets for them.
    let content_height = match app.mode {
        // The rewind confirmation card sizes to its (variable) body lines plus
        // borders; the outer clamp keeps it within bounds.
        AppMode::ModalConfirmRewind => app.rewind_confirm_lines().map_or(8, |lines| {
            u16::try_from(lines.len())
                .unwrap_or(u16::MAX)
                .saturating_add(2)
        }),
        AppMode::ModalEffort
        // ModalDiff sizes itself via `diff_modal_rect`; this is only a fallback.
        | AppMode::ModalDiff
        // ModalRewind sizes itself via `diff_modal_rect`; this is only a fallback.
        | AppMode::ModalRewind
        // ModalWorkflow/ModalAgents/ModalTeamInbox size themselves via `diff_modal_rect`;
        // this is only a fallback.
        | AppMode::ModalWorkflow
        | AppMode::ModalAgents
        | AppMode::ModalTeamInbox
        // ModalTools sizes itself via `diff_modal_rect`; this is only a fallback.
        | AppMode::ModalTools
        | AppMode::ModalHunks
        // ModalUsage and ModalSmartSettings size themselves via `diff_modal_rect`;
        // this is only a fallback.
        | AppMode::ModalUsage
        | AppMode::ModalSmartSettings
        | AppMode::ModalDeepTier
        | AppMode::ModalRemoteOnboarding
        // ModalReport sizes itself via `centered_modal_rect`; fallback only.
        | AppMode::ModalReport
        // ModalFile sizes itself via `palette_modal_rect`; this is only a fallback.
        | AppMode::ModalFile
        // These anchored modal modes live on the unified slot, sized by the
        // early-return above; these branches are unreachable but keep the match
        // exhaustive.
        | AppMode::ModalModel
        | AppMode::ModalPermissions
        | AppMode::ModalArgPick
        | AppMode::ModalSession
        | AppMode::ModalLogin
        | AppMode::ModalQuestion
        | AppMode::ModalApiKey
        | AppMode::ModalCustomProvider
        | AppMode::ModalChoice
        | AppMode::Normal
        | AppMode::Search
        | AppMode::Pager
        | AppMode::Focus => 8,
    };
    let height = content_height
        .clamp(6, 18)
        .min(area.height.saturating_sub(2).max(6));
    (width, height)
}

/// Pure geometry helper: center a modal of the given (width, height)
/// inside `area`, clamping the size to leave a small margin on all
/// sides. Production rect for [`ModalPlacement::Centered`] surfaces (the
/// report popup), which size themselves via `Modal::desired_size`.
pub(super) fn centered_modal_rect(area: Rect, size: (u16, u16)) -> Rect {
    let width = size.0.min(area.width.saturating_sub(4));
    let height = size.1.min(area.height.saturating_sub(4));
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

pub(super) fn palette_modal_rect(area: Rect) -> Rect {
    let width = (area.width * 80 / 100)
        .clamp(40, 110)
        .min(area.width.saturating_sub(2));
    let height = (area.height * 70 / 100)
        .clamp(10, 30)
        .min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

/// Geometry for the `/effort` slider: a wide, short banner centered
/// inside the *chat* column so it never paints over the sidebar /
/// HUD ledger on the right. Needs more width than a list modal so the
/// six-stop gradient bar and its labels fit without truncation; falls
/// back to the full `area` when no transcript region is available
/// (e.g. before the first layout pass).
pub(super) fn effort_modal_rect(regions: &LayoutRegions, area: Rect) -> Rect {
    // Confine to the transcript column (sidebar-excluded) when known.
    let host = if regions.transcript.width > 0 {
        regions.transcript
    } else {
        area
    };
    let width = (host.width * 92 / 100)
        .clamp(44, 100)
        .min(host.width.saturating_sub(2));
    // 8 content rows + 2 border + 2 vertical padding = 12.
    let height = 12u16.min(host.height.saturating_sub(2).max(9));
    let x = host.x + host.width.saturating_sub(width) / 2;
    // Vertically center within the chat column, biased slightly upward
    // so the slider sits above the input box rather than over it.
    let y = host.y + host.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

/// Near-full-screen rect for the interactive `/diff` viewer, confined to
/// the transcript column so it never spills under the sidebar.
pub(super) fn diff_modal_rect(regions: &LayoutRegions, area: Rect) -> Rect {
    let host = if regions.transcript.width > 0 {
        regions.transcript
    } else {
        area
    };
    let width = (host.width * 96 / 100)
        .clamp(40, host.width.saturating_sub(2).max(40))
        .min(host.width);
    let height = (host.height * 90 / 100)
        .clamp(8, host.height.saturating_sub(2).max(8))
        .min(host.height);
    let x = host.x + host.width.saturating_sub(width) / 2;
    let y = host.y + host.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

/// Point-in-rect test. Mouse hit-testing now compares columns/rows inline
/// against the full-height sidebar rect, so this helper is only used by the
/// modal-geometry unit tests — gated to `cfg(test)` to avoid a dead-code
/// warning in release builds.
#[cfg(test)]
pub(super) fn point_in_rect(column: u16, row: u16, rect: Rect) -> bool {
    column >= rect.x
        && column < rect.x.saturating_add(rect.width)
        && row >= rect.y
        && row < rect.y.saturating_add(rect.height)
}

pub(super) fn anchored_modal_rect(app: &App, regions: &LayoutRegions, area: Rect) -> Rect {
    // Modals are horizontally centered against the full terminal `area`
    // and anchored to sit just above the input row. Falls back to a true
    // screen center when the input row is unavailable (e.g. startup
    // splash before any layout has been computed).
    let (width, height) = modal_size_for_mode(app, area);
    let width = width.min(area.width.saturating_sub(4));
    let height = height.min(area.height.saturating_sub(4));
    let input = regions.input;
    let x = if input.width == 0 {
        area.x + 2
    } else {
        input.x
    };
    let y = if input.height == 0 || input.y <= area.y {
        area.y + area.height.saturating_sub(height) / 2
    } else {
        // One-row gap above the input box; clamp inside `area`.
        let gap: u16 = 1;
        let desired = input.y.saturating_sub(height).saturating_sub(gap);
        desired.max(area.y + 1)
    };
    Rect::new(x, y, width, height)
}
