//! Priority-budget segment allocator for the responsive sidebar / HUD rows.
//!
//! The old layout sized each field with an ad-hoc `width.saturating_sub(magic)`
//! budget and then hard-truncated, which produced mid-word chops like
//! `agent-workflow-too…` / `waiting for ap…` on a narrow terminal — every field
//! got *some* space, so all of them ended up mangled. This allocator instead
//! decides, by priority, *which* segments get to appear at all: a low-priority
//! segment that cannot fit its `min` is dropped whole (a clean omission), and the
//! cells it would have wasted go to the segments that survive. Live signal (what
//! an agent is doing) outranks static text (its model id), so truncation eats the
//! least-informative field first.
//!
//! Pure cell arithmetic — no string or style handling — so it is trivially unit
//! testable and reused identically by the sidebar fleet rows and the HUD.

/// One candidate segment competing for horizontal space, measured in terminal
/// cells (CJK = 2, via the caller's width fn). A segment is shown in full when
/// space allows, shrunk toward `min` (and ellipsized by the caller) when tight,
/// and dropped whole when it cannot even reach `min`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Seg {
    /// Natural display width if shown in full.
    pub width: usize,
    /// Floor below which the segment is dropped entirely rather than shown
    /// mangled — this is what turns a tight row into clean omissions instead of
    /// mid-word chops.
    pub min: usize,
    /// Higher priority is granted space first (ties broken by original order).
    pub prio: u8,
}

impl Seg {
    /// A flexible segment shown in `min..=width` cells (or dropped below `min`).
    pub fn flex(width: usize, min: usize, prio: u8) -> Self {
        Self {
            width,
            min: min.min(width),
            prio,
        }
    }
}

/// Allocate `budget` cells across `segs`, charging `sep` cells between each pair
/// of *surviving* segments. Returns, per input segment (in original order), the
/// granted width in cells, or `None` if the segment was dropped.
///
/// Guarantee: `Σ granted + sep·(survivors − 1) ≤ budget`, so a caller that
/// truncates each surviving segment to its granted width and joins with a
/// `sep`-wide separator never overflows the row.
pub(super) fn allocate(budget: usize, segs: &[Seg], sep: usize) -> Vec<Option<usize>> {
    let mut grant: Vec<Option<usize>> = vec![None; segs.len()];

    // Pass 1 — admit segments by priority (desc), stable on original index.
    // Each admitted segment is granted its floor; admission charges a separator
    // once a prior segment already survived.
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by(|&a, &b| segs[b].prio.cmp(&segs[a].prio).then(a.cmp(&b)));
    let mut used = 0usize;
    let mut survivors = 0usize;
    for &i in &order {
        let need = segs[i].min + usize::from(survivors > 0) * sep;
        if used + need <= budget {
            grant[i] = Some(segs[i].min);
            used += need;
            survivors += 1;
        }
    }

    // Pass 2 — grow survivors toward their natural width, in original order (so
    // the leftmost / primary fields fill out first).
    let mut leftover = budget.saturating_sub(used);
    for (i, slot) in grant.iter_mut().enumerate() {
        if leftover == 0 {
            break;
        }
        if let Some(w) = slot {
            if segs[i].width > *w {
                let grow = (segs[i].width - *w).min(leftover);
                *w += grow;
                leftover -= grow;
            }
        }
    }

    grant
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(grant: &[Option<usize>], sep: usize) -> usize {
        let survivors = grant.iter().filter(|g| g.is_some()).count();
        grant.iter().flatten().sum::<usize>() + sep * survivors.saturating_sub(1)
    }

    #[test]
    fn everything_fits_when_budget_is_ample() {
        let segs = [Seg::flex(10, 4, 3), Seg::flex(8, 4, 2), Seg::flex(6, 6, 1)];
        let grant = allocate(100, &segs, 2);
        assert_eq!(grant, vec![Some(10), Some(8), Some(6)]);
        assert!(total(&grant, 2) <= 100);
    }

    #[test]
    fn never_exceeds_budget() {
        let segs = [Seg::flex(20, 5, 3), Seg::flex(20, 5, 2), Seg::flex(20, 5, 1)];
        for budget in 0..60 {
            let grant = allocate(budget, &segs, 2);
            assert!(
                total(&grant, 2) <= budget,
                "overflow at budget={budget}: {grant:?}"
            );
        }
    }

    #[test]
    fn drops_lowest_priority_whole_not_mangled() {
        // Budget fits the two high-prio mins + sep, but not the third.
        let segs = [
            Seg::flex(20, 6, 3), // highest
            Seg::flex(20, 6, 2),
            Seg::flex(20, 6, 1), // lowest — should drop whole
        ];
        let grant = allocate(14, &segs, 2); // 6 + 2 + 6 = 14, no room for a 3rd
        assert!(grant[0].is_some());
        assert!(grant[1].is_some());
        assert_eq!(grant[2], None, "lowest priority must drop, not shrink to noise");
    }

    #[test]
    fn all_or_nothing_segment_drops_when_below_min() {
        // A segment whose min == width behaves all-or-nothing.
        let segs = [Seg::flex(10, 4, 1), Seg::flex(8, 8, 2)];
        let grant = allocate(14, &segs, 2); // 8 + 2 + 4 = 14
        assert_eq!(grant[1], Some(8), "min==width segment kept at full width");
        assert_eq!(grant[0], Some(4), "the other shrinks to its min");
        // Too tight for the rigid-like segment → it drops, the flex takes the room.
        let grant = allocate(5, &segs, 2);
        assert_eq!(grant[1], None, "min==width segment drops whole when it cannot fit");
        assert_eq!(grant[0], Some(5), "flex grows into the freed space up to width");
    }

    #[test]
    fn higher_priority_survives_when_only_one_fits() {
        let segs = [Seg::flex(10, 5, 1), Seg::flex(10, 5, 9)];
        let grant = allocate(6, &segs, 2); // only one min(5) fits
        assert_eq!(grant[0], None);
        assert!(grant[1].is_some(), "the priority-9 segment must win the single slot");
    }

    #[test]
    fn leftover_grows_primary_first() {
        let segs = [Seg::flex(10, 3, 2), Seg::flex(10, 3, 1)];
        let grant = allocate(15, &segs, 1); // mins 3+3+sep1=7, leftover 8
        // Original-order growth: seg0 fills to 10 first (uses 7), seg1 gets the
        // remaining 1 on top of its min.
        assert_eq!(grant[0], Some(10));
        assert_eq!(grant[1], Some(4));
        assert!(total(&grant, 1) <= 15);
    }
}
