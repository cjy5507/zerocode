use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, mpsc, oneshot};

use super::protocol::{
    ControlRole, FrameRecord, MAX_COMMAND_ID_BYTES, MAX_DEVICE_NAME_CHARS, PromptMode,
    ServerMessage, SessionInfo, ToolApprovalDecision, ToolApprovalRequest, ToolApprovalSource,
    TurnPhase,
};
use super::push::{PushReason, PushService, SendOutcome, ValidatedSubscription};

const OFFER_TTL: Duration = Duration::from_secs(90);
const APPROVAL_PUSH_DELAY: Duration = Duration::from_secs(30);
const PUSH_MIN_GAP: Duration = Duration::from_secs(15);
const APPROVED_POLL_TTL: Duration = Duration::from_secs(2 * 60);
const DEVICE_TTL: Duration = Duration::from_secs(8 * 60 * 60);
const MAX_DEVICES: usize = 4;
const MAX_PENDING: usize = 4;
const MAX_FRAMES: usize = 512;
const MAX_SEEN_COMMANDS: usize = 1_024;
const MAX_RESOLVED_APPROVALS: usize = 256;
const EVENT_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
pub(crate) struct PairingNotice {
    pub(crate) device_name: String,
    pub(crate) comparison_code: String,
    pub(crate) expires_in_seconds: u64,
}

#[derive(Debug, Clone)]
pub(crate) enum RemoteEffect {
    Prompt {
        text: String,
        mode: PromptMode,
        turn_generation: Option<u64>,
    },
    Cancel {
        turn_generation: u64,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedDevice {
    pub(crate) id: String,
    pub(crate) expires_at: Instant,
}

#[derive(Debug, Clone)]
pub(crate) struct PairingStarted {
    pub(crate) id: String,
    pub(crate) comparison_code: String,
    pub(crate) expires_in_seconds: u64,
    pub(crate) poll_expires_in_seconds: u64,
}

#[derive(Debug, Clone)]
pub(crate) enum PairPoll {
    Pending,
    Approved {
        token: String,
        role: ControlRole,
    },
    Denied,
    Expired,
}

#[derive(Debug, Clone)]
pub(crate) struct ApprovalResult {
    pub(crate) device_name: String,
    pub(crate) role: ControlRole,
}

#[derive(Debug, Clone)]
pub(crate) struct PendingPairOverview {
    pub(crate) device_name: String,
    pub(crate) comparison_code: String,
}

#[derive(Debug, Clone)]
pub(crate) struct RemoteStatus {
    pub(crate) devices: usize,
    pub(crate) pending: usize,
    pub(crate) pending_pairs: Vec<PendingPairOverview>,
    pub(crate) turn: TurnPhase,
    pub(crate) controller_name: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum SnapshotPlan {
    Full {
        frames: Vec<FrameRecord>,
        next_seq: u64,
    },
    Replay {
        frames: Vec<FrameRecord>,
        next_seq: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandDecision {
    New,
    Duplicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolApprovalResolution {
    pub(crate) decision: ToolApprovalDecision,
    pub(crate) source: ToolApprovalSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolApprovalAttempt {
    Resolved,
    AlreadyResolved(ToolApprovalResolution),
    InvalidChoice,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ControllerGrace {
    device_id: String,
    generation: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum PairingError {
    #[error("the pairing offer expired; run /remote qr")]
    Expired,
    #[error("the pairing offer is invalid or has already been used")]
    InvalidOffer,
    #[error("too many pending pairing requests")]
    TooManyPending,
    #[error("the device name is empty or too long")]
    InvalidDeviceName,
    #[error("no pending device matches comparison code {0}")]
    UnknownCode(String),
    #[error("the remote device limit has been reached")]
    DeviceLimit,
}

#[derive(Clone)]
pub(crate) struct RemoteShared {
    inner: Arc<Mutex<State>>,
    events: broadcast::Sender<ServerMessage>,
    prompt_effects: mpsc::Sender<RemoteEffect>,
    control_effects: mpsc::Sender<RemoteEffect>,
    pairing_notices: mpsc::Sender<PairingNotice>,
    push: PushService,
}

struct State {
    session: SessionInfo,
    offer: Option<PairOffer>,
    pending: HashMap<String, PendingPair>,
    devices: HashMap<[u8; 32], Device>,
    controller_id: Option<String>,
    live_websockets: HashMap<String, usize>,
    controller_grace: Option<ControllerGrace>,
    next_controller_grace: u64,
    frames: VecDeque<FrameRecord>,
    next_seq: u64,
    snapshot_floor: u64,
    turn: TurnPhase,
    turn_generation: u64,
    seen_commands: HashSet<(String, String)>,
    seen_order: VecDeque<(String, String)>,
    pending_approvals: HashMap<String, PendingToolApproval>,
    resolved_approvals: HashMap<String, ToolApprovalResolution>,
    resolved_approval_order: VecDeque<String>,
    push_subscriptions: HashMap<String, ValidatedSubscription>,
    push_last_sent: HashMap<String, Instant>,
    push_sent_approvals: HashSet<String>,
    push_scheduled_approvals: HashSet<String>,
}

struct PushTarget {
    device_id: String,
    endpoint_key: String,
    subscription: ValidatedSubscription,
}

struct PairOffer {
    secret_hash: [u8; 32],
    expires_at: Instant,
    claimed: bool,
}

struct PendingPair {
    device_name: String,
    comparison_code: String,
    expires_at: Instant,
    decision: PairDecision,
}

enum PairDecision {
    Waiting,
    Approved {
        token: String,
        role: ControlRole,
    },
    Denied,
}

struct Device {
    id: String,
    name: String,
    expires_at: Instant,
}

struct PendingToolApproval {
    request: ToolApprovalRequest,
    responder: oneshot::Sender<ToolApprovalDecision>,
}

impl RemoteShared {
    pub(crate) fn new(
        session_id: String,
        title: String,
        prompt_effects: mpsc::Sender<RemoteEffect>,
        control_effects: mpsc::Sender<RemoteEffect>,
        pairing_notices: mpsc::Sender<PairingNotice>,
    ) -> Self {
        Self::new_with_push(
            session_id,
            title,
            prompt_effects,
            control_effects,
            pairing_notices,
            PushService::from_env(),
        )
    }

    pub(crate) fn new_with_push(
        session_id: String,
        title: String,
        prompt_effects: mpsc::Sender<RemoteEffect>,
        control_effects: mpsc::Sender<RemoteEffect>,
        pairing_notices: mpsc::Sender<PairingNotice>,
        push: PushService,
    ) -> Self {
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        Self {
            inner: Arc::new(Mutex::new(State {
                session: SessionInfo {
                    id: session_id,
                    title,
                },
                offer: None,
                pending: HashMap::new(),
                devices: HashMap::new(),
                controller_id: None,
                live_websockets: HashMap::new(),
                controller_grace: None,
                next_controller_grace: 0,
                frames: VecDeque::new(),
                next_seq: 1,
                snapshot_floor: 1,
                turn: TurnPhase::Idle,
                turn_generation: 0,
                seen_commands: HashSet::new(),
                seen_order: VecDeque::new(),
                pending_approvals: HashMap::new(),
                resolved_approvals: HashMap::new(),
                resolved_approval_order: VecDeque::new(),
                push_subscriptions: HashMap::new(),
                push_last_sent: HashMap::new(),
                push_sent_approvals: HashSet::new(),
                push_scheduled_approvals: HashSet::new(),
            })),
            events,
            prompt_effects,
            control_effects,
            pairing_notices,
            push,
        }
    }

    pub(crate) fn refresh_offer(&self, secret: &str) {
        let mut state = self.lock();
        state.offer = Some(PairOffer {
            secret_hash: digest(secret),
            expires_at: Instant::now() + OFFER_TTL,
            claimed: false,
        });
    }

    pub(crate) fn offer_available(&self, secret: &str) -> bool {
        let now = Instant::now();
        let state = self.lock();
        matches!(
            &state.offer,
            Some(offer)
                if !offer.claimed
                    && now < offer.expires_at
                    && constant_time_eq(&offer.secret_hash, &digest(secret))
        )
    }

    pub(crate) fn rotate(&self, secret: &str) {
        let mut state = self.lock();
        state.offer = Some(PairOffer {
            secret_hash: digest(secret),
            expires_at: Instant::now() + OFFER_TTL,
            claimed: false,
        });
        state.pending.clear();
        state.devices.clear();
        state.controller_id = None;
        state.live_websockets.clear();
        state.controller_grace = None;
        state.seen_commands.clear();
        state.seen_order.clear();
        state.push_subscriptions.clear();
        state.push_last_sent.clear();
        state.push_scheduled_approvals.clear();
        drop(state);
        let _ = self.events.send(ServerMessage::Error {
            code: "session_revoked",
            message: "Remote credentials were rotated.".to_string(),
            recoverable: false,
        });
    }

    pub(crate) fn revoke_all(&self) {
        let _ = self.events.send(ServerMessage::Error {
            code: "session_revoked",
            message: "Remote access was stopped.".to_string(),
            recoverable: false,
        });
        let mut state = self.lock();
        state.offer = None;
        state.pending.clear();
        state.devices.clear();
        state.controller_id = None;
        state.live_websockets.clear();
        state.controller_grace = None;
        state.seen_commands.clear();
        state.seen_order.clear();
        state.push_subscriptions.clear();
        state.push_last_sent.clear();
        state.push_scheduled_approvals.clear();
    }

    pub(crate) fn begin_pairing(
        &self,
        secret: &str,
        device_name: &str,
    ) -> Result<PairingStarted, PairingError> {
        let device_name = normalized_device_name(device_name)?;
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        if state.pending.len() >= MAX_PENDING {
            return Err(PairingError::TooManyPending);
        }
        let Some(offer) = state.offer.as_mut() else {
            return Err(PairingError::InvalidOffer);
        };
        if now >= offer.expires_at {
            return Err(PairingError::Expired);
        }
        if offer.claimed || !constant_time_eq(&offer.secret_hash, &digest(secret)) {
            return Err(PairingError::InvalidOffer);
        }
        offer.claimed = true;

        let id = random_token(12);
        let comparison_code = comparison_code();
        state.pending.insert(
            id.clone(),
            PendingPair {
                device_name: device_name.clone(),
                comparison_code: comparison_code.clone(),
                expires_at: now + OFFER_TTL,
                decision: PairDecision::Waiting,
            },
        );
        drop(state);
        let _ = self.pairing_notices.try_send(PairingNotice {
            device_name,
            comparison_code: comparison_code.clone(),
            expires_in_seconds: OFFER_TTL.as_secs(),
        });
        Ok(PairingStarted {
            id,
            comparison_code,
            expires_in_seconds: OFFER_TTL.as_secs(),
            poll_expires_in_seconds: OFFER_TTL.as_secs() + APPROVED_POLL_TTL.as_secs(),
        })
    }

    pub(crate) fn pairing_status(&self, id: &str) -> PairPoll {
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        let Some(pending) = state.pending.get(id) else {
            return PairPoll::Expired;
        };
        match &pending.decision {
            PairDecision::Waiting => PairPoll::Pending,
            PairDecision::Approved { token, role } => PairPoll::Approved {
                token: token.clone(),
                role: *role,
            },
            PairDecision::Denied => PairPoll::Denied,
        }
    }

    pub(crate) fn approve(&self, code: &str) -> Result<ApprovalResult, PairingError> {
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        if state.devices.len() >= MAX_DEVICES {
            return Err(PairingError::DeviceLimit);
        }
        let pending_id = state
            .pending
            .iter()
            .find(|(_, pending)| {
                matches!(pending.decision, PairDecision::Waiting)
                    && pending.comparison_code.eq_ignore_ascii_case(code)
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| PairingError::UnknownCode(code.to_string()))?;

        let token = random_token(32);
        let token_hash = digest(&token);
        let device_id = random_token(12);
        let controller_changed = state.controller_id.is_none();
        let role = if controller_changed {
            state.controller_id = Some(device_id.clone());
            ControlRole::Controller
        } else {
            ControlRole::Observer
        };
        let device_name = state
            .pending
            .get(&pending_id)
            .map(|pending| pending.device_name.clone())
            .expect("pending pair exists");
        state.devices.insert(
            token_hash,
            Device {
                id: device_id,
                name: device_name.clone(),
                expires_at: now + DEVICE_TTL,
            },
        );
        if let Some(pending) = state.pending.get_mut(&pending_id) {
            pending.decision = PairDecision::Approved { token, role };
            pending.expires_at = now + APPROVED_POLL_TTL;
        }
        drop(state);
        if controller_changed {
            self.broadcast_control_state(true);
        }
        Ok(ApprovalResult { device_name, role })
    }

    pub(crate) fn deny(&self, code: &str) -> Result<String, PairingError> {
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        let pending = state
            .pending
            .values_mut()
            .find(|pending| {
                matches!(pending.decision, PairDecision::Waiting)
                    && pending.comparison_code.eq_ignore_ascii_case(code)
            })
            .ok_or_else(|| PairingError::UnknownCode(code.to_string()))?;
        pending.decision = PairDecision::Denied;
        Ok(pending.device_name.clone())
    }

    pub(crate) fn authenticate(&self, token: &str) -> Option<AuthenticatedDevice> {
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        let device = state.devices.get(&digest(token))?;
        Some(AuthenticatedDevice {
            id: device.id.clone(),
            expires_at: device.expires_at,
        })
    }

    pub(crate) fn request_control(&self, device_id: &str) -> ControlRole {
        let mut state = self.lock();
        let known = state.devices.values().any(|device| device.id == device_id);
        let controller_changed = known && state.controller_id.is_none();
        if controller_changed {
            state.controller_id = Some(device_id.to_string());
        }
        let role = role_for(&state, device_id);
        drop(state);
        if controller_changed {
            self.broadcast_control_state(true);
        }
        role
    }

    pub(crate) fn websocket_connected(&self, device_id: &str) -> bool {
        let mut state = self.lock();
        if !state.devices.values().any(|device| device.id == device_id) {
            return false;
        }
        *state
            .live_websockets
            .entry(device_id.to_string())
            .or_default() += 1;
        if state
            .controller_grace
            .as_ref()
            .is_some_and(|grace| grace.device_id == device_id)
        {
            state.controller_grace = None;
        }
        true
    }

    pub(crate) fn websocket_disconnected(
        &self,
        device_id: &str,
    ) -> Option<ControllerGrace> {
        let mut state = self.lock();
        let count = state.live_websockets.get_mut(device_id)?;
        *count = count.saturating_sub(1);
        if *count > 0 {
            return None;
        }
        state.live_websockets.remove(device_id);
        if state.controller_id.as_deref() != Some(device_id) {
            return None;
        }
        state.next_controller_grace = state.next_controller_grace.wrapping_add(1);
        let grace = ControllerGrace {
            device_id: device_id.to_string(),
            generation: state.next_controller_grace,
        };
        state.controller_grace = Some(grace.clone());
        Some(grace)
    }

    pub(crate) fn expire_controller_grace(&self, grace: &ControllerGrace) -> bool {
        let mut state = self.lock();
        if state.controller_grace.as_ref() != Some(grace)
            || state.controller_id.as_deref() != Some(grace.device_id.as_str())
            || state
                .live_websockets
                .get(&grace.device_id)
                .is_some_and(|count| *count > 0)
        {
            return false;
        }
        state.controller_id = None;
        state.controller_grace = None;
        drop(state);
        self.broadcast_control_state(false);
        true
    }

    pub(crate) fn control_state_for(&self, device_id: &str) -> (bool, ControlRole) {
        let state = self.lock();
        (state.controller_id.is_some(), role_for(&state, device_id))
    }

    pub(crate) fn begin_command(
        &self,
        device_id: &str,
        command_id: &str,
    ) -> Result<CommandDecision, &'static str> {
        if command_id.is_empty() || command_id.len() > MAX_COMMAND_ID_BYTES {
            return Err("invalid_command_id");
        }
        let mut state = self.lock();
        let key = (device_id.to_string(), command_id.to_string());
        if state.seen_commands.contains(&key) {
            return Ok(CommandDecision::Duplicate);
        }
        state.seen_commands.insert(key.clone());
        state.seen_order.push_back(key);
        while state.seen_order.len() > MAX_SEEN_COMMANDS {
            if let Some(oldest) = state.seen_order.pop_front() {
                state.seen_commands.remove(&oldest);
            }
        }
        Ok(CommandDecision::New)
    }

    pub(crate) fn forget_command(&self, device_id: &str, command_id: &str) {
        let mut state = self.lock();
        let key = (device_id.to_string(), command_id.to_string());
        state.seen_commands.remove(&key);
        state.seen_order.retain(|seen| seen != &key);
    }

    pub(crate) fn is_device_active(&self, device_id: &str) -> bool {
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        state.devices.values().any(|device| device.id == device_id)
    }

    pub(crate) fn next_seq(&self) -> u64 {
        self.lock().next_seq
    }

    pub(crate) fn turn(&self) -> TurnPhase {
        self.lock().turn
    }

    pub(crate) fn turn_state(&self) -> (TurnPhase, u64) {
        let state = self.lock();
        (state.turn, state.turn_generation)
    }

    pub(crate) fn set_turn(&self, turn: TurnPhase, turn_generation: u64) {
        let mut state = self.lock();
        if state.turn == turn && state.turn_generation == turn_generation {
            return;
        }
        state.turn = turn;
        state.turn_generation = turn_generation;
        drop(state);
        let _ = self.events.send(ServerMessage::TurnState { turn });
    }

    pub(crate) fn replace_snapshot(&self, blocks: impl IntoIterator<Item = Value>) {
        let mut tail = VecDeque::with_capacity(MAX_FRAMES);
        for block in blocks.into_iter().filter(should_project_block) {
            if tail.len() == MAX_FRAMES {
                tail.pop_front();
            }
            tail.push_back(block);
        }
        let next_seq = {
            let mut state = self.lock();
            state.frames.clear();
            state.snapshot_floor = state.next_seq;
            state.next_seq = state.next_seq.saturating_add(1);
            for block in tail {
                let frame = FrameRecord {
                    seq: state.next_seq,
                    block,
                };
                state.next_seq = state.next_seq.saturating_add(1);
                state.frames.push_back(frame);
            }
            state.next_seq
        };
        let _ = self.events.send(ServerMessage::ResyncRequired { next_seq });
    }

    pub(crate) fn record_frame(&self, block: Value) {
        if !should_project_block(&block) {
            return;
        }
        let frame = {
            let mut state = self.lock();
            let frame = FrameRecord {
                seq: state.next_seq,
                block,
            };
            state.next_seq = state.next_seq.saturating_add(1);
            state.frames.push_back(frame.clone());
            while state.frames.len() > MAX_FRAMES {
                state.frames.pop_front();
            }
            frame
        };
        let _ = self.events.send(ServerMessage::Frame { frame });
    }

    pub(crate) fn snapshot_for(&self, last_seq: u64) -> SnapshotPlan {
        let state = self.lock();
        let next_seq = state.next_seq;
        let earliest = state.frames.front().map_or(next_seq, |frame| frame.seq);
        let can_replay_current_snapshot = last_seq >= state.snapshot_floor
            && last_seq < next_seq
            && last_seq.saturating_add(1) >= earliest;
        if can_replay_current_snapshot {
            SnapshotPlan::Replay {
                frames: state
                    .frames
                    .iter()
                    .filter(|frame| frame.seq > last_seq)
                    .cloned()
                    .collect(),
                next_seq,
            }
        } else {
            SnapshotPlan::Full {
                frames: state.frames.iter().cloned().collect(),
                next_seq,
            }
        }
    }

    pub(crate) fn session_info(&self) -> SessionInfo {
        self.lock().session.clone()
    }

    pub(crate) fn set_session_info(&self, session_id: &str, title: &str) {
        let mut state = self.lock();
        if state.session.id == session_id && state.session.title == title {
            return;
        }
        state.session = SessionInfo {
            id: session_id.to_string(),
            title: title.to_string(),
        };
    }

    pub(crate) fn role(&self, device_id: &str) -> ControlRole {
        role_for(&self.lock(), device_id)
    }

    pub(crate) fn events(&self) -> broadcast::Receiver<ServerMessage> {
        self.events.subscribe()
    }

    pub(crate) fn push_enabled(&self) -> bool {
        self.push.is_enabled()
    }

    pub(crate) fn push_server_key(&self) -> Option<String> {
        self.push.server_key().map(str::to_string)
    }

    pub(crate) fn replace_push_subscription(
        &self,
        device_id: &str,
        subscription: ValidatedSubscription,
    ) -> bool {
        let mut state = self.lock();
        cleanup_expired(&mut state, Instant::now());
        if !state.devices.values().any(|device| device.id == device_id) {
            return false;
        }
        state
            .push_subscriptions
            .insert(device_id.to_string(), subscription);
        prune_push_rate_limits(&mut state);
        true
    }

    pub(crate) fn remove_push_subscription(&self, device_id: &str) {
        let mut state = self.lock();
        state.push_subscriptions.remove(device_id);
        prune_push_rate_limits(&mut state);
    }

    pub(crate) fn push_turn_idle_if_disconnected(&self) {
        if !self.push.is_enabled() {
            return;
        }
        let targets = {
            let mut state = self.lock();
            if has_open_websockets(&state) {
                return;
            }
            collect_push_targets(&mut state, Instant::now(), false)
        };
        self.spawn_pushes(targets, PushReason::TurnIdle);
    }

    fn push_approval_now(&self, request_id: &str) {
        let targets = self.prepare_approval_push(request_id, false);
        self.spawn_pushes(targets, PushReason::Approval);
    }

    fn arm_approval_push(&self, request_id: String) {
        let should_arm = self
            .lock()
            .push_scheduled_approvals
            .insert(request_id.clone());
        if !should_arm {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            self.lock().push_scheduled_approvals.remove(&request_id);
            return;
        };
        let shared = self.clone();
        runtime.spawn(async move {
            tokio::time::sleep(APPROVAL_PUSH_DELAY).await;
            let targets = shared.prepare_approval_push(&request_id, true);
            shared.spawn_pushes(targets, PushReason::Approval);
        });
    }

    fn prepare_approval_push(
        &self,
        request_id: &str,
        require_scheduled: bool,
    ) -> Vec<PushTarget> {
        let mut state = self.lock();
        if require_scheduled && !state.push_scheduled_approvals.remove(request_id) {
            return Vec::new();
        }
        if !state.pending_approvals.contains_key(request_id)
            || !state.push_sent_approvals.insert(request_id.to_string())
        {
            return Vec::new();
        }
        collect_push_targets(&mut state, Instant::now(), true)
    }

    fn spawn_pushes(&self, targets: Vec<PushTarget>, reason: PushReason) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        for target in targets {
            let shared = self.clone();
            let push = self.push.clone();
            runtime.spawn(async move {
                if push.send(&target.subscription, reason).await == SendOutcome::Gone {
                    shared.drop_push_subscription(&target.device_id, &target.endpoint_key);
                }
            });
        }
    }

    fn drop_push_subscription(&self, device_id: &str, endpoint_key: &str) {
        let mut state = self.lock();
        let still_matches = state
            .push_subscriptions
            .get(device_id)
            .is_some_and(|subscription| subscription.endpoint_key() == endpoint_key);
        if still_matches {
            state.push_subscriptions.remove(device_id);
            prune_push_rate_limits(&mut state);
        }
    }

    pub(crate) fn publish_tool_approval(
        &self,
        mut approval: ToolApprovalRequest,
        responder: oneshot::Sender<ToolApprovalDecision>,
    ) -> String {
        let (request_id, has_clients) = {
            let mut state = self.lock();
            let request_id = loop {
                let candidate = random_token(18);
                if !state.pending_approvals.contains_key(&candidate)
                    && !state.resolved_approvals.contains_key(&candidate)
                {
                    break candidate;
                }
            };
            approval.request_id.clone_from(&request_id);
            state.pending_approvals.insert(
                request_id.clone(),
                PendingToolApproval {
                    request: approval.clone(),
                    responder,
                },
            );
            let has_clients = has_open_websockets(&state);
            (request_id, has_clients)
        };
        let _ = self
            .events
            .send(ServerMessage::ToolApprovalRequest { approval });
        if self.push.is_enabled() {
            if has_clients {
                self.arm_approval_push(request_id.clone());
            } else {
                self.push_approval_now(&request_id);
            }
        }
        request_id
    }

    pub(crate) fn pending_tool_approvals(&self) -> Vec<ToolApprovalRequest> {
        let mut approvals = self
            .lock()
            .pending_approvals
            .values()
            .map(|pending| pending.request.clone())
            .collect::<Vec<_>>();
        approvals.sort_by(|left, right| left.request_id.cmp(&right.request_id));
        approvals
    }

    pub(crate) fn resolve_tool_approval(
        &self,
        request_id: &str,
        decision: ToolApprovalDecision,
        source: ToolApprovalSource,
    ) -> ToolApprovalAttempt {
        let (pending, resolution) = {
            let mut state = self.lock();
            if let Some(resolution) = state.resolved_approvals.get(request_id).copied() {
                return ToolApprovalAttempt::AlreadyResolved(resolution);
            }
            let Some(pending) = state.pending_approvals.get(request_id) else {
                return ToolApprovalAttempt::Unknown;
            };
            if !pending
                .request
                .choices
                .iter()
                .any(|choice| choice.decision == decision)
            {
                return ToolApprovalAttempt::InvalidChoice;
            }
            let pending = state
                .pending_approvals
                .remove(request_id)
                .expect("pending approval was checked above");
            let resolution = ToolApprovalResolution { decision, source };
            state
                .resolved_approvals
                .insert(request_id.to_string(), resolution);
            state
                .resolved_approval_order
                .push_back(request_id.to_string());
            while state.resolved_approval_order.len() > MAX_RESOLVED_APPROVALS {
                if let Some(oldest) = state.resolved_approval_order.pop_front() {
                    state.resolved_approvals.remove(&oldest);
                }
            }
            (pending, resolution)
        };

        if source == ToolApprovalSource::Remote {
            let _ = pending.responder.send(decision);
        }
        let _ = self.events.send(ServerMessage::ToolApprovalResolved {
            request_id: request_id.to_string(),
            decision: resolution.decision,
            source: resolution.source,
        });
        ToolApprovalAttempt::Resolved
    }

    pub(crate) fn try_send_effect(&self, effect: RemoteEffect) -> Result<(), &'static str> {
        let sender = match &effect {
            RemoteEffect::Prompt {
                mode: PromptMode::New | PromptMode::Queue,
                ..
            } => &self.prompt_effects,
            RemoteEffect::Prompt {
                mode: PromptMode::Steer,
                ..
            }
            | RemoteEffect::Cancel { .. } => &self.control_effects,
        };
        sender.try_send(effect).map_err(|_| "remote_busy")
    }

    pub(crate) fn status(&self) -> RemoteStatus {
        let now = Instant::now();
        let mut state = self.lock();
        cleanup_expired(&mut state, now);
        let controller_name = state.controller_id.as_ref().and_then(|id| {
            state
                .devices
                .values()
                .find(|device| &device.id == id)
                .map(|device| device.name.clone())
        });
        let mut pending_pairs = state
            .pending
            .values()
            .filter(|pending| matches!(pending.decision, PairDecision::Waiting))
            .map(|pending| PendingPairOverview {
                device_name: pending.device_name.clone(),
                comparison_code: pending.comparison_code.clone(),
            })
            .collect::<Vec<_>>();
        pending_pairs.sort_by(|left, right| {
            left.comparison_code
                .cmp(&right.comparison_code)
                .then_with(|| left.device_name.cmp(&right.device_name))
        });
        RemoteStatus {
            devices: state.devices.len(),
            pending: pending_pairs.len(),
            pending_pairs,
            turn: state.turn,
            controller_name,
        }
    }

    pub(crate) fn offer_ttl_seconds() -> u64 {
        OFFER_TTL.as_secs()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn broadcast_control_state(&self, controller_exists: bool) {
        let _ = self.events.send(ServerMessage::ControlState {
            controller_exists,
            role: ControlRole::Observer,
        });
    }
}

fn normalized_device_name(name: &str) -> Result<String, PairingError> {
    let normalized = name.trim();
    if normalized.is_empty() || normalized.chars().count() > MAX_DEVICE_NAME_CHARS {
        return Err(PairingError::InvalidDeviceName);
    }
    Ok(normalized.to_string())
}

fn should_project_block(block: &Value) -> bool {
    if block.get("type").and_then(Value::as_str) != Some("system") {
        return true;
    }
    let Some(text) = block.get("text").and_then(Value::as_str) else {
        return true;
    };
    !text.starts_with("Copied block to clipboard") && !text.starts_with("Usage: /")
}

fn cleanup_expired(state: &mut State, now: Instant) {
    state.pending.retain(|_, pending| now < pending.expires_at);
    state.devices.retain(|_, device| now < device.expires_at);
    let active_device_ids = state
        .devices
        .values()
        .map(|device| device.id.clone())
        .collect::<HashSet<_>>();
    state
        .live_websockets
        .retain(|device_id, _| active_device_ids.contains(device_id));
    state
        .push_subscriptions
        .retain(|device_id, _| active_device_ids.contains(device_id));
    prune_push_rate_limits(state);
    if state
        .controller_id
        .as_ref()
        .is_some_and(|id| !active_device_ids.contains(id))
    {
        state.controller_id = None;
        state.controller_grace = None;
    }
}

fn has_open_websockets(state: &State) -> bool {
    state.live_websockets.values().any(|count| *count > 0)
}

fn collect_push_targets(
    state: &mut State,
    now: Instant,
    bypass_rate_limit: bool,
) -> Vec<PushTarget> {
    let subscriptions = state
        .push_subscriptions
        .iter()
        .map(|(device_id, subscription)| (device_id.clone(), subscription.clone()))
        .collect::<Vec<_>>();
    let mut selected_endpoints = HashSet::new();
    let mut targets = Vec::new();
    for (device_id, subscription) in subscriptions {
        let endpoint_key = subscription.endpoint_key().to_string();
        let outside_gap = state
            .push_last_sent
            .get(&endpoint_key)
            .and_then(|last_sent| now.checked_duration_since(*last_sent))
            .is_none_or(|elapsed| elapsed >= PUSH_MIN_GAP);
        if !selected_endpoints.insert(endpoint_key.clone())
            || (!bypass_rate_limit && !outside_gap)
        {
            continue;
        }
        state.push_last_sent.insert(endpoint_key.clone(), now);
        targets.push(PushTarget {
            device_id,
            endpoint_key,
            subscription,
        });
    }
    targets
}

fn prune_push_rate_limits(state: &mut State) {
    let active_endpoints = state
        .push_subscriptions
        .values()
        .map(|subscription| subscription.endpoint_key().to_string())
        .collect::<HashSet<_>>();
    state
        .push_last_sent
        .retain(|endpoint, _| active_endpoints.contains(endpoint));
}

fn role_for(state: &State, device_id: &str) -> ControlRole {
    if state.controller_id.as_deref() == Some(device_id) {
        ControlRole::Controller
    } else {
        ControlRole::Observer
    }
}

fn digest(value: &str) -> [u8; 32] {
    Sha256::digest(value.as_bytes()).into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

pub(crate) fn random_token(bytes: usize) -> String {
    let mut data = vec![0_u8; bytes];
    for chunk in data.chunks_mut(32) {
        let random = rand::random::<[u8; 32]>();
        chunk.copy_from_slice(&random[..chunk.len()]);
    }
    URL_SAFE_NO_PAD.encode(data)
}

fn comparison_code() -> String {
    const ALPHABET: &[u8] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
    let random = rand::random::<[u8; 6]>();
    let chars = random.map(|byte| ALPHABET[usize::from(byte) % ALPHABET.len()] as char);
    format!(
        "{}{}{}-{}{}{}",
        chars[0], chars[1], chars[2], chars[3], chars[4], chars[5]
    )
}

#[cfg(test)]
mod tests {
    use tokio::sync::{mpsc, oneshot};

    use super::{
        CommandDecision, ControlRole, PairPoll, PairingError, RemoteShared, TurnPhase,
        has_open_websockets, should_project_block,
    };
    use crate::remote_control::protocol::{
        ToolApprovalChoice, ToolApprovalDecision, ToolApprovalRequest, ToolApprovalSource,
    };
    use crate::remote_control::push::validated_subscription_for_test;

    fn shared() -> RemoteShared {
        let (prompt_effect_tx, _prompt_effect_rx) = mpsc::channel(8);
        let (control_effect_tx, _control_effect_rx) = mpsc::channel(8);
        let (notice_tx, _notice_rx) = mpsc::channel(8);
        RemoteShared::new_with_push(
            "session-1".to_string(),
            "zo".to_string(),
            prompt_effect_tx,
            control_effect_tx,
            notice_tx,
            super::PushService::disabled_for_test(),
        )
    }

    fn pair_device(
        shared: &RemoteShared,
        offer: &str,
        name: &str,
    ) -> super::AuthenticatedDevice {
        shared.refresh_offer(offer);
        let pairing = shared
            .begin_pairing(offer, name)
            .expect("pairing starts");
        shared
            .approve(&pairing.comparison_code)
            .expect("pairing is approved");
        let PairPoll::Approved { token, .. } = shared.pairing_status(&pairing.id) else {
            panic!("approved pairing must mint a credential")
        };
        shared.authenticate(&token).expect("credential is valid")
    }

    fn approval() -> ToolApprovalRequest {
        ToolApprovalRequest {
            request_id: String::new(),
            tool_name: "Bash".to_string(),
            input_summary: "cargo test".to_string(),
            input_hash: "sha256:test".to_string(),
            choices: vec![ToolApprovalChoice {
                label: "Allow once".to_string(),
                decision: ToolApprovalDecision::AllowOnce,
            }],
        }
    }

    #[test]
    fn session_info_update_projects_named_and_unnamed_titles() {
        let shared = shared();
        assert_eq!(shared.session_info().title, "zo");

        shared.set_session_info("session-1", "Zo · 배포 관찰");
        assert_eq!(shared.session_info().title, "Zo · 배포 관찰");

        shared.set_session_info("session-1", "Zo · zo");
        assert_eq!(shared.session_info().title, "Zo · zo");
    }

    #[test]
    fn controller_grace_is_last_socket_only_and_generation_fenced() {
        let shared = shared();
        let controller = pair_device(&shared, "controller-offer", "Phone");

        assert!(shared.websocket_connected(&controller.id));
        assert!(shared.websocket_connected(&controller.id));
        assert!(shared.websocket_disconnected(&controller.id).is_none());
        let first_grace = shared
            .websocket_disconnected(&controller.id)
            .expect("last controller socket starts grace");

        assert!(shared.websocket_connected(&controller.id));
        assert!(!shared.expire_controller_grace(&first_grace));
        assert_eq!(shared.role(&controller.id), ControlRole::Controller);

        let second_grace = shared
            .websocket_disconnected(&controller.id)
            .expect("the reconnected controller starts a new grace");
        assert!(shared.expire_controller_grace(&second_grace));
        assert_eq!(shared.role(&controller.id), ControlRole::Observer);
    }

    #[test]
    fn first_controller_grant_broadcasts_control_state() {
        let shared = shared();
        let mut events = shared.events();

        let controller = pair_device(&shared, "controller-offer", "Phone");

        assert_eq!(shared.role(&controller.id), ControlRole::Controller);
        assert!(matches!(
            events.try_recv(),
            Ok(super::ServerMessage::ControlState {
                controller_exists: true,
                ..
            })
        ));
    }

    #[test]
    fn rotate_and_revoke_invalidate_pending_controller_grace() {
        for revoke in [false, true] {
            let shared = shared();
            let controller = pair_device(&shared, "controller-offer", "Phone");
            assert!(shared.websocket_connected(&controller.id));
            let grace = shared
                .websocket_disconnected(&controller.id)
                .expect("controller disconnect starts grace");

            if revoke {
                shared.revoke_all();
            } else {
                shared.rotate("rotated-offer");
            }

            assert!(!shared.expire_controller_grace(&grace));
            assert_eq!(shared.role(&controller.id), ControlRole::Observer);
        }
    }

    #[test]
    fn remote_onboarding_status_sorts_waiting_pairs_and_expires_them() {
        let shared = shared();
        shared.refresh_offer("first");
        let first = shared
            .begin_pairing("first", "Zebra phone")
            .expect("first pairing starts");
        shared.refresh_offer("second");
        let second = shared
            .begin_pairing("second", "Alpha tablet")
            .expect("second pairing starts");
        {
            let mut state = shared.lock();
            state
                .pending
                .get_mut(&first.id)
                .expect("first pending pair")
                .comparison_code = "ZZZ-999".to_string();
            state
                .pending
                .get_mut(&second.id)
                .expect("second pending pair")
                .comparison_code = "AAA-222".to_string();
        }

        let status = shared.status();
        assert_eq!(status.pending, 2);
        assert_eq!(
            status
                .pending_pairs
                .iter()
                .map(|pair| (pair.device_name.clone(), pair.comparison_code.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("Alpha tablet".to_string(), "AAA-222".to_string()),
                ("Zebra phone".to_string(), "ZZZ-999".to_string()),
            ]
        );

        shared
            .lock()
            .pending
            .get_mut(&first.id)
            .expect("first pending pair")
            .expires_at = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("one second before now is representable");
        let status = shared.status();
        assert_eq!(status.pending, 1);
        assert_eq!(status.pending_pairs[0].comparison_code, "AAA-222");
    }

    #[test]
    fn pairing_is_single_use_and_requires_local_approval() {
        let shared = shared();
        shared.refresh_offer("secret");
        let started = shared
            .begin_pairing("secret", "Phone")
            .expect("offer should be accepted once");
        assert_eq!(started.expires_in_seconds, super::OFFER_TTL.as_secs());
        assert_eq!(
            started.poll_expires_in_seconds,
            super::OFFER_TTL.as_secs() + super::APPROVED_POLL_TTL.as_secs()
        );
        assert!(matches!(shared.pairing_status(&started.id), PairPoll::Pending));
        assert!(matches!(
            shared.begin_pairing("secret", "Other"),
            Err(PairingError::InvalidOffer)
        ));

        let approved = shared
            .approve(&started.comparison_code)
            .expect("local approval should succeed");
        assert_eq!(approved.role, ControlRole::Controller);
        let PairPoll::Approved { token, role } = shared.pairing_status(&started.id) else {
            panic!("approval must mint a credential")
        };
        assert_eq!(role, ControlRole::Controller);
        assert!(shared.authenticate(&token).is_some());
        let PairPoll::Approved {
            token: retry_token,
            role: retry_role,
        } = shared.pairing_status(&started.id)
        else {
            panic!("approved status must be replayable after a lost response")
        };
        assert_eq!(retry_role, ControlRole::Controller);
        assert_eq!(retry_token, token);
    }

    #[test]
    fn approval_extends_polling_deadline_from_decision_time() {
        let shared = shared();
        shared.refresh_offer("secret");
        let started = shared
            .begin_pairing("secret", "Phone")
            .expect("pairing starts");
        {
            let mut state = shared.lock();
            state
                .pending
                .get_mut(&started.id)
                .expect("pending pair exists")
                .expires_at = std::time::Instant::now() + std::time::Duration::from_secs(1);
        }

        let approved_at = std::time::Instant::now();
        shared
            .approve(&started.comparison_code)
            .expect("near-expiry approval succeeds");
        {
            let state = shared.lock();
            let pending = state.pending.get(&started.id).expect("approval retained");
            assert!(pending.expires_at >= approved_at + super::APPROVED_POLL_TTL);
        }
        assert!(matches!(
            shared.pairing_status(&started.id),
            PairPoll::Approved { .. }
        ));
    }

    #[test]
    fn refreshed_offer_preserves_devices_and_pending_pairings() {
        let shared = shared();
        shared.refresh_offer("controller-offer");
        let controller_pair = shared
            .begin_pairing("controller-offer", "Phone")
            .expect("controller pairing");
        shared
            .approve(&controller_pair.comparison_code)
            .expect("approve controller");
        let PairPoll::Approved {
            token: controller_token,
            role: ControlRole::Controller,
        } = shared.pairing_status(&controller_pair.id)
        else {
            panic!("first device must be the controller")
        };
        assert!(!shared.offer_available("controller-offer"));

        shared.refresh_offer("observer-offer");
        assert!(shared.offer_available("observer-offer"));
        let observer_pair = shared
            .begin_pairing("observer-offer", "Tablet")
            .expect("observer pairing");
        shared.refresh_offer("next-offer");

        let approved = shared
            .approve(&observer_pair.comparison_code)
            .expect("refresh must preserve pending approval");
        assert_eq!(approved.role, ControlRole::Observer);
        let PairPoll::Approved {
            token: observer_token,
            role: ControlRole::Observer,
        } = shared.pairing_status(&observer_pair.id)
        else {
            panic!("second device must be an observer")
        };
        assert!(shared.authenticate(&controller_token).is_some());
        assert!(shared.authenticate(&observer_token).is_some());
        assert!(shared.begin_pairing("next-offer", "Laptop").is_ok());
    }

    #[test]
    fn rotate_revokes_devices_and_pending_offers() {
        let shared = shared();
        shared.refresh_offer("first");
        let started = shared.begin_pairing("first", "Phone").expect("pairing");
        shared.approve(&started.comparison_code).expect("approve");
        let PairPoll::Approved { token, .. } = shared.pairing_status(&started.id) else {
            panic!("approved")
        };
        shared.rotate("second");
        assert!(shared.authenticate(&token).is_none());
        assert!(matches!(shared.pairing_status(&started.id), PairPoll::Expired));
        assert!(shared.begin_pairing("second", "Phone").is_ok());
    }

    #[test]
    fn command_ids_are_idempotent_per_device() {
        let shared = shared();
        assert_eq!(
            shared.begin_command("device", "command-1"),
            Ok(CommandDecision::New)
        );
        assert_eq!(
            shared.begin_command("device", "command-1"),
            Ok(CommandDecision::Duplicate)
        );
    }

    #[test]
    fn snapshot_replays_only_available_sequence_range() {
        let shared = shared();
        shared.record_frame(serde_json::json!({"type":"text_delta","text":"a"}));
        shared.record_frame(serde_json::json!({"type":"text_delta","text":"b"}));
        shared.set_turn(TurnPhase::Running, 1);
        assert!(matches!(
            shared.snapshot_for(1),
            super::SnapshotPlan::Replay { ref frames, .. } if frames.len() == 1
        ));
        assert!(matches!(
            shared.snapshot_for(99),
            super::SnapshotPlan::Full { .. }
        ));
    }

    #[test]
    fn remote_projection_filters_only_desktop_local_system_notices() {
        let clipboard = serde_json::json!({
            "type": "system",
            "level": "info",
            "text": "Copied block to clipboard via pbcopy (5414 chars)",
        });
        let slash_usage = serde_json::json!({
            "type": "system",
            "level": "error",
            "text": "Usage: /remote [start|status|qr|...]",
        });
        let regular_error = serde_json::json!({
            "type": "system",
            "level": "error",
            "text": "The model request failed; retry the turn.",
        });

        assert!(!should_project_block(&clipboard));
        assert!(!should_project_block(&slash_usage));
        assert!(should_project_block(&regular_error));

        let shared = shared();
        shared.replace_snapshot([clipboard, regular_error.clone(), slash_usage]);
        let super::SnapshotPlan::Full { frames, .. } = shared.snapshot_for(0) else {
            panic!("new clients receive the filtered snapshot")
        };
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].block, regular_error);
    }

    #[test]
    fn empty_replacement_invalidates_the_previous_sequence_epoch() {
        let shared = shared();
        shared.record_frame(serde_json::json!({"type":"text_delta","text":"stale"}));
        let previous_seq = shared.next_seq().saturating_sub(1);

        shared.replace_snapshot(std::iter::empty::<serde_json::Value>());
        let super::SnapshotPlan::Full { frames, next_seq } =
            shared.snapshot_for(previous_seq)
        else {
            panic!("a previous snapshot epoch must receive a full replacement")
        };
        assert!(frames.is_empty());
        assert!(next_seq > previous_seq.saturating_add(1));
        assert!(matches!(
            shared.snapshot_for(next_seq.saturating_sub(1)),
            super::SnapshotPlan::Replay { ref frames, next_seq: replay_next }
                if frames.is_empty() && replay_next == next_seq
        ));
    }

    #[test]
    fn replacement_keeps_the_latest_frames() {
        let shared = shared();
        let input_len = super::MAX_FRAMES + 2;
        shared.replace_snapshot(
            (0..input_len).map(|index| serde_json::json!({"index": index})),
        );

        let super::SnapshotPlan::Full { frames, .. } = shared.snapshot_for(0) else {
            panic!("a new client must receive a full snapshot")
        };
        assert_eq!(frames.len(), super::MAX_FRAMES);
        assert_eq!(frames.first().and_then(|frame| frame.block["index"].as_u64()), Some(2));
        assert_eq!(
            frames.last().and_then(|frame| frame.block["index"].as_u64()),
            Some((input_len - 1) as u64),
        );
    }

    #[test]
    fn non_empty_replacement_cannot_replay_from_the_previous_epoch() {
        let shared = shared();
        shared.record_frame(serde_json::json!({"type":"text_delta","text":"stale"}));
        let previous_seq = shared.next_seq().saturating_sub(1);
        let fresh = serde_json::json!({"type":"text_delta","text":"fresh"});

        shared.replace_snapshot([fresh.clone()]);
        let super::SnapshotPlan::Full { frames, .. } = shared.snapshot_for(previous_seq) else {
            panic!("a previous snapshot epoch must receive a full replacement")
        };
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].block, fresh);
        assert!(frames[0].seq > previous_seq.saturating_add(1));
    }

    #[test]
    fn push_registry_replaces_per_device_and_delete_is_idempotent() {
        let shared = shared();
        let device = pair_device(&shared, "push-offer", "Phone");
        let first = validated_subscription_for_test(
            "https://fcm.googleapis.com/send/first-capability",
        );
        assert!(shared.replace_push_subscription(&device.id, first));
        assert_eq!(shared.lock().push_subscriptions.len(), 1);

        let second = validated_subscription_for_test(
            "https://fcm.googleapis.com/send/second-capability",
        );
        assert!(shared.replace_push_subscription(&device.id, second));
        let state = shared.lock();
        assert_eq!(state.push_subscriptions.len(), 1);
        assert_eq!(
            state
                .push_subscriptions
                .get(&device.id)
                .expect("subscription is registered")
                .endpoint_for_test(),
            "https://fcm.googleapis.com/send/second-capability",
        );
        drop(state);

        shared.remove_push_subscription(&device.id);
        shared.remove_push_subscription(&device.id);
        assert!(shared.lock().push_subscriptions.is_empty());
    }

    #[test]
    fn push_trigger_predicates_require_no_clients_pending_and_once_per_approval() {
        let shared = shared();
        let device = pair_device(&shared, "push-offer", "Phone");
        assert!(shared.replace_push_subscription(
            &device.id,
            validated_subscription_for_test("https://fcm.googleapis.com/send/capability"),
        ));

        assert!(!has_open_websockets(&shared.lock()));
        assert!(shared.websocket_connected(&device.id));
        assert!(has_open_websockets(&shared.lock()));
        let _ = shared.websocket_disconnected(&device.id);
        assert!(!has_open_websockets(&shared.lock()));

        let (first_tx, _first_rx) = oneshot::channel();
        let first_id = shared.publish_tool_approval(approval(), first_tx);
        assert_eq!(shared.prepare_approval_push(&first_id, false).len(), 1);
        assert!(shared.prepare_approval_push(&first_id, false).is_empty());

        let (second_tx, _second_rx) = oneshot::channel();
        let second_id = shared.publish_tool_approval(approval(), second_tx);
        assert_eq!(shared.prepare_approval_push(&second_id, false).len(), 1);
        assert!(super::collect_push_targets(
            &mut shared.lock(),
            std::time::Instant::now(),
            false,
        )
        .is_empty());

        let (resolved_tx, _resolved_rx) = oneshot::channel();
        let resolved_id = shared.publish_tool_approval(approval(), resolved_tx);
        assert_eq!(
            shared.resolve_tool_approval(
                &resolved_id,
                ToolApprovalDecision::AllowOnce,
                ToolApprovalSource::Tui,
            ),
            super::ToolApprovalAttempt::Resolved,
        );
        assert!(shared.prepare_approval_push(&resolved_id, false).is_empty());
    }
}
