use std::sync::Arc;
use std::time::Duration;

use qrcode::QrCode;
use qrcode::render::unicode;
use runtime::message_stream::{BlockIdGen, RenderBlock, SystemLevel};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use zo_cli::sinks::SerializableRenderBlock;
use zo_cli::tui::AgentCommand;

use super::exposure::{Exposure, bind_listener, discover_or_configure, remove_serve_mapping};
use super::gateway::{GatewayState, serve};
use super::protocol::{ControlRole, PromptMode, TurnPhase};
use super::state::{
    PairingNotice, RemoteEffect, RemoteShared, RemoteStatus, random_token,
};

const INPUT_CAPACITY: usize = 8;
const EFFECT_CAPACITY: usize = 16;
const NOTICE_CAPACITY: usize = 4;
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

type RemoteRenderObserver = Arc<dyn Fn(&RenderBlock) + Send + Sync>;

#[derive(Debug, Clone)]
pub(crate) enum RemoteInbox {
    Prompt { text: String, mode: PromptMode },
}

#[derive(Debug, Clone)]
pub(crate) struct RemoteOverview {
    pub(crate) running: bool,
    pub(crate) origin: Option<String>,
    pub(crate) status: RemoteStatus,
}

pub(crate) struct RemoteManager {
    agent_commands: mpsc::Sender<AgentCommand>,
    local_render: mpsc::Sender<RenderBlock>,
    ids: BlockIdGen,
    turn_generation: u64,
    active: Option<ActiveRemote>,
}

struct ActiveRemote {
    shared: RemoteShared,
    exposure: Exposure,
    secret: String,
    cancellation: CancellationToken,
    tasks: TaskTracker,
    input_rx: mpsc::Receiver<RemoteInbox>,
}

struct DispatcherOutputs {
    input: mpsc::Sender<RemoteInbox>,
    local_render: mpsc::Sender<RenderBlock>,
    agent_commands: mpsc::Sender<AgentCommand>,
    remote_url: String,
}

impl RemoteManager {
    pub(crate) fn new(
        agent_commands: mpsc::Sender<AgentCommand>,
        local_render: mpsc::Sender<RenderBlock>,
        ids: BlockIdGen,
    ) -> Self {
        Self {
            agent_commands,
            local_render,
            ids,
            turn_generation: 0,
            active: None,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Return a local-only, credential-free snapshot for the TUI onboarding modal.
    pub(crate) fn overview(&self) -> RemoteOverview {
        let Some(active) = self.active.as_ref() else {
            return RemoteOverview {
                running: false,
                origin: None,
                status: RemoteStatus {
                    devices: 0,
                    pending: 0,
                    pending_pairs: Vec::new(),
                    turn: TurnPhase::Idle,
                    controller_name: None,
                },
            };
        };
        RemoteOverview {
            running: true,
            origin: Some(active.exposure.url()),
            status: active.shared.status(),
        }
    }

    pub(crate) async fn start(
        &mut self,
        session_id: String,
        title: String,
        snapshot: Vec<serde_json::Value>,
        progress: mpsc::UnboundedSender<String>,
    ) -> Result<String, String> {
        if self.active.is_some() {
            return self.qr_report();
        }

        let listener = bind_listener()
            .await
            .map_err(|error| format!("Zo Remote\n  Not started       {error}"))?;
        let port = listener
            .local_addr()
            .map_err(|error| format!("Zo Remote\n  Not started       {error}"))?
            .port();
        let exposure = discover_or_configure(port, |update| {
            let _ = progress.send(update.message());
        })
            .await
            .map_err(|error| format!("Zo Remote\n  Not started       {error}"))?;
        if listener
            .local_addr()
            .map_err(|error| error.to_string())?
            != exposure.bind_addr
        {
            return Err("Zo Remote\n  Refused an unexpected listener address".to_string());
        }

        let (prompt_effect_tx, prompt_effect_rx) = mpsc::channel(EFFECT_CAPACITY);
        let (control_effect_tx, control_effect_rx) = mpsc::channel(EFFECT_CAPACITY);
        let (notice_tx, notice_rx) = mpsc::channel(NOTICE_CAPACITY);
        let (input_tx, input_rx) = mpsc::channel(INPUT_CAPACITY);
        let shared = RemoteShared::new(
            session_id,
            title,
            prompt_effect_tx,
            control_effect_tx,
            notice_tx,
        );
        shared.replace_snapshot(snapshot);
        let secret = random_token(32);
        shared.refresh_offer(&secret);

        let cancellation = CancellationToken::new();
        let tasks = TaskTracker::new();
        let gateway_state = GatewayState::new(
            shared.clone(),
            exposure.host.clone(),
            exposure.origin.clone(),
            exposure.mount_path.clone(),
            cancellation.child_token(),
        );
        let gateway_cancel = cancellation.child_token();
        tasks.spawn(async move {
            let result = serve(listener, gateway_state).await;
            if let Err(error) = result {
                if !gateway_cancel.is_cancelled() {
                    eprintln!("[zo] Zo Remote gateway stopped unexpectedly: {error}");
                }
            }
        });
        spawn_dispatcher(
            &tasks,
            &cancellation,
            prompt_effect_rx,
            control_effect_rx,
            notice_rx,
            DispatcherOutputs {
                input: input_tx,
                local_render: self.local_render.clone(),
                agent_commands: self.agent_commands.clone(),
                remote_url: exposure.url(),
            },
            self.ids.clone(),
        );

        self.active = Some(ActiveRemote {
            shared,
            exposure,
            secret,
            cancellation,
            tasks,
            input_rx,
        });
        self.qr_report()
    }

    pub(crate) fn project_snapshot(transcript: &[RenderBlock]) -> Vec<serde_json::Value> {
        transcript.iter().filter_map(project_block).collect()
    }

    pub(crate) fn observer(
        &self,
    ) -> Option<RemoteRenderObserver> {
        let shared = self.active.as_ref()?.shared.clone();
        Some(Arc::new(move |block: &RenderBlock| {
            if let Some(block) = project_block(block) {
                shared.record_frame(block);
            }
        }))
    }

    pub(crate) fn approval_shared(&self) -> Option<RemoteShared> {
        self.active.as_ref().map(|active| active.shared.clone())
    }

    pub(crate) fn status_report(&self) -> String {
        let Some(active) = self.active.as_ref() else {
            return "Zo Remote\n  Status            stopped\n  Start             /remote".to_string();
        };
        format_status(&active.exposure.url(), &active.shared.status())
    }

    pub(crate) fn qr_report(&mut self) -> Result<String, String> {
        let Some(active) = self.active.as_mut() else {
            return Err("Zo Remote is not running. Start it with /remote.".to_string());
        };
        if !active.shared.offer_available(&active.secret) {
            active.secret = random_token(32);
            active.shared.refresh_offer(&active.secret);
        }
        Ok(format_offer(&active.exposure.url(), &active.secret))
    }

    pub(crate) fn rotate(&mut self) -> Result<String, String> {
        let Some(active) = self.active.as_mut() else {
            return Err("Zo Remote is not running. Start it with /remote.".to_string());
        };
        active.secret = random_token(32);
        active.shared.rotate(&active.secret);
        Ok(format_offer(&active.exposure.url(), &active.secret))
    }

    pub(crate) fn approve(&self, code: &str) -> Result<String, String> {
        let Some(active) = self.active.as_ref() else {
            return Err("Zo Remote is not running.".to_string());
        };
        active.shared.approve(code).map_or_else(
            |error| Err(format!("Zo Remote\n  Approval failed   {error}")),
            |approved| {
                let role = match approved.role {
                    ControlRole::Controller => "controller",
                    ControlRole::Observer => "observer",
                };
                Ok(format!(
                    "Zo Remote\n  Approved          {}\n  Role              {role}",
                    approved.device_name
                ))
            },
        )
    }

    pub(crate) fn deny(&self, code: &str) -> Result<String, String> {
        let Some(active) = self.active.as_ref() else {
            return Err("Zo Remote is not running.".to_string());
        };
        active.shared.deny(code).map_or_else(
            |error| Err(format!("Zo Remote\n  Denial failed     {error}")),
            |device_name| {
                Ok(format!(
                    "Zo Remote\n  Denied            {device_name}"
                ))
            },
        )
    }

    pub(crate) fn set_turn(&mut self, turn: TurnPhase) -> u64 {
        if turn == TurnPhase::Running {
            self.turn_generation = self.turn_generation.saturating_add(1);
        }
        let turn_generation = self.turn_generation;
        if let Some(active) = self.active.as_ref() {
            let previous = active.shared.turn();
            active.shared.set_turn(turn, turn_generation);
            if previous != TurnPhase::Idle && turn == TurnPhase::Idle {
                active.shared.push_turn_idle_if_disconnected();
            }
        }
        turn_generation
    }

    pub(crate) async fn next_inbox(&mut self) -> Option<RemoteInbox> {
        match self.active.as_mut() {
            Some(active) => active.input_rx.recv().await,
            None => std::future::pending().await,
        }
    }

    pub(crate) fn replace_snapshot(&self, transcript: &[RenderBlock]) {
        if let Some(active) = self.active.as_ref() {
            active
                .shared
                .replace_snapshot(transcript.iter().filter_map(project_block));
        }
    }

    pub(crate) fn set_session_info(&self, session_id: &str, title: &str) {
        if let Some(active) = self.active.as_ref() {
            active.shared.set_session_info(session_id, title);
        }
    }

    pub(crate) async fn stop(&mut self) -> String {
        let Some(active) = self.active.take() else {
            return "Zo Remote\n  Status            already stopped".to_string();
        };
        active.shared.revoke_all();
        let cleanup_error = remove_serve_mapping(&active.exposure).await.err();
        active.tasks.close();
        active.cancellation.cancel();
        let shutdown_timed_out = tokio::time::timeout(SHUTDOWN_TIMEOUT, active.tasks.wait())
            .await
            .is_err();
        if shutdown_timed_out || cleanup_error.is_some() {
            let warning = match (shutdown_timed_out, cleanup_error) {
                (true, Some(error)) => {
                    format!("background shutdown timed out; Serve cleanup failed: {error}")
                }
                (true, None) => "background shutdown timed out".to_string(),
                (false, Some(error)) => format!("Serve cleanup failed: {error}"),
                (false, None) => unreachable!(),
            };
            return format!(
                "Zo Remote\n  Status            stopped\n  Credentials       revoked\n  Warning           {warning}"
            );
        }
        "Zo Remote\n  Status            stopped\n  Credentials       revoked".to_string()
    }
}

impl Drop for RemoteManager {
    fn drop(&mut self) {
        if let Some(active) = self.active.take() {
            active.shared.revoke_all();
            active.tasks.close();
            active.cancellation.cancel();
        }
    }
}

fn spawn_dispatcher(
    tasks: &TaskTracker,
    cancellation: &CancellationToken,
    mut prompt_effects: mpsc::Receiver<RemoteEffect>,
    mut control_effects: mpsc::Receiver<RemoteEffect>,
    mut notices: mpsc::Receiver<PairingNotice>,
    outputs: DispatcherOutputs,
    ids: BlockIdGen,
) {
    let DispatcherOutputs {
        input,
        local_render,
        agent_commands,
        remote_url,
    } = outputs;

    let notice_cancel = cancellation.child_token();
    tasks.spawn(async move {
        loop {
            let notice = tokio::select! {
                () = notice_cancel.cancelled() => break,
                notice = notices.recv() => notice,
            };
            let Some(notice) = notice else { break };
            let block = pairing_notice_block(&ids, &notice, &remote_url);
            tokio::select! {
                () = notice_cancel.cancelled() => break,
                result = local_render.send(block) => {
                    if result.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let prompt_cancel = cancellation.child_token();
    tasks.spawn(async move {
        loop {
            let effect = tokio::select! {
                () = prompt_cancel.cancelled() => break,
                effect = prompt_effects.recv() => effect,
            };
            let Some(RemoteEffect::Prompt { text, mode, .. }) = effect else {
                break;
            };
            tokio::select! {
                () = prompt_cancel.cancelled() => break,
                result = input.send(RemoteInbox::Prompt { text, mode }) => {
                    if result.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let control_cancel = cancellation.child_token();
    tasks.spawn(async move {
        loop {
            let effect = tokio::select! {
                () = control_cancel.cancelled() => break,
                effect = control_effects.recv() => effect,
            };
            let command = match effect {
                Some(RemoteEffect::Prompt {
                    text,
                    mode: PromptMode::Steer,
                    turn_generation: Some(turn_generation),
                }) => Some(AgentCommand::RemoteSteer {
                    turn_generation,
                    text,
                }),
                Some(RemoteEffect::Cancel { turn_generation }) => {
                    Some(AgentCommand::RemoteCancelTurn { turn_generation })
                }
                Some(RemoteEffect::Prompt { .. }) => None,
                None => break,
            };
            if let Some(command) = command {
                tokio::select! {
                    () = control_cancel.cancelled() => break,
                    result = agent_commands.send(command) => {
                        if result.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    });
}

fn pairing_notice_block(
    ids: &BlockIdGen,
    notice: &PairingNotice,
    remote_url: &str,
) -> RenderBlock {
    RenderBlock::System {
        id: ids.next(),
        level: SystemLevel::Warn,
        text: format!(
            "Zo Remote pairing\n  URL                {remote_url}\n  Device             {}\n  Compare code       {}\n  Expires in         {} seconds\n\nApprove locally with /remote approve {} or deny with /remote deny {}.",
            notice.device_name,
            notice.comparison_code,
            notice.expires_in_seconds,
            notice.comparison_code,
            notice.comparison_code,
        ),
    }
}

fn project_block(block: &RenderBlock) -> Option<serde_json::Value> {
    if matches!(block, RenderBlock::PermissionPrompt(_)) {
        // Permission prompts use the dedicated authenticated approval event so
        // the browser receives only the display-safe summary + input hash.
        return None;
    }
    if matches!(
        block,
        RenderBlock::System { text, .. } if text.starts_with("Zo Remote")
    ) {
        return None;
    }
    Some(
        serde_json::to_value(SerializableRenderBlock::from_block(block)).unwrap_or_else(|_| {
            serde_json::json!({
                "type": "system",
                "level": "error",
                "text": "Zo could not project a local render event."
            })
        }),
    )
}

fn format_status(origin: &str, status: &RemoteStatus) -> String {
    let controller = status.controller_name.as_deref().unwrap_or("none");
    let turn = match status.turn {
        TurnPhase::Idle => "idle",
        TurnPhase::Running => "running",
    };
    format!(
        "Zo Remote\n  Status            running\n  URL               {origin}\n  Devices           {}\n  Pending           {}\n  Controller        {controller}\n  Turn              {turn}\n  Stop              /remote stop",
        status.devices, status.pending
    )
}

fn format_offer(remote_url: &str, secret: &str) -> String {
    let url = format!("{remote_url}#{secret}");
    let qr = QrCode::new(url.as_bytes()).map_or_else(
        |_| "[QR unavailable]".to_string(),
        |code| {
            code.render::<unicode::Dense1x2>()
                .quiet_zone(true)
                .build()
        },
    );
    format!(
        "Zo Remote\n  Status            running\n  Tailnet only      {remote_url}\n  Pairing expires   {} seconds\n  Local approval    required\n\n{qr}\n{url}\n\nScan with your phone, compare the code, then run /remote approve <code>.",
        RemoteShared::offer_ttl_seconds()
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use runtime::message_stream::{BlockId, BlockIdGen, RenderBlock, SystemLevel};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use tokio_util::task::TaskTracker;

    use super::super::protocol::ControlRole;
    use super::super::state::{PairPoll, RemoteShared};
    use super::{
        ActiveRemote, Exposure, PairingNotice, RemoteManager, project_block, spawn_dispatcher,
    };
    use zo_cli::tui::AgentCommand;

    #[test]
    fn inactive_manager_has_near_zero_runtime_state() {
        let (tx, _rx) = mpsc::channel::<AgentCommand>(1);
        let (render_tx, _render_rx) = mpsc::channel::<RenderBlock>(1);
        let manager = RemoteManager::new(tx, render_tx, BlockIdGen::default());
        assert!(!manager.is_active());
        assert!(manager.observer().is_none());
        assert!(manager.status_report().contains("stopped"));
        let overview = manager.overview();
        assert!(!overview.running);
        assert!(overview.origin.is_none());
        assert_eq!(overview.status.devices, 0);
        assert!(overview.status.pending_pairs.is_empty());
    }

    #[test]
    fn qr_report_refreshes_claimed_offer_without_revoking_devices() {
        let (agent_tx, _agent_rx) = mpsc::channel::<AgentCommand>(1);
        let (render_tx, _render_rx) = mpsc::channel::<RenderBlock>(1);
        let mut manager = RemoteManager::new(agent_tx, render_tx, BlockIdGen::default());
        let (prompt_tx, _prompt_rx) = mpsc::channel(1);
        let (control_tx, _control_rx) = mpsc::channel(1);
        let (notice_tx, _notice_rx) = mpsc::channel(1);
        let shared = RemoteShared::new(
            "session".to_string(),
            "zo".to_string(),
            prompt_tx,
            control_tx,
            notice_tx,
        );
        shared.refresh_offer("first-offer");
        let (_input_tx, input_rx) = mpsc::channel(1);
        manager.active = Some(ActiveRemote {
            shared: shared.clone(),
            exposure: Exposure {
                bind_addr: "127.0.0.1:8790".parse().expect("loopback address"),
                host: "zo.example".to_string(),
                mount_path: "/s8790".to_string(),
                origin: "https://zo.example".to_string(),
            },
            secret: "first-offer".to_string(),
            cancellation: CancellationToken::new(),
            tasks: TaskTracker::new(),
            input_rx,
        });

        let offer = manager.qr_report().expect("unclaimed offer is displayed");
        assert!(offer.contains("https://zo.example/s8790/#first-offer"));
        assert!(manager.status_report().contains("https://zo.example/s8790/"));
        assert_eq!(
            manager.overview().origin.as_deref(),
            Some("https://zo.example/s8790/")
        );
        assert_eq!(
            manager.active.as_ref().map(|active| active.secret.as_str()),
            Some("first-offer"),
        );
        let controller_pair = shared
            .begin_pairing("first-offer", "Phone")
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

        manager.qr_report().expect("claimed offer is refreshed");
        let refreshed_secret = manager
            .active
            .as_ref()
            .map(|active| active.secret.clone())
            .expect("manager remains active");
        assert_ne!(refreshed_secret, "first-offer");
        assert!(shared.offer_available(&refreshed_secret));
        assert!(shared.authenticate(&controller_token).is_some());

        let observer_pair = shared
            .begin_pairing(&refreshed_secret, "Tablet")
            .expect("observer pairing");
        let approved = shared
            .approve(&observer_pair.comparison_code)
            .expect("approve observer");
        assert_eq!(approved.role, ControlRole::Observer);
    }

    #[tokio::test]
    async fn pairing_notices_use_live_render_channel() {
        let tasks = TaskTracker::new();
        let cancellation = CancellationToken::new();
        let (_prompt_effect_tx, prompt_effect_rx) = mpsc::channel(1);
        let (_control_effect_tx, control_effect_rx) = mpsc::channel(1);
        let (notice_tx, notice_rx) = mpsc::channel(1);
        let (input_tx, mut input_rx) = mpsc::channel(1);
        let (render_tx, mut render_rx) = mpsc::channel(1);
        let (agent_tx, _agent_rx) = mpsc::channel(1);
        spawn_dispatcher(
            &tasks,
            &cancellation,
            prompt_effect_rx,
            control_effect_rx,
            notice_rx,
            super::DispatcherOutputs {
                input: input_tx,
                local_render: render_tx,
                agent_commands: agent_tx,
                remote_url: "https://laptop.example.ts.net/s8790/".to_string(),
            },
            BlockIdGen::default(),
        );

        notice_tx
            .send(PairingNotice {
                device_name: "phone".to_string(),
                comparison_code: "123456".to_string(),
                expires_in_seconds: 90,
            })
            .await
            .expect("dispatcher accepts pairing notice");
        let block = tokio::time::timeout(Duration::from_secs(1), render_rx.recv())
            .await
            .expect("pairing notice reaches live render channel")
            .expect("render channel remains open");
        let RenderBlock::System { text, .. } = &block else {
            panic!("pairing notice is a system block");
        };
        assert!(text.contains("Device             phone"));
        assert!(text.contains("Compare code       123456"));
        assert!(text.contains("URL                https://laptop.example.ts.net/s8790/"));
        assert!(project_block(&block).is_none());
        assert!(input_rx.try_recv().is_err());

        cancellation.cancel();
        tasks.close();
        tasks.wait().await;
    }

    #[test]
    fn remote_lifecycle_reports_are_local_only() {
        let offer = RenderBlock::System {
            id: BlockId(1),
            level: SystemLevel::Info,
            text: "Zo Remote\n  URL               https://host/#pairing-secret".to_string(),
        };
        let pairing = RenderBlock::System {
            id: BlockId(2),
            level: SystemLevel::Warn,
            text: "Zo Remote pairing\n  Compare code      123456".to_string(),
        };
        assert!(project_block(&offer).is_none());
        assert!(project_block(&pairing).is_none());

        let ordinary = RenderBlock::System {
            id: BlockId(3),
            level: SystemLevel::Info,
            text: "Build finished".to_string(),
        };
        let projected = project_block(&ordinary).expect("ordinary system block is projected");
        assert_eq!(projected["type"], "system");
        assert_eq!(projected["text"], "Build finished");
    }
}
