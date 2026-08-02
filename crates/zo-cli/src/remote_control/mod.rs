mod exposure;
mod gateway;
mod manager;
mod protocol;
mod push;
mod state;

pub(crate) use manager::{RemoteInbox, RemoteManager};
pub(crate) use protocol::{
    PromptMode, ToolApprovalChoice, ToolApprovalDecision, ToolApprovalRequest, ToolApprovalSource,
    TurnPhase,
};
pub(crate) use state::{RemoteShared, ToolApprovalAttempt};
#[cfg(test)]
pub(crate) use protocol::ServerMessage;
#[cfg(test)]
pub(crate) use state::ToolApprovalResolution;
