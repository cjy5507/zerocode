use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const PROTOCOL_VERSION: u16 = 1;
pub(crate) const MAX_PROMPT_BYTES: usize = 32 * 1024;
pub(crate) const MAX_COMMAND_ID_BYTES: usize = 128;
pub(crate) const MAX_APPROVAL_ID_BYTES: usize = 128;
pub(crate) const MAX_DEVICE_NAME_CHARS: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptMode {
    New,
    Queue,
    Steer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TurnPhase {
    Idle,
    Running,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ControlRole {
    Observer,
    Controller,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolApprovalDecision {
    AllowOnce,
    AllowAlways,
    Deny,
    DenyAlways,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ToolApprovalChoice {
    pub(crate) label: String,
    pub(crate) decision: ToolApprovalDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ToolApprovalRequest {
    pub(crate) request_id: String,
    pub(crate) tool_name: String,
    pub(crate) input_summary: String,
    pub(crate) input_hash: String,
    pub(crate) choices: Vec<ToolApprovalChoice>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolApprovalSource {
    Tui,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FrameRecord {
    pub(crate) seq: u64,
    pub(crate) block: Value,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SessionInfo {
    pub(crate) id: String,
    pub(crate) title: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ClientMessage {
    Hello {
        version: u16,
        #[serde(default)]
        last_seq: u64,
    },
    ControlRequest {
        command_id: String,
    },
    PromptSubmit {
        command_id: String,
        text: String,
        mode: PromptMode,
    },
    TurnCancel {
        command_id: String,
    },
    ToolApprovalRespond {
        command_id: String,
        request_id: String,
        decision: ToolApprovalDecision,
    },
    Ping,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ServerMessage {
    Snapshot {
        version: u16,
        session: SessionInfo,
        frames: Vec<FrameRecord>,
        turn: TurnPhase,
        role: ControlRole,
        replace: bool,
        next_seq: u64,
        approvals: Vec<ToolApprovalRequest>,
    },
    Frame {
        frame: FrameRecord,
    },
    TurnState {
        turn: TurnPhase,
    },
    ControlState {
        controller_exists: bool,
        role: ControlRole,
    },
    CommandAccepted {
        command_id: String,
        duplicate: bool,
    },
    CommandRejected {
        command_id: String,
        code: &'static str,
        message: String,
    },
    ResyncRequired {
        next_seq: u64,
    },
    ToolApprovalRequest {
        approval: ToolApprovalRequest,
    },
    ToolApprovalResolved {
        request_id: String,
        decision: ToolApprovalDecision,
        source: ToolApprovalSource,
    },
    Error {
        code: &'static str,
        message: String,
        recoverable: bool,
    },
    Pong,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PairStartRequest {
    pub(crate) secret: String,
    pub(crate) device_name: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct PairStartResponse {
    pub(crate) pairing_id: String,
    pub(crate) comparison_code: String,
    pub(crate) expires_in_seconds: u64,
    pub(crate) poll_expires_in_seconds: u64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum PairStatusResponse {
    Pending,
    Approved { role: ControlRole },
    Denied,
    Expired,
}

#[cfg(test)]
mod tests {
    use super::{
        ClientMessage, ControlRole, PromptMode, PROTOCOL_VERSION, ServerMessage,
        ToolApprovalChoice, ToolApprovalDecision, ToolApprovalRequest, ToolApprovalSource,
    };

    #[test]
    fn wire_protocol_rejects_unknown_message_types() {
        let parsed = serde_json::from_str::<ClientMessage>(r#"{"type":"session.delete"}"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn prompt_mode_is_explicit_on_the_wire() {
        let parsed = serde_json::from_str::<ClientMessage>(&format!(
            r#"{{"type":"prompt_submit","command_id":"c1","text":"hello","mode":"queue","version":{PROTOCOL_VERSION}}}"#
        ))
        .expect("valid message");
        assert!(matches!(
            parsed,
            ClientMessage::PromptSubmit {
                mode: PromptMode::Queue,
                ..
            }
        ));
    }

    #[test]
    fn control_state_matches_the_browser_role_contract() {
        let message = ServerMessage::ControlState {
            controller_exists: true,
            role: ControlRole::Controller,
        };
        assert_eq!(
            serde_json::to_value(message).expect("control state serializes"),
            serde_json::json!({
                "type": "control_state",
                "controller_exists": true,
                "role": "controller",
            })
        );
    }

    #[test]
    fn remote_tool_approval_messages_serialize_with_explicit_choices() {
        let request = ServerMessage::ToolApprovalRequest {
            approval: ToolApprovalRequest {
                request_id: "approval-1".to_string(),
                tool_name: "Bash".to_string(),
                input_summary: "cargo test -p runtime".to_string(),
                input_hash: "abc123".to_string(),
                choices: vec![
                    ToolApprovalChoice {
                        label: "Allow once".to_string(),
                        decision: ToolApprovalDecision::AllowOnce,
                    },
                    ToolApprovalChoice {
                        label: "Deny".to_string(),
                        decision: ToolApprovalDecision::Deny,
                    },
                ],
            },
        };
        assert_eq!(
            serde_json::to_value(request).expect("approval request serializes"),
            serde_json::json!({
                "type": "tool_approval_request",
                "approval": {
                    "request_id": "approval-1",
                    "tool_name": "Bash",
                    "input_summary": "cargo test -p runtime",
                    "input_hash": "abc123",
                    "choices": [
                        { "label": "Allow once", "decision": "allow_once" },
                        { "label": "Deny", "decision": "deny" },
                    ],
                },
            })
        );

        let response = serde_json::from_value::<ClientMessage>(serde_json::json!({
            "type": "tool_approval_respond",
            "command_id": "command-1",
            "request_id": "approval-1",
            "decision": "allow_once",
        }))
        .expect("approval response deserializes");
        assert!(matches!(
            response,
            ClientMessage::ToolApprovalRespond {
                decision: ToolApprovalDecision::AllowOnce,
                ..
            }
        ));

        let resolved = ServerMessage::ToolApprovalResolved {
            request_id: "approval-1".to_string(),
            decision: ToolApprovalDecision::Deny,
            source: ToolApprovalSource::Tui,
        };
        assert_eq!(
            serde_json::to_value(resolved).expect("approval resolution serializes"),
            serde_json::json!({
                "type": "tool_approval_resolved",
                "request_id": "approval-1",
                "decision": "deny",
                "source": "tui",
            })
        );
    }
}
