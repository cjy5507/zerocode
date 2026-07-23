//! Token counting via the Anthropic `/v1/messages/count_tokens` endpoint.
//!
//! Accepts the same request body as `/v1/messages` and returns only the
//! token count, without generating a response. Useful for pre-flight
//! context-window checks and cost estimation.

use crate::error::ApiError;
use crate::providers::anthropic::AnthropicClient;
use crate::types::MessageRequest;

use serde::{Deserialize, Serialize};

/// Response from the Anthropic token counting endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCount {
    /// Number of input tokens the request would consume.
    pub input_tokens: u32,
}

impl AnthropicClient {
    /// Count the input tokens for a [`MessageRequest`] without generating
    /// a response.
    ///
    /// Calls `POST /v1/messages/count_tokens` with the same body shape
    /// as a normal message request.
    pub async fn count_tokens(&self, request: &MessageRequest) -> Result<TokenCount, ApiError> {
        let base_url = self.base_url().trim_end_matches('/');
        let url = format!("{base_url}/v1/messages/count_tokens");

        // NOTE: this serializes the raw request directly. It currently has no
        // production callers; if wired up, route `request` through the same
        // `normalize_thinking_for_wire` + `strip_thinking_blocks_when_disabled`
        // chain as `send_raw_request`, or a thinking-off request carrying replayed
        // thinking blocks (or a legacy `budget_tokens` on an adaptive model) will
        // 400 here just as it would on the message endpoint.
        let request_body = self.request_profile().render_json_body(request)?;

        let mut builder = self
            .http_client()
            .post(&url)
            .header("content-type", "application/json")
            .json(&request_body);

        builder = self.auth_source().apply(builder);
        for (header_name, header_value) in self.request_profile().header_pairs() {
            builder = builder.header(header_name, header_value);
        }

        let response = builder.send().await.map_err(ApiError::from)?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::Api {
                status,
                error_type: None,
                message: Some(format!("count_tokens failed: {body}")),
                body,
                retryable: status.as_u16() == 429
                    || status.as_u16() == 529
                    || status.is_server_error(),
                retry_after: None,
            });
        }

        response.json::<TokenCount>().await.map_err(ApiError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_deserializes() {
        let json = r#"{"input_tokens": 42}"#;
        let count: TokenCount = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(count.input_tokens, 42);
    }

    #[test]
    fn token_count_serializes() {
        let count = TokenCount { input_tokens: 100 };
        let json = serde_json::to_string(&count).expect("should serialize");
        assert!(json.contains("\"input_tokens\":100"));
    }
}
