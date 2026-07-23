//! 붙여넣은 텍스트를 인라인 이미지로 받아들이는 파서 — `data:` URL 형태의
//! 클립보드 페이스트를 (`media_type`, `base64`) 로 정규화한다. 순수 함수.

pub(super) fn pasted_image_data_url(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();
    let (header, data) = trimmed.split_once(',')?;
    let metadata = header.strip_prefix("data:")?;
    let mut parts = metadata.split(';');
    let media_type = normalize_pasted_image_media_type(parts.next()?)?;
    if !parts.any(|part| part.eq_ignore_ascii_case("base64")) {
        return None;
    }
    let data = compact_base64_payload(data)?;
    Some((media_type, data))
}

fn normalize_pasted_image_media_type(media_type: &str) -> Option<String> {
    let lowered = media_type.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "image/png" | "image/jpeg" | "image/webp" | "image/gif" => Some(lowered),
        "image/jpg" => Some("image/jpeg".to_string()),
        _ => None,
    }
}

fn compact_base64_payload(data: &str) -> Option<String> {
    let mut compact = String::with_capacity(data.len());
    for ch in data.chars() {
        if ch.is_ascii_whitespace() {
            continue;
        }
        if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '-' | '_') {
            compact.push(ch);
        } else {
            return None;
        }
    }
    (!compact.is_empty()).then_some(compact)
}
