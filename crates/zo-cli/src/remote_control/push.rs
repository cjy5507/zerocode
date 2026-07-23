use std::net::IpAddr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use reqwest::StatusCode;
use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey};
use ring::agreement;
use ring::hkdf;
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{self, EcdsaKeyPair, KeyPair};
use serde::Deserialize;
use serde_json::json;

const RECORD_SIZE: u32 = 4096;
const MAX_PLAINTEXT_BYTES: usize = 3993;
const VAPID_EXPIRY: Duration = Duration::from_secs(12 * 60 * 60);
const PUSH_TIMEOUT: Duration = Duration::from_secs(10);
const BUILTIN_ALLOW_HOSTS: [&str; 4] = [
    "fcm.googleapis.com",
    "push.apple.com",
    "push.services.mozilla.com",
    "notify.windows.com",
];

static ENABLED_PUSH: OnceLock<Arc<EnabledPush>> = OnceLock::new();

#[derive(Clone)]
pub(crate) struct PushService {
    inner: Option<Arc<EnabledPush>>,
}

struct EnabledPush {
    client: reqwest::Client,
    vapid: VapidKey,
    debug_logging: bool,
}

struct VapidKey {
    key_pair: EcdsaKeyPair,
    public_key_b64: String,
}

#[derive(Deserialize)]
pub(crate) struct PushSubscriptionRequest {
    endpoint: String,
    keys: PushSubscriptionKeys,
}

#[derive(Deserialize)]
struct PushSubscriptionKeys {
    p256dh: String,
    auth: String,
}

#[derive(Clone)]
pub(crate) struct ValidatedSubscription {
    endpoint: reqwest::Url,
    endpoint_host: String,
    p256dh: Vec<u8>,
    auth: [u8; 16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubscriptionValidationError {
    Malformed,
    EndpointNotAllowed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PushReason {
    Approval,
    TurnIdle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendOutcome {
    Delivered,
    Gone,
    Failed,
}

#[derive(Clone, Copy)]
struct HkdfLength(usize);

impl hkdf::KeyType for HkdfLength {
    fn len(&self) -> usize {
        self.0
    }
}

impl PushService {
    pub(crate) fn from_env() -> Self {
        let disabled = std::env::var("ZO_REMOTE_PUSH")
            .ok()
            .as_deref()
            .is_some_and(push_disabled_value);
        if disabled {
            Self { inner: None }
        } else {
            Self::enabled()
        }
    }

    fn enabled() -> Self {
        let inner = ENABLED_PUSH
            .get_or_init(|| Arc::new(EnabledPush::new()))
            .clone();
        Self { inner: Some(inner) }
    }

    #[cfg(test)]
    pub(crate) fn enabled_for_test() -> Self {
        Self::enabled()
    }

    #[cfg(test)]
    pub(crate) fn disabled_for_test() -> Self {
        Self { inner: None }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    pub(crate) fn server_key(&self) -> Option<&str> {
        self.inner
            .as_ref()
            .map(|inner| inner.vapid.public_key_b64.as_str())
    }

    pub(crate) async fn send(
        &self,
        subscription: &ValidatedSubscription,
        reason: PushReason,
    ) -> SendOutcome {
        let Some(inner) = self.inner.as_ref() else {
            return SendOutcome::Failed;
        };
        let Ok(body) = encrypt(subscription, reason.plaintext()) else {
            inner.log_failure(&subscription.endpoint_host, None);
            return SendOutcome::Failed;
        };
        let audience = endpoint_audience(&subscription.endpoint);
        let Ok(authorization) = inner.vapid.authorization(&audience) else {
            inner.log_failure(&subscription.endpoint_host, None);
            return SendOutcome::Failed;
        };
        let response = inner
            .client
            .post(subscription.endpoint.clone())
            .header(reqwest::header::CONTENT_ENCODING, "aes128gcm")
            .header("TTL", reason.ttl())
            .header("Urgency", reason.urgency())
            .header("Topic", reason.topic())
            .header(reqwest::header::AUTHORIZATION, authorization)
            .body(body)
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => SendOutcome::Delivered,
            Ok(response)
                if matches!(response.status(), StatusCode::NOT_FOUND | StatusCode::GONE) =>
            {
                SendOutcome::Gone
            }
            Ok(response) => {
                inner.log_failure(&subscription.endpoint_host, Some(response.status()));
                SendOutcome::Failed
            }
            Err(_) => {
                inner.log_failure(&subscription.endpoint_host, None);
                SendOutcome::Failed
            }
        }
    }
}

impl EnabledPush {
    fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(PUSH_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("static Zo Remote push client configuration");
        let vapid = VapidKey::generate().expect("operating system random source creates VAPID key");
        Self {
            client,
            vapid,
            debug_logging: std::env::var("ZO_REMOTE_DEBUG").as_deref() == Ok("1"),
        }
    }

    fn log_failure(&self, host: &str, status: Option<StatusCode>) {
        if !self.debug_logging {
            return;
        }
        if let Some(status) = status {
            eprintln!("[remote] push {host} failed ({})", status.as_u16());
        } else {
            eprintln!("[remote] push {host} failed");
        }
    }
}

impl VapidKey {
    fn generate() -> Result<Self, ()> {
        let rng = SystemRandom::new();
        let pkcs8 = EcdsaKeyPair::generate_pkcs8(
            &signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &rng,
        )
        .map_err(|_| ())?;
        let key_pair = EcdsaKeyPair::from_pkcs8(
            &signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            pkcs8.as_ref(),
            &rng,
        )
        .map_err(|_| ())?;
        let public_key_b64 = URL_SAFE_NO_PAD.encode(key_pair.public_key().as_ref());
        Ok(Self {
            key_pair,
            public_key_b64,
        })
    }

    fn authorization(&self, audience: &str) -> Result<String, ()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let token = self.jwt_at(audience, now)?;
        Ok(format!(
            "vapid t={token}, k={}",
            self.public_key_b64
        ))
    }

    fn jwt_at(&self, audience: &str, now: u64) -> Result<String, ()> {
        let header = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&json!({"typ": "JWT", "alg": "ES256"}))
                .map_err(|_| ())?,
        );
        let claims = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&json!({
                "aud": audience,
                "exp": now.saturating_add(VAPID_EXPIRY.as_secs()),
                "sub": "mailto:zo-remote@localhost",
            }))
            .map_err(|_| ())?,
        );
        let signing_input = format!("{header}.{claims}");
        let signature = self
            .key_pair
            .sign(&SystemRandom::new(), signing_input.as_bytes())
            .map_err(|_| ())?;
        Ok(format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature.as_ref())
        ))
    }
}

impl PushSubscriptionRequest {
    #[cfg(test)]
    fn new(endpoint: impl Into<String>, p256dh: &[u8], auth: &[u8]) -> Self {
        Self {
            endpoint: endpoint.into(),
            keys: PushSubscriptionKeys {
                p256dh: URL_SAFE_NO_PAD.encode(p256dh),
                auth: URL_SAFE_NO_PAD.encode(auth),
            },
        }
    }
}

impl ValidatedSubscription {
    pub(crate) fn endpoint_key(&self) -> &str {
        self.endpoint.as_str()
    }

    #[cfg(test)]
    pub(crate) fn endpoint_for_test(&self) -> &str {
        self.endpoint.as_str()
    }
}

#[cfg(test)]
pub(crate) fn validated_subscription_for_test(endpoint: &str) -> ValidatedSubscription {
    let private_key = agreement::EphemeralPrivateKey::generate(
        &agreement::ECDH_P256,
        &SystemRandom::new(),
    )
    .expect("test push private key");
    let p256dh = private_key
        .compute_public_key()
        .expect("test push public key")
        .as_ref()
        .to_vec();
    let endpoint = reqwest::Url::parse(endpoint).expect("test push endpoint URL");
    let endpoint_host = endpoint
        .host_str()
        .expect("test push endpoint host")
        .to_string();
    ValidatedSubscription {
        endpoint,
        endpoint_host,
        p256dh,
        auth: [7_u8; 16],
    }
}

impl PushReason {
    fn plaintext(self) -> &'static [u8] {
        match self {
            Self::Approval => br#"{"reason":"approval"}"#,
            Self::TurnIdle => br#"{"reason":"turn_idle"}"#,
        }
    }

    fn ttl(self) -> &'static str {
        match self {
            Self::Approval => "600",
            Self::TurnIdle => "3600",
        }
    }

    fn urgency(self) -> &'static str {
        match self {
            Self::Approval => "high",
            Self::TurnIdle => "normal",
        }
    }

    fn topic(self) -> &'static str {
        match self {
            Self::Approval => "zo-appr",
            Self::TurnIdle => "zo-turn",
        }
    }
}

pub(crate) fn validate_subscription(
    request: PushSubscriptionRequest,
) -> Result<ValidatedSubscription, SubscriptionValidationError> {
    let extra_allow_hosts = std::env::var("ZO_REMOTE_PUSH_ALLOW_HOSTS").ok();
    validate_subscription_with_allow_hosts(request, extra_allow_hosts.as_deref())
}

fn validate_subscription_with_allow_hosts(
    request: PushSubscriptionRequest,
    extra_allow_hosts: Option<&str>,
) -> Result<ValidatedSubscription, SubscriptionValidationError> {
    let endpoint = reqwest::Url::parse(&request.endpoint)
        .map_err(|_| SubscriptionValidationError::Malformed)?;
    if endpoint.scheme() != "https" || endpoint.port().is_some_and(|port| port != 443) {
        return Err(SubscriptionValidationError::EndpointNotAllowed);
    }
    let endpoint_host = endpoint
        .host_str()
        .ok_or(SubscriptionValidationError::EndpointNotAllowed)?
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let ip_candidate = endpoint_host.trim_matches(['[', ']']);
    if ip_candidate.parse::<IpAddr>().is_ok()
        || endpoint_host == "localhost"
        || endpoint_host.ends_with(".localhost")
        || !host_is_allowed(&endpoint_host, extra_allow_hosts)
    {
        return Err(SubscriptionValidationError::EndpointNotAllowed);
    }

    let p256dh = URL_SAFE_NO_PAD
        .decode(request.keys.p256dh)
        .map_err(|_| SubscriptionValidationError::Malformed)?;
    if p256dh.len() != 65 || p256dh.first() != Some(&0x04) || !valid_p256_public_key(&p256dh) {
        return Err(SubscriptionValidationError::Malformed);
    }
    let auth = URL_SAFE_NO_PAD
        .decode(request.keys.auth)
        .map_err(|_| SubscriptionValidationError::Malformed)?;
    let auth: [u8; 16] = auth
        .try_into()
        .map_err(|_| SubscriptionValidationError::Malformed)?;

    Ok(ValidatedSubscription {
        endpoint,
        endpoint_host,
        p256dh,
        auth,
    })
}

fn host_is_allowed(host: &str, extra_allow_hosts: Option<&str>) -> bool {
    BUILTIN_ALLOW_HOSTS
        .iter()
        .copied()
        .chain(
            extra_allow_hosts
                .into_iter()
                .flat_map(|hosts| hosts.split(','))
                .map(str::trim)
                .filter(|suffix| !suffix.is_empty()),
        )
        .any(|suffix| {
            let suffix = suffix
                .trim_matches('.')
                .to_ascii_lowercase();
            !suffix.is_empty()
                && (host == suffix || host.ends_with(&format!(".{suffix}")))
        })
}

fn valid_p256_public_key(public_key: &[u8]) -> bool {
    let rng = SystemRandom::new();
    let Ok(private_key) = agreement::EphemeralPrivateKey::generate(&agreement::ECDH_P256, &rng)
    else {
        return false;
    };
    let peer = agreement::UnparsedPublicKey::new(&agreement::ECDH_P256, public_key);
    agreement::agree_ephemeral(private_key, &peer, |_| ()).is_ok()
}

fn endpoint_audience(endpoint: &reqwest::Url) -> String {
    let host = endpoint.host_str().unwrap_or_default();
    match endpoint.port() {
        Some(port) => format!("{}://{host}:{port}", endpoint.scheme()),
        None => format!("{}://{host}", endpoint.scheme()),
    }
}

fn push_disabled_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "off"
    )
}

fn encrypt(
    subscription: &ValidatedSubscription,
    plaintext: &[u8],
) -> Result<Vec<u8>, ()> {
    let rng = SystemRandom::new();
    let private_key = agreement::EphemeralPrivateKey::generate(&agreement::ECDH_P256, &rng)
        .map_err(|_| ())?;
    let public_key = private_key.compute_public_key().map_err(|_| ())?;
    let mut salt = [0_u8; 16];
    rng.fill(&mut salt).map_err(|_| ())?;
    let peer = agreement::UnparsedPublicKey::new(&agreement::ECDH_P256, &subscription.p256dh);
    agreement::agree_ephemeral(private_key, &peer, |ecdh_secret| {
        encrypt_with_materials(
            ecdh_secret,
            &subscription.auth,
            &salt,
            public_key.as_ref(),
            &subscription.p256dh,
            plaintext,
        )
    })
    .map_err(|_| ())
}

pub(crate) fn encrypt_with_materials(
    ecdh_secret: &[u8],
    auth_secret: &[u8],
    salt: &[u8],
    as_public: &[u8],
    ua_public: &[u8],
    plaintext: &[u8],
) -> Vec<u8> {
    assert_eq!(auth_secret.len(), 16);
    assert_eq!(salt.len(), 16);
    assert_eq!(as_public.len(), 65);
    assert_eq!(ua_public.len(), 65);
    assert!(plaintext.len() <= MAX_PLAINTEXT_BYTES);

    let mut key_info = b"WebPush: info\0".to_vec();
    key_info.extend_from_slice(ua_public);
    key_info.extend_from_slice(as_public);
    let ikm = hkdf_expand::<32>(auth_secret, ecdh_secret, &key_info);
    let cek = hkdf_expand::<16>(salt, &ikm, b"Content-Encoding: aes128gcm\0");
    let nonce = hkdf_expand::<12>(salt, &ikm, b"Content-Encoding: nonce\0");

    let mut ciphertext = Vec::with_capacity(plaintext.len() + 1 + aead::MAX_TAG_LEN);
    ciphertext.extend_from_slice(plaintext);
    ciphertext.push(0x02);
    let key = UnboundKey::new(&aead::AES_128_GCM, &cek)
        .expect("HKDF produces an AES-128-GCM key");
    LessSafeKey::new(key)
        .seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::empty(),
            &mut ciphertext,
        )
        .expect("Web Push plaintext is within AES-GCM limits");

    let mut body = Vec::with_capacity(21 + as_public.len() + ciphertext.len());
    body.extend_from_slice(salt);
    body.extend_from_slice(&RECORD_SIZE.to_be_bytes());
    body.push(u8::try_from(as_public.len()).expect("VAPID public key length fits in u8"));
    body.extend_from_slice(as_public);
    body.extend_from_slice(&ciphertext);
    body
}

fn hkdf_expand<const N: usize>(salt: &[u8], ikm: &[u8], info: &[u8]) -> [u8; N] {
    let salt = hkdf::Salt::new(hkdf::HKDF_SHA256, salt);
    let prk = salt.extract(ikm);
    let info_parts = [info];
    let okm = prk
        .expand(&info_parts, HkdfLength(N))
        .expect("requested HKDF output is within SHA-256 limits");
    let mut output = [0_u8; N];
    okm.fill(&mut output)
        .expect("HKDF output buffer has the requested length");
    output
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use ring::agreement;
    use ring::rand::SystemRandom;
    use ring::signature;

    use super::{
        PushReason, PushSubscriptionRequest, SubscriptionValidationError, VapidKey,
        encrypt_with_materials, push_disabled_value, validate_subscription_with_allow_hosts,
    };

    fn decode(value: &str) -> Vec<u8> {
        URL_SAFE_NO_PAD.decode(value).expect("RFC base64url value")
    }

    fn public_key() -> Vec<u8> {
        let private = agreement::EphemeralPrivateKey::generate(
            &agreement::ECDH_P256,
            &SystemRandom::new(),
        )
        .expect("test private key");
        private
            .compute_public_key()
            .expect("test public key")
            .as_ref()
            .to_vec()
    }

    fn request(endpoint: &str) -> PushSubscriptionRequest {
        PushSubscriptionRequest::new(endpoint, &public_key(), &[7_u8; 16])
    }

    #[test]
    fn endpoint_allowlist_is_boundary_checked_and_fail_closed() {
        let rejected = [
            "http://fcm.googleapis.com/send",
            "https://fcm.googleapis.com:8443/send",
            "https://127.0.0.1/send",
            "https://localhost/send",
            "https://evil-fcm.googleapis.com.attacker.example/send",
        ];
        for endpoint in rejected {
            assert_eq!(
                validate_subscription_with_allow_hosts(request(endpoint), None).err(),
                Some(SubscriptionValidationError::EndpointNotAllowed),
                "{endpoint} must be rejected",
            );
        }

        for endpoint in [
            "https://fcm.googleapis.com/send",
            "https://updates.push.services.mozilla.com/send",
            "https://push.apple.com:443/send",
        ] {
            assert!(
                validate_subscription_with_allow_hosts(request(endpoint), None).is_ok(),
                "{endpoint} must be accepted",
            );
        }

        assert!(
            validate_subscription_with_allow_hosts(
                request("https://device.push.example.test/send"),
                Some(" other.example, push.example.test "),
            )
            .is_ok()
        );
    }

    #[test]
    fn subscription_keys_require_valid_p256_and_auth_lengths() {
        let valid_public = public_key();
        let endpoint = "https://fcm.googleapis.com/send";
        let cases = [
            PushSubscriptionRequest::new(endpoint, &valid_public[..64], &[7_u8; 16]),
            PushSubscriptionRequest::new(endpoint, &[0x04_u8; 65], &[7_u8; 16]),
            PushSubscriptionRequest::new(endpoint, &valid_public, &[7_u8; 15]),
        ];
        for request in cases {
            assert_eq!(
                validate_subscription_with_allow_hosts(request, None).err(),
                Some(SubscriptionValidationError::Malformed),
            );
        }
    }

    #[test]
    fn vapid_jwt_has_fixed_claims_and_raw_p256_signature() {
        let vapid = VapidKey::generate().expect("VAPID key");
        let jwt = vapid
            .jwt_at("https://fcm.googleapis.com", 1_700_000_000)
            .expect("VAPID JWT");
        let parts = jwt.split('.').collect::<Vec<_>>();
        assert_eq!(parts.len(), 3);
        let header: serde_json::Value = serde_json::from_slice(&decode(parts[0]))
            .expect("JWT header JSON");
        let claims: serde_json::Value = serde_json::from_slice(&decode(parts[1]))
            .expect("JWT claims JSON");
        assert_eq!(header, serde_json::json!({"typ": "JWT", "alg": "ES256"}));
        assert_eq!(claims["aud"], "https://fcm.googleapis.com");
        assert_eq!(claims["exp"], 1_700_043_200_u64);
        assert_eq!(claims["sub"], "mailto:zo-remote@localhost");
        let signature_bytes = decode(parts[2]);
        assert_eq!(signature_bytes.len(), 64);
        let advertised_public_key = decode(&vapid.public_key_b64);
        signature::UnparsedPublicKey::new(
            &signature::ECDSA_P256_SHA256_FIXED,
            &advertised_public_key,
        )
        .verify(
            format!("{}.{}", parts[0], parts[1]).as_bytes(),
            &signature_bytes,
        )
        .expect("raw ES256 signature verifies");
    }

    #[test]
    fn rfc8291_section_5_vector_matches_full_aes128gcm_body() {
        let plaintext = decode(
            "V2hlbiBJIGdyb3cgdXAsIEkgd2FudCB0byBiZSBhIHdhdGVybWVsb24",
        );
        let ecdh_secret = decode("kyrL1jIIOHEzg3sM2ZWRHDRB62YACZhhSlknJ672kSs");
        let auth_secret = decode("BTBZMqHH6r4Tts7J_aSIgg");
        let salt = decode("DGv6ra1nlYgDCS1FRnbzlw");
        let as_public = decode(concat!(
            "BP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIg",
            "Dll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A8",
        ));
        let ua_public = decode(concat!(
            "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-",
            "JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4",
        ));
        let expected = decode(concat!(
            "DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27ml",
            "mlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPT",
            "pK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQexSgSxsj_Qulcy4a-fN",
        ));

        assert_eq!(
            encrypt_with_materials(
                &ecdh_secret,
                &auth_secret,
                &salt,
                &as_public,
                &ua_public,
                &plaintext,
            ),
            expected,
        );
    }

    #[test]
    fn kill_switch_and_payload_literals_match_the_contract() {
        for value in ["0", "false", "FALSE", " off "] {
            assert!(push_disabled_value(value));
        }
        for value in ["", "1", "true", "disabled"] {
            assert!(!push_disabled_value(value));
        }
        assert_eq!(PushReason::Approval.plaintext(), br#"{"reason":"approval"}"#);
        assert_eq!(PushReason::TurnIdle.plaintext(), br#"{"reason":"turn_idle"}"#);
    }
}
