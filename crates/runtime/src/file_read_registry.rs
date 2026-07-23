//! 대화(세션) 스코프 read-before-edit 레지스트리 — CC(Claude Code) 패리티.
//!
//! `path → (mtime, len, content-hash)` 스냅샷을 기록해, 모델 툴 경로의
//! `edit_file`/`write_file`이 (a) 이 대화에서 읽은 적 없는 기존 파일이나
//! (b) 마지막 읽기 이후 디스크에서 바뀐 파일(사용자/외부 도구 편집)을
//! 조용히 덮어쓰는 것을 막는 기준 상태를 제공한다.
//!
//! ## 스코프 — 절대 프로세스 전역 금지
//!
//! 이 레지스트리는 **대화 단위 상태**로만 소유되어야 한다(`ToolContext`의
//! `Arc<Mutex<…>>` 필드). 프로세스 전역 캐시는 서브에이전트와 공유되어
//! 다른 대화의 읽기 기록이 이 대화의 가드를 통과시키는 오염을 일으킨다 —
//! 이 저장소에서 프로세스 전역 reasoning-replay 캐시가 서브에이전트와
//! 공유돼 대형 토큰 누수 사고가 난 전력이 있는 클래스다. 서브에이전트는
//! fresh `ToolContext::new()`를 받으므로 레지스트리도 자동으로 격리된다.
//!
//! ## 판정 규칙
//!
//! hash 불일치가 권위(authoritative)이고 mtime+len 일치는 빠른 경로다:
//! mtime과 len이 기록과 같으면 재해싱 없이 Fresh, 다르면 실제 바이트를
//! 다시 해싱해 내용이 정말 달라졌는지 확인한다(`touch`나 무변경 재작성은
//! Fresh로 판정 — 재읽기 강요 없음).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// `edit_file`/`write_file` 가드가 소비하는 신선도 판정.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFreshness {
    /// 디스크에 파일이 없음 — 신규 생성(`write_file`) 경로는 가드 면제.
    Missing,
    /// 파일은 존재하지만 이 대화에서 읽힌 적 없음 → 먼저 `read_file` 필요.
    NeverRead,
    /// 마지막 읽기 이후 디스크 내용이 달라짐(사용자 또는 외부 도구) →
    /// 재읽기 후 최신 내용 기준으로 다시 편집해야 함.
    ModifiedSinceRead,
    /// 마지막으로 관측(read/write/edit)한 상태 그대로 — 편집 허용.
    Fresh,
}

/// 파일 하나의 관측 스냅샷. `mtime`/`len`은 빠른 경로, `hash`가 권위.
#[derive(Debug, Clone, Copy)]
struct FileSnapshot {
    mtime: Option<SystemTime>,
    len: u64,
    hash: u64,
}

/// 대화 스코프 `path → 스냅샷` 맵. 키는 canonicalize된 절대 경로라
/// 상대/절대·심링크 표기가 달라도 같은 파일은 같은 엔트리로 수렴한다.
#[derive(Debug, Default)]
pub struct FileReadRegistry {
    entries: HashMap<PathBuf, FileSnapshot>,
}

impl FileReadRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 기록·조회가 같은 키를 쓰도록 canonicalize로 정규화. 파일이 방금
    /// 삭제돼 canonicalize가 실패해도 부모는 대개 존재하므로 부모
    /// canonicalize + 파일명으로 키를 안정화한다(`normalize_path_allow_missing`
    /// 미러) — 삭제 직후의 엔트리 제거가 기록 때와 같은 키에 닿는다.
    fn key(path: &Path) -> PathBuf {
        if let Ok(canonical) = path.canonicalize() {
            return canonical;
        }
        if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
            if let Ok(canonical_parent) = parent.canonicalize() {
                return canonical_parent.join(name);
            }
        }
        path.to_path_buf()
    }

    /// 현재 디스크 상태를 관측해 등재/갱신한다. `read_file` 성공 시,
    /// 그리고 `edit_file`/`write_file` 성공 시(자기 자신이 만든 변경은
    /// 신선한 것으로 간주) 호출된다. 읽기 실패(삭제·권한)는 stale 엔트리를
    /// 남기지 않도록 제거한다 — 다음 편집이 [`FileFreshness::NeverRead`]로 재읽기를 유도.
    pub fn record_from_disk(&mut self, path: &Path) {
        let key = Self::key(path);
        match fs::read(&key) {
            Ok(bytes) => {
                let mtime = fs::metadata(&key).and_then(|meta| meta.modified()).ok();
                self.entries.insert(
                    key,
                    FileSnapshot {
                        mtime,
                        len: bytes.len() as u64,
                        hash: fnv1a64(&bytes),
                    },
                );
            }
            Err(_) => {
                self.entries.remove(&key);
            }
        }
    }

    /// 디스크의 현재 상태를 마지막 관측 스냅샷과 비교해 신선도를 판정한다.
    /// mtime+len 일치 → 재해싱 없이 `Fresh`(빠른 경로); 불일치 → 바이트를
    /// 다시 해싱해 내용 기준으로 판정(hash가 권위).
    #[must_use]
    pub fn check(&self, path: &Path) -> FileFreshness {
        let key = Self::key(path);
        let Ok(meta) = fs::metadata(&key) else {
            return FileFreshness::Missing;
        };
        let Some(snapshot) = self.entries.get(&key) else {
            return FileFreshness::NeverRead;
        };
        if snapshot.mtime.is_some()
            && meta.modified().ok() == snapshot.mtime
            && meta.len() == snapshot.len
        {
            return FileFreshness::Fresh;
        }
        match fs::read(&key) {
            Ok(bytes) => {
                if bytes.len() as u64 == snapshot.len && fnv1a64(&bytes) == snapshot.hash {
                    // 내용은 그대로(touch·무변경 재작성) — 재읽기 강요 없음.
                    FileFreshness::Fresh
                } else {
                    FileFreshness::ModifiedSinceRead
                }
            }
            // 방금 존재를 확인했는데 읽을 수 없게 됐다(권한 변경 등) —
            // 보수적으로 재읽기를 강제한다.
            Err(_) => FileFreshness::ModifiedSinceRead,
        }
    }

    /// 등재된 파일 수 — 테스트/진단용.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 모든 기록 제거 — 대화 리셋(/clear류) 배선용.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// FNV-1a 64-bit — 짧은 의존성 없는 내용 해시. 보안 목적이 아니라
/// "마지막 읽기 이후 바뀌었는가"의 변경 감지용이므로 충분하다.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{FileFreshness, FileReadRegistry};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("zo-read-registry-{name}-{unique}"))
    }

    #[test]
    fn missing_file_is_exempt_and_unread_file_is_never_read() {
        let registry = FileReadRegistry::new();
        let path = temp_path("missing.txt");
        assert_eq!(registry.check(&path), FileFreshness::Missing);

        std::fs::write(&path, "content").expect("seed");
        assert_eq!(registry.check(&path), FileFreshness::NeverRead);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recorded_file_is_fresh_until_content_changes() {
        let mut registry = FileReadRegistry::new();
        let path = temp_path("fresh.txt");
        std::fs::write(&path, "v1").expect("seed");
        registry.record_from_disk(&path);
        assert_eq!(registry.check(&path), FileFreshness::Fresh);

        std::fs::write(&path, "v2").expect("external modify");
        assert_eq!(registry.check(&path), FileFreshness::ModifiedSinceRead);

        // 재관측하면 다시 Fresh — read_file 재호출에 해당.
        registry.record_from_disk(&path);
        assert_eq!(registry.check(&path), FileFreshness::Fresh);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn same_content_rewrite_stays_fresh_because_hash_is_authoritative() {
        // mtime이 바뀌어도(무변경 재작성/touch) 내용 hash가 같으면 Fresh —
        // mtime은 빠른 경로일 뿐, 판정 권위는 hash다.
        let mut registry = FileReadRegistry::new();
        let path = temp_path("touch.txt");
        std::fs::write(&path, "same bytes").expect("seed");
        registry.record_from_disk(&path);

        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&path, "same bytes").expect("rewrite identical content");
        assert_eq!(registry.check(&path), FileFreshness::Fresh);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn deleted_file_recording_drops_the_entry() {
        let mut registry = FileReadRegistry::new();
        let path = temp_path("deleted.txt");
        std::fs::write(&path, "v1").expect("seed");
        registry.record_from_disk(&path);
        assert_eq!(registry.len(), 1);

        std::fs::remove_file(&path).expect("delete");
        registry.record_from_disk(&path);
        assert!(registry.is_empty(), "stale entry must not survive deletion");
        // 삭제된 파일 자체는 Missing — 재생성(write_file)은 면제 경로.
        assert_eq!(registry.check(&path), FileFreshness::Missing);
    }

    #[test]
    fn relative_and_canonical_paths_share_one_entry() {
        let mut registry = FileReadRegistry::new();
        let dir = temp_path("canon-dir");
        std::fs::create_dir_all(&dir).expect("dir");
        let file = dir.join("a.txt");
        std::fs::write(&file, "v1").expect("seed");

        // 비정규 표기(`dir/./a.txt`)로 기록해도 canonicalize 키로 수렴한다.
        let dotted = dir.join(".").join("a.txt");
        registry.record_from_disk(&dotted);
        assert_eq!(registry.check(&file), FileFreshness::Fresh);
        assert_eq!(registry.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
