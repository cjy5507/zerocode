//! TUI 활성 구간 동안 `stderr` 를 파일로 우회하여 ratatui alt-screen
//! 위에 `eprintln!`/패닉 로그가 직접 그려지지 않도록 막는 가드.
//!
//! ## 왜 필요한가
//!
//! `crossterm::enable_raw_mode` + `EnterAlternateScreen` 진입 후 ratatui
//! 가 화면을 frame 단위로 그리는 동안, 라이브러리/리트라이 루프 등이
//! `eprintln!` 로 stderr 에 직접 출력하면 cursor 위치를 우회하여 입력
//! 프롬프트 줄과 status 줄에 partial overwrite 가 발생한다 (e.g. retry
//! 메시지가 `(attempt 2/6)/6)` 처럼 잘려 보이는 현상).
//!
//! 해법은 fd 레벨에서 stderr 를 파일로 dup2 — `eprintln!` 은
//! `std::io::stderr().write_all` 로 결국 fd 2 를 호출하므로 fd 자체를
//! 바꿔두면 호출 코드의 변경 없이 일괄 보호된다.
//!
//! ## 라이프사이클
//!
//! ```text
//!   raw mode ON  →  StderrRedirectGuard::activate(log_path)?
//!                   { TUI 실행 구간 }
//!   raw mode OFF →  drop(guard)   ← stderr fd 복원
//! ```
//!
//! 추가로 백업 fd 는 process-wide 글로벌 ([`SAVED_STDERR_FD`]) 에
//! 보관되어 패닉 훅 등 비정상 종료 경로에서도
//! [`restore_stderr_if_active`] 한 번 호출로 복원 가능하다.
//!
//! ## Windows
//!
//! 현재는 Unix 한정 (`nix` dep `cfg(unix)`). Windows 빌드에서는 본 모듈
//! 의 활성화가 noop — alt-screen 진입 이후의 화면 침범은 윈도우
//! 콘솔에선 별도 mechanism 으로 다뤄야 하므로 별도 PR 대상.
//!
//! ## `unsafe` 0 보장
//!
//! workspace 정책 `unsafe_code = "forbid"` 를 그대로 유지. fd 조작은
//! 모두 `nix::unistd::{dup, dup2, close}` 의 safe wrapper 만 사용.
//! raw fd 정수를 글로벌에 보관하지만 `OwnedFd` 변환은 수행하지 않음
//! (`FromRawFd` 가 `unsafe` 라 정책 위반).

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::sync::atomic::{AtomicI32, Ordering};

/// 백업한 원본 stderr fd. `-1` 은 "활성 redirect 없음" 을 의미하고,
/// 활성 중일 때는 `nix::unistd::dup` 가 반환한 fd 번호 (`> 2`) 가
/// 들어 있다.
///
/// 패닉 훅과 정상 종료 양쪽에서 [`restore_stderr_if_active`] 로 접근
/// 가능. 다중 활성화는 가정하지 않는다 — 두 번째 호출은 atomic
/// compare-exchange 단계에서 거절된다.
#[cfg(unix)]
static SAVED_STDERR_FD: AtomicI32 = AtomicI32::new(-1);

/// stderr fd 가 가리키는 원본 위치를 보관하고 `Drop` 시 복원한다.
///
/// 활성 인스턴스가 살아 있는 동안 fd 2 는 호출자가 지정한 로그 파일
/// (기본 `~/.zo/logs/zo.log`) 을 가리킨다. drop 되면 자동 복원.
#[must_use = "stderr redirect 가드는 drop 되면 stderr 가 즉시 복원되므로 변수에 보관해야 한다"]
pub struct StderrRedirectGuard {
    /// 로그 파일 핸들 — guard 가 살아 있는 동안 열려 있으면 충분.
    /// 복원 후 drop 되어 닫힘.
    _log_file: File,
    /// 디버깅/로그 메시지용 경로 보관 (소비자가 사용자에게 안내할 때).
    log_path: PathBuf,
}

impl StderrRedirectGuard {
    /// 활성화. `log_path` 의 부모 디렉토리를 자동 생성하고 append 모드
    /// 로 연 다음 fd 2 를 그 파일로 dup2.
    ///
    /// 호출자는 `enable_raw_mode` / `EnterAlternateScreen` *직전* 에
    /// 호출해야 그 이후 발생하는 모든 stderr 출력이 보호된다.
    ///
    /// Windows 빌드에서는 redirect 가 수행되지 않고, guard 는 단순히
    /// 로그 파일만 보관 (호출자가 `log_path()` 안내 용도로 사용 가능).
    pub fn activate(log_path: impl AsRef<Path>) -> io::Result<Self> {
        let log_path = log_path.as_ref().to_path_buf();
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        #[cfg(unix)]
        unix_activate(&log_file)?;

        Ok(Self {
            _log_file: log_file,
            log_path,
        })
    }

    /// 활성 redirect 의 로그 파일 경로. UI 가 "로그는 X 에 있습니다"
    /// 안내를 띄울 때 사용.
    #[must_use]
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// 명시적 복원. drop 만으로도 동일 효과지만 이쪽은 실패 시 에러를
    /// 반환하여 진단할 수 있다.
    pub fn restore(self) -> io::Result<()> {
        // `self` 가 여기서 소비되어 drop 트리거 → restore_stderr_if_active
        // 가 fd 를 복원. 명시적 호출 형태로 표면화한 결과.
        restore_stderr_if_active()
    }
}

impl Drop for StderrRedirectGuard {
    fn drop(&mut self) {
        // Drop 경로는 실패를 묵살 — 이미 다른 path 에서 종료 중일 수
        // 있고, 여기서 패닉을 발생시키면 double-panic 위험.
        let _ = restore_stderr_if_active();
    }
}

/// 활성 redirect 가 있으면 stderr 를 백업해 둔 원본 fd 로 되돌리고,
/// 백업 fd 는 close. 없으면 noop.
///
/// 패닉 훅, `main()` 의 비정상 종료 경로, 정상 [`StderrRedirectGuard`]
/// drop 모두에서 안전하게 호출 가능 — 멱등이며 fd race 없이 한 번만
/// 복원한다 (`AcqRel` swap 기반).
pub fn restore_stderr_if_active() -> io::Result<()> {
    #[cfg(unix)]
    {
        use nix::libc::STDERR_FILENO;
        use nix::unistd;

        // -1 로 swap — 동시 호출 중 한 명만 실제 fd 복원을 수행.
        let backup = SAVED_STDERR_FD.swap(-1, Ordering::AcqRel);
        if backup < 0 {
            return Ok(());
        }
        // backup fd description 을 fd 2 위에 복제 → stderr 가 원본
        // file description 을 가리키도록 복원.
        unistd::dup2(backup, STDERR_FILENO).map_err(io::Error::from)?;
        // backup fd 자체는 더 이상 필요 없음 — close. dup2 이후 fd 2
        // 가 동일 description 을 가지므로 close 해도 stderr 는 유지.
        unistd::close(backup).map_err(io::Error::from)?;
    }
    Ok(())
}

#[cfg(unix)]
fn unix_activate(log_file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    use nix::libc::STDERR_FILENO;
    use nix::unistd;

    // 동시 활성화 방지 — 첫 진입자만 fd 백업을 저장한다.
    // -1 → 임시 sentinel `-2` 로 교환하여 다른 호출자가 동시 진입해도
    // 두 번째는 여기서 거절된다.
    if SAVED_STDERR_FD
        .compare_exchange(-1, -2, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "stderr redirect already active",
        ));
    }

    // fd 2 백업 — dup 은 새 fd 번호를 할당하고 같은 file description
    // 을 가리키게 한다. nix::unistd::dup 은 safe wrapper.
    let backup = match unistd::dup(STDERR_FILENO) {
        Ok(fd) => fd,
        Err(err) => {
            SAVED_STDERR_FD.store(-1, Ordering::Release);
            return Err(io::Error::from(err));
        }
    };

    // fd 2 를 log_file 로 교체.
    if let Err(err) = unistd::dup2(log_file.as_raw_fd(), STDERR_FILENO) {
        // backup fd 누수 방지 — close 후 sentinel 해제.
        let _ = unistd::close(backup);
        SAVED_STDERR_FD.store(-1, Ordering::Release);
        return Err(io::Error::from(err));
    }

    SAVED_STDERR_FD.store(backup, Ordering::Release);
    Ok(())
}

/// `~/.zo/logs/zo.log` 의 기본 경로. `$ZO_CONFIG_HOME` 가
/// 설정돼 있으면 그 아래로, 아니면 `$HOME/.zo` 기준.
///
/// `runtime::oauth::credentials_home_dir` 와 같은 컨벤션을 따른다 —
/// zo 의 모든 사용자 상태는 `~/.zo/` 또는 `$ZO_CONFIG_HOME`
/// 아래에 모이도록.
#[must_use]
pub fn default_log_path() -> PathBuf {
    core_types::paths::default_config_home()
        .join("logs")
        .join("zo.log")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// 격리된 임시 디렉토리 — 테스트 간 충돌을 막기 위해 process id +
    /// nanos 로 유니크한 디렉토리를 만들고 사용 후 정리한다.
    fn temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "zo-stderr-redirect-{}-{}-{nanos}",
            std::process::id(),
            label
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[cfg(unix)]
    #[test]
    fn redirect_captures_raw_stderr_writes_into_file() {
        // 본 모듈은 process-wide singleton 백업이라 다른 테스트가 동시
        // 활성화하지 않아야 한다. 다른 테스트는 redirect 를 활성화하지
        // 않으므로 단발 호출만으로 안전.
        //
        // 주의: `eprintln!` 은 libtest 의 `set_output_capture()` 가
        // thread-local 로 가로채 self-buffer 에 저장한다 — fd 2 dup2
        // 가 무력화된 것처럼 보이게 만든다. test 환경에서 fd 2 redirect
        // 동작을 검증하려면 raw fd write 로 libtest sink 를 우회한다.
        // 실제 zo 실행 시에는 sink 가 없어 `eprintln!` 도 그대로
        // 잡힌다.
        let tmp = temp_dir("capture");
        let log_path = tmp.join("logs").join("zo.log");

        let guard = StderrRedirectGuard::activate(&log_path).expect("activate");
        let payload = b"captured: hello 42\n";
        nix::unistd::write(io::stderr(), payload).expect("raw stderr write");
        let _ = io::stderr().flush();
        guard.restore().expect("restore");

        let mut buf = String::new();
        File::open(&log_path)
            .expect("open log")
            .read_to_string(&mut buf)
            .expect("read log");
        assert!(
            buf.contains("captured: hello 42"),
            "log should contain raw stderr write, got: {buf:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn restore_when_inactive_is_noop() {
        restore_stderr_if_active().expect("noop ok");
    }

    #[test]
    fn default_log_path_ends_with_logs_zo_log() {
        let path = default_log_path();
        assert!(
            path.ends_with("logs/zo.log"),
            "path should end with logs/zo.log, got: {path:?}"
        );
    }
}
