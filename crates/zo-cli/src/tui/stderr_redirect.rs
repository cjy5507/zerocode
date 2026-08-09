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
use std::time::{SystemTime, UNIX_EPOCH};

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
    /// 스탬프 리더 스레드의 종료 신호. 스레드가 끝나면 송신단이 drop 되어
    /// 이 수신단이 `Disconnected` 를 즉시 돌려준다 — 복원 직후 로그의 꼬리를
    /// 결정적으로 기다리는 수단. `None` 은 파이프 없이(스탬프 없이) 파일에
    /// 직접 붙은 폴백 경로.
    #[cfg(unix)]
    drain: Option<std::sync::mpsc::Receiver<()>>,
}

/// 복원 후 스탬프 리더가 꼬리를 비울 때까지 기다리는 상한.
///
/// 무한 join 이 아니라 상한인 이유: fd 2 는 자식 프로세스가 상속한다. 백그라운드
/// Bash 가 살아 있으면 파이프 쓰기단이 닫히지 않아 EOF 가 영영 오지 않고, join
/// 이었다면 zo 가 종료에서 멈춘다. 상한을 두면 최악이라도 이만큼만 늦어진다.
#[cfg(unix)]
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(150);

/// 최대 stderr 로그 파일 크기 제한 (10MB).
pub const MAX_STDERR_LOG_BYTES: u64 = 10 * 1024 * 1024;

/// 스탬프 없이 흘려보낼 최대 한 줄 길이. 서브프로세스가 개행 없는
/// 블롭을 stderr 로 쏟아도 리더의 버퍼가 무한히 자라지 않도록 강제 분할.
const MAX_STAMPED_LINE_BYTES: usize = 64 * 1024;

/// `[HH:MM:SSZ <pid>] ` — 한 줄 앞에 붙는 고정폭 스탬프.
///
/// UTC 로 찍는다. 로컬 시간은 timezone DB 조회가 필요한데 이 워크스페이스는
/// `unsafe_code = "forbid"` 라 `localtime_r` 을 부를 수 없고, 그것만을 위해
/// 새 의존성을 들이지 않았다. 정렬(ordering)과 귀속(attribution) — 이 스탬프의
/// 목적 — 은 UTC 로 온전히 달성되고, `Z` 접미사가 로컬 시간이 아님을 명시한다.
///
/// 날짜를 넣지 않는 이유: 로그는 10MB 로테이션이라 하루를 넘기는 경우가 드물고,
/// 폭이 좁을수록 본문이 앞으로 나온다.
fn stamp_prefix(now: SystemTime, pid: u32) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let day = secs % 86_400;
    let (hour, minute, second) = (day / 3600, (day % 3600) / 60, day % 60);
    format!("[{hour:02}:{minute:02}:{second:02}Z {pid}] ")
}

/// 지정된 파일이 상한 크기 이상이면 기존 파일명을 `.1`로 변경하여 회전시킵니다.
fn rotate_if_oversized(path: &Path, max_bytes: u64) {
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.len() >= max_bytes {
            let mut dest = path.to_path_buf();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                dest.set_extension(format!("{ext}.1"));
            } else {
                dest.set_extension("1");
            }
            let _ = std::fs::rename(path, dest);
        }
    }
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
        rotate_if_oversized(&log_path, MAX_STDERR_LOG_BYTES);
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        #[cfg(unix)]
        let drain = unix_activate(&log_file)?;

        Ok(Self {
            _log_file: log_file,
            log_path,
            #[cfg(unix)]
            drain,
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
        // 복원이 파이프 쓰기단의 마지막 참조를 닫으므로 리더는 EOF 를 보고
        // 남은 줄을 비운 뒤 종료한다. 여기서 (상한을 두고) 기다려야 복원
        // 직후 로그를 읽는 쪽 — 패닉 직후 사용자, 테스트 — 이 잘린 꼬리를
        // 보지 않는다. 송신단이 drop 되면 `Disconnected` 로 즉시 깨어난다.
        #[cfg(unix)]
        if let Some(drain) = self.drain.take() {
            let _ = drain.recv_timeout(DRAIN_TIMEOUT);
        }
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

/// fd 2 로 들어온 바이트를 줄 단위로 재조립해 스탬프를 붙여 로그에 쓴다.
///
/// 파이프가 필요한 이유: `eprintln!` 한 번이 항상 한 번의 `write` 는 아니다.
/// Rust 의 `Stderr` 는 무버퍼라 포맷 조각마다 syscall 이 나가므로, write 단위로
/// 접두사를 붙이면 한 줄 중간에 스탬프가 박힌다. 줄 경계는 이 쪽에서만 알 수 있다.
///
/// 이 루프는 **절대 패닉하지 않고, EOF 전에는 절대 종료하지 않는다**. 리더가 먼저
/// 죽으면 파이프가 차서 `eprintln!` 이 블록되거나(프리즈) EPIPE 로 패닉하므로,
/// 쓰기 실패는 삼키고 읽기만은 계속한다.
#[cfg(unix)]
fn spawn_stamping_reader(
    read_end: std::os::fd::OwnedFd,
    mut log_file: File,
) -> Option<std::sync::mpsc::Receiver<()>> {
    use std::io::{Read, Write};

    // 안전한 변환 — `File: From<OwnedFd>` 는 safe 라 `unsafe` 정책을 지킨다.
    let mut pipe = File::from(read_end);
    let pid = std::process::id();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();

    std::thread::Builder::new()
        .name("zo-stderr-stamp".to_string())
        .spawn(move || {
            // 스레드가 끝날 때 함께 drop 되어 수신단을 깨운다.
            let _done = done_tx;
            let mut pending: Vec<u8> = Vec::with_capacity(256);
            let mut chunk = [0u8; 8192];
            let emit = move |line: &[u8], log: &mut File| {
                let mut out = stamp_prefix(SystemTime::now(), pid).into_bytes();
                out.extend_from_slice(line);
                out.push(b'\n');
                // 스탬프+본문을 한 번의 write 로 내보낸다. O_APPEND 의 write 는
                // 원자적이므로 여러 zo 프로세스가 한 파일을 공유해도 줄이
                // 서로의 중간에 끼어들지 않는다 — 오늘의 조각난 로그도 함께 낫는다.
                let _ = log.write_all(&out);
            };
            loop {
                let read = match pipe.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                };
                for &byte in &chunk[..read] {
                    if byte == b'\n' {
                        emit(&pending, &mut log_file);
                        pending.clear();
                    } else {
                        pending.push(byte);
                        if pending.len() >= MAX_STAMPED_LINE_BYTES {
                            emit(&pending, &mut log_file);
                            pending.clear();
                        }
                    }
                }
            }
            // 개행 없이 끝난 꼬리도 잃지 않는다.
            if !pending.is_empty() {
                emit(&pending, &mut log_file);
            }
        })
        .ok()
        .map(|_handle| done_rx)
}

#[cfg(unix)]
fn unix_activate(log_file: &File) -> io::Result<Option<std::sync::mpsc::Receiver<()>>> {
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

    // fd 2 의 새 목적지: 스탬프 리더의 파이프. 파이프를 못 만들면 예전대로
    // 파일에 직접 붙인다 — 스탬프는 진단 편의고, redirect 자체는 alt-screen
    // 보호라는 더 중요한 계약이라 절대 실패해선 안 된다.
    let stamped_target = unistd::pipe().ok().and_then(|(read_end, write_end)| {
        log_file
            .try_clone()
            .ok()
            .map(|clone| (read_end, write_end, clone))
    });
    let mut drain = None;
    let redirect_result = match stamped_target {
        Some((read_end, write_end, log_clone)) => {
            let result = unistd::dup2(write_end.as_raw_fd(), STDERR_FILENO);
            // dup2 이후 fd 2 가 쓰기단의 유일한 참조가 되도록 원본을 닫는다.
            // 이래야 fd 2 가 복원될 때 리더가 EOF 를 보고 꼬리를 비운다.
            drop(write_end);
            if result.is_ok() {
                drain = spawn_stamping_reader(read_end, log_clone);
            }
            result
        }
        None => unistd::dup2(log_file.as_raw_fd(), STDERR_FILENO),
    };
    if let Err(err) = redirect_result {
        // backup fd 누수 방지 — close 후 sentinel 해제.
        let _ = unistd::close(backup);
        SAVED_STDERR_FD.store(-1, Ordering::Release);
        return Err(io::Error::from(err));
    }

    SAVED_STDERR_FD.store(backup, Ordering::Release);
    Ok(drain)
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

    /// redirect 는 process-wide 싱글턴이라 두 번째 동시 활성화는
    /// `AlreadyExists` 로 거절된다. "활성화하는 테스트는 하나뿐" 이라는
    /// 주석상의 약속에 기대면 테스트가 하나 늘어나는 순간 깨지므로,
    /// 활성화하는 테스트끼리는 이 락으로 직렬화한다.
    #[cfg(unix)]
    static REDIRECT_ACTIVATION: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// 앞선 테스트가 패닉해 락이 poison 돼도 다음 테스트는 계속 돌아야 한다 —
    /// 이 락이 지키는 것은 데이터가 아니라 활성화 순서뿐이다.
    #[cfg(unix)]
    fn serialize_activation() -> std::sync::MutexGuard<'static, ()> {
        REDIRECT_ACTIVATION
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

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
        // 본 모듈은 process-wide singleton 백업이라 활성화가 겹치면
        // 두 번째가 `AlreadyExists` 로 거절된다 — [`serialize_activation`].
        //
        let _serialized = serialize_activation();
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

    /// 스탬프는 고정폭이고 PID 를 들고 있어야 한다 — 여러 zo 프로세스가 한
    /// 파일에 섞여 쓰는 상황에서 줄을 귀속시키는 유일한 단서.
    #[test]
    fn stamp_prefix_carries_a_fixed_width_clock_and_the_pid() {
        // 1970-01-02 03:04:05 UTC → 하루 경계를 넘겨도 시각만 남는다.
        let at = UNIX_EPOCH + std::time::Duration::from_secs(86_400 + 3 * 3600 + 4 * 60 + 5);
        assert_eq!(stamp_prefix(at, 57_085), "[03:04:05Z 57085] ");
        // 한 자리 시/분/초는 0 으로 채워져 폭이 흔들리지 않는다.
        let early = UNIX_EPOCH + std::time::Duration::from_secs(9);
        assert_eq!(stamp_prefix(early, 7), "[00:00:09Z 7] ");
    }

    /// 진짜 회귀 대상: 로그의 모든 줄이 시각+PID 를 달고 나오되 `[zo]` 태그는
    /// 그대로 남아 기존 grep 이 계속 맞아야 한다. 그리고 한 번의 `eprintln!` 이
    /// 여러 write 로 쪼개져도 스탬프는 줄당 하나여야 한다 (write 단위가 아니라
    /// 줄 단위로 재조립한다는 계약).
    #[cfg(unix)]
    #[test]
    fn every_log_line_is_stamped_with_time_and_pid_without_losing_the_zo_tag() {
        let _serialized = serialize_activation();
        let tmp = temp_dir("stamp");
        let log_path = tmp.join("logs").join("zo.log");

        let guard = StderrRedirectGuard::activate(&log_path).expect("activate");
        // libtest 의 출력 캡처를 우회하려면 raw fd write 여야 한다 (위 테스트의
        // 주석 참조). 한 줄을 일부러 세 번에 나눠 써서 write 단위 접두사와
        // 줄 단위 접두사를 구별한다.
        for fragment in [
            "[zo] gpt stream stalled".as_bytes(),
            " (timeout); restarting".as_bytes(),
            " in 1.0s\n".as_bytes(),
        ] {
            nix::unistd::write(io::stderr(), fragment).expect("raw stderr write");
        }
        nix::unistd::write(io::stderr(), b"[boot] total=163ms\n").expect("raw stderr write");
        let _ = io::stderr().flush();
        // drop 이 복원 + 꼬리 배출을 끝내므로 이후 읽기는 결정적이다.
        guard.restore().expect("restore");

        let mut buf = String::new();
        File::open(&log_path)
            .expect("open log")
            .read_to_string(&mut buf)
            .expect("read log");

        let lines: Vec<&str> = buf.lines().filter(|line| !line.is_empty()).collect();
        assert_eq!(lines.len(), 2, "one stamp per line, not per write: {buf:?}");

        let pid = std::process::id();
        for line in &lines {
            assert!(
                line.starts_with('['),
                "every line leads with the stamp: {line:?}"
            );
            assert!(
                line.contains(&format!("Z {pid}] ")),
                "every line carries this process id: {line:?}"
            );
        }
        assert!(
            lines[0].ends_with("[zo] gpt stream stalled (timeout); restarting in 1.0s"),
            "the fragmented line is reassembled whole behind the stamp: {:?}",
            lines[0]
        );
        assert!(
            lines[0].contains("[zo]"),
            "the bracketed tag existing greps rely on must survive: {:?}",
            lines[0]
        );
        assert!(
            lines[1].contains("[boot] total=163ms"),
            "boot lines ride the same sink and are stamped too: {:?}",
            lines[1]
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

    #[test]
    fn test_rotate_if_oversized() {
        let tmp = temp_dir("rotate");
        let log_path = tmp.join("zo.log");

        // case 1: does not exist -> noop
        rotate_if_oversized(&log_path, 10);
        assert!(!log_path.exists());

        // case 2: less than max_bytes -> not rotated
        std::fs::write(&log_path, "12345").unwrap();
        rotate_if_oversized(&log_path, 10);
        assert!(log_path.exists());
        assert_eq!(std::fs::read_to_string(&log_path).unwrap(), "12345");

        // case 3: greater than or equal to max_bytes -> rotated
        rotate_if_oversized(&log_path, 5);
        assert!(!log_path.exists());
        let rotated = tmp.join("zo.log.1");
        assert!(rotated.exists());
        assert_eq!(std::fs::read_to_string(&rotated).unwrap(), "12345");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
