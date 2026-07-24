use std::io::{self, IsTerminal, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const FRAME_INTERVAL: Duration = Duration::from_millis(100);
const FRAMES: [&str; 4] = ["|", "/", "-", "\\"];

/// Shows progress while the synchronous runtime is being assembled.
///
/// Runtime construction must stay on the calling thread because the runtime
/// owns non-`Send` resources such as MCP stdio transports. The loader therefore
/// animates independently and is scoped only around that blocking operation.
pub(crate) struct StartupLoader {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl StartupLoader {
    /// Starts the loader when both standard streams are interactive terminals.
    pub(crate) fn start() -> Self {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return Self::inactive();
        }

        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("zo-startup-loader".to_owned())
            .spawn(move || run(&worker_stop))
            .ok();

        Self { stop, worker }
    }

    fn inactive() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
        }
    }
}

impl Drop for StartupLoader {
    fn drop(&mut self) {
        let Some(worker) = self.worker.take() else {
            return;
        };

        self.stop.store(true, Ordering::Release);
        worker.thread().unpark();
        let _ = worker.join();

        let mut stdout = io::stdout().lock();
        let _ = write!(stdout, "\r\x1b[2K");
        let _ = stdout.flush();
    }
}

fn run(stop: &AtomicBool) {
    let mut frame = 0;

    while !stop.load(Ordering::Acquire) {
        let mut stdout = io::stdout().lock();
        if write!(stdout, "\r\x1b[2K  {} Starting zo…", FRAMES[frame]).is_err()
            || stdout.flush().is_err()
        {
            return;
        }
        drop(stdout);

        frame = (frame + 1) % FRAMES.len();
        thread::park_timeout(FRAME_INTERVAL);
    }
}

#[cfg(test)]
mod tests {
    use super::FRAMES;

    #[test]
    fn spinner_frames_are_stable_and_ascii() {
        assert_eq!(FRAMES, ["|", "/", "-", "\\"]);
    }
}
