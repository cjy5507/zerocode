use std::process::Command;
use std::time::{Duration, Instant};

pub fn notify_if_slow(tool_name: &str, start: Instant, threshold: Duration) {
    let elapsed = start.elapsed();
    if elapsed < threshold {
        return;
    }
    let title = "zo";
    let message = format!("{} completed ({:.1}s)", tool_name, elapsed.as_secs_f64());
    send_notification(title, &message);
}

fn send_notification(title: &str, message: &str) {
    #[cfg(target_os = "macos")]
    {
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            message.replace('"', "\\\""),
            title.replace('"', "\\\""),
        );
        let _ = Command::new("osascript").args(["-e", &script]).output();
    }

    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("notify-send").args([title, message]).output();
    }

    #[cfg(target_os = "windows")]
    {
        let ps_script = format!(
            "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
             $n = New-Object System.Windows.Forms.NotifyIcon; \
             $n.Icon = [System.Drawing.SystemIcons]::Information; \
             $n.Visible = $true; \
             $n.ShowBalloonTip(5000, '{}', '{}', 'Info')",
            title.replace('\'', "''"),
            message.replace('\'', "''"),
        );
        let _ = Command::new("powershell")
            .args(["-Command", &ps_script])
            .output();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_notification_below_threshold() {
        let start = Instant::now();
        notify_if_slow("test_tool", start, Duration::from_secs(60));
    }
}
