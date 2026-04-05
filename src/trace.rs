/// Trace emission utilities for the observability layer.
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::{OnTraceFn, TraceEvent};

/// Return current time as Unix epoch milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Safely emit a trace event. Swallows panics so a broken subscriber
/// never crashes agent execution.
pub fn emit_trace(on_trace: &Option<OnTraceFn>, event: TraceEvent) {
    if let Some(f) = on_trace {
        // Catch panics so observability never breaks execution.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(event)));
    }
}

/// Generate a unique run ID for trace correlation.
pub fn generate_run_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn emit_trace_calls_callback() {
        use crate::types::{AgentTrace, TraceEvent, TraceEventBase, TokenUsage};
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        let on_trace: OnTraceFn = Arc::new(move |event| {
            received_clone.lock().unwrap().push(event);
        });
        let base = TraceEventBase {
            run_id: "r1".to_string(),
            start_ms: 0,
            end_ms: 10,
            duration_ms: 10,
            agent: "a".to_string(),
            task_id: None,
        };
        emit_trace(
            &Some(on_trace),
            TraceEvent::Agent(AgentTrace {
                base,
                turns: 1,
                tokens: TokenUsage::default(),
                tool_calls: 0,
            }),
        );
        assert_eq!(received.lock().unwrap().len(), 1);
    }

    #[test]
    fn emit_trace_none_is_noop() {
        // Should not panic when on_trace is None.
        emit_trace(&None, TraceEvent::Agent(crate::types::AgentTrace {
            base: crate::types::TraceEventBase {
                run_id: "r".to_string(),
                start_ms: 0, end_ms: 0, duration_ms: 0,
                agent: "a".to_string(), task_id: None,
            },
            turns: 0,
            tokens: crate::types::TokenUsage::default(),
            tool_calls: 0,
        }));
    }

    #[test]
    fn generate_run_id_is_unique() {
        let a = generate_run_id();
        let b = generate_run_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 36); // UUID format
    }

    #[test]
    fn now_ms_is_nonzero() {
        assert!(now_ms() > 0);
    }
}
