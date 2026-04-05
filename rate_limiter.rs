use std::time::{Duration, Instant};

/// A token bucket rate limiter that controls throughput via lazy-refilled tokens.
pub struct TokenBucket {
    /// Maximum tokens the bucket can hold.
    capacity: u64,
    /// Current available tokens (capped at `capacity` after each refill).
    tokens: f64,
    /// Token generation rate in tokens per second.
    refill_rate: f64,
    /// Timestamp of the last refill calculation.
    last_refill: Instant,
}

impl TokenBucket {
    /// Creates a bucket at full `capacity` with the given `refill_rate` (tokens/sec).
    pub fn with_capacity_and_rate(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Lazily refills tokens based on elapsed time, capping at capacity.
    fn lazy_refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate)
            .min(self.capacity as f64);
        self.last_refill = Instant::now();
    }

    /// Non-blocking attempt to consume `count` tokens. Returns success status.
    pub fn try_acquire(&mut self, count: u64) -> bool {
        self.lazy_refill();
        if self.tokens >= count as f64 {
            self.tokens -= count as f64;
            true
        } else {
            false
        }
    }

    /// Predicts the wait duration until `count` tokens are available.
    /// Returns `Duration::ZERO` if immediately satisfiable.
    pub fn time_until_available(&self, count: u64) -> Duration {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        let current = (self.tokens + elapsed * self.refill_rate)
            .min(self.capacity as f64);
        if current >= count as f64 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64((count as f64 - current) / self.refill_rate)
        }
    }
}
