pub struct TokenBucket {
    tokens: f64,
    last_refill: std::time::Instant,
    capacity: f64,
    refill_rate: f64,
}

impl TokenBucket {
    /// Creates a new token bucket with the specified capacity and refill rate.
    /// 
    /// # Arguments
    /// * `capacity` - Maximum number of tokens the bucket can hold
    /// * `refill_rate` - Tokens per second to add to the bucket
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: std::time::Instant::now(),
            capacity,
            refill_rate,
        }
    }

    /// Tries to acquire the specified number of tokens without blocking.
    /// 
    /// # Arguments
    /// * `tokens` - Number of tokens to acquire
    /// 
    /// # Returns
    /// `true` if tokens were acquired, `false` if insufficient tokens available
    pub fn try_acquire(&mut self, tokens: f64) -> bool {
        self._refill();
        if tokens <= self.tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Blocks until the specified number of tokens can be acquired.
    /// 
    /// # Arguments
    /// * `tokens` - Number of tokens to acquire
    pub fn acquire(&mut self, tokens: f64) {
        while !self.try_acquire(tokens) {
            let wait_time = (tokens - self.tokens) / self.refill_rate;
            std::thread::sleep(std::time::Duration::from_millis((wait_time * 1000.0) as u64));
            self._refill();
        }
    }

    /// Internal method to refill tokens based on elapsed time
    fn _refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }
}