use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::error::Result;

/// Controls how many agent runs execute concurrently.
pub struct AgentPool {
    semaphore: Arc<Semaphore>,
    pub max_concurrent: usize,
}

impl AgentPool {
    pub fn new(max_concurrent: usize) -> Self {
        AgentPool {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// Acquire a slot, run `f`, release the slot.
    pub async fn run<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let _permit = self.semaphore.acquire().await.expect("semaphore closed");
        f().await
    }
}
