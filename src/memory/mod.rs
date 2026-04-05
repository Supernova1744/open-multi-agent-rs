use chrono::Utc;
use std::collections::HashMap;
use tokio::sync::Mutex;
use std::sync::Arc;

use crate::types::MemoryEntry;

// ---------------------------------------------------------------------------
// MemoryStore trait
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync {
    async fn get(&self, key: &str) -> Option<MemoryEntry>;
    async fn set(&self, key: &str, value: &str, metadata: Option<HashMap<String, serde_json::Value>>);
    async fn list(&self) -> Vec<MemoryEntry>;
    async fn delete(&self, key: &str);
    async fn clear(&self);
}

// ---------------------------------------------------------------------------
// InMemoryStore
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct InMemoryStore {
    data: Arc<Mutex<HashMap<String, MemoryEntry>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        InMemoryStore {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl MemoryStore for InMemoryStore {
    async fn get(&self, key: &str) -> Option<MemoryEntry> {
        self.data.lock().await.get(key).cloned()
    }

    async fn set(&self, key: &str, value: &str, metadata: Option<HashMap<String, serde_json::Value>>) {
        let entry = MemoryEntry {
            key: key.to_string(),
            value: value.to_string(),
            metadata,
            created_at: Utc::now(),
        };
        self.data.lock().await.insert(key.to_string(), entry);
    }

    async fn list(&self) -> Vec<MemoryEntry> {
        self.data.lock().await.values().cloned().collect()
    }

    async fn delete(&self, key: &str) {
        self.data.lock().await.remove(key);
    }

    async fn clear(&self) {
        self.data.lock().await.clear();
    }
}

// ---------------------------------------------------------------------------
// SharedMemory — namespaced wrapper for cross-agent communication
// ---------------------------------------------------------------------------

pub struct SharedMemory {
    store: Arc<dyn MemoryStore>,
}

impl SharedMemory {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        SharedMemory { store }
    }

    fn namespace(agent_name: &str, key: &str) -> String {
        format!("{}/{}", agent_name, key)
    }

    pub async fn write(&self, agent_name: &str, key: &str, value: &str) {
        let ns_key = Self::namespace(agent_name, key);
        self.store.set(&ns_key, value, None).await;
    }

    pub async fn read(&self, agent_name: &str, key: &str) -> Option<MemoryEntry> {
        let ns_key = Self::namespace(agent_name, key);
        self.store.get(&ns_key).await
    }

    pub async fn read_all(&self) -> Vec<MemoryEntry> {
        self.store.list().await
    }

    /// Format all memory entries as a markdown string for injection into prompts.
    pub async fn to_markdown(&self) -> String {
        let entries = self.store.list().await;
        if entries.is_empty() {
            return String::new();
        }

        let mut lines = vec!["## Shared Memory\n".to_string()];
        for entry in entries {
            lines.push(format!("**{}**: {}", entry.key, entry.value));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_store_set_and_get() {
        let store = InMemoryStore::new();
        store.set("key1", "value1", None).await;
        let entry = store.get("key1").await;
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().value, "value1");
    }

    #[tokio::test]
    async fn in_memory_store_get_missing() {
        let store = InMemoryStore::new();
        assert!(store.get("missing").await.is_none());
    }

    #[tokio::test]
    async fn in_memory_store_overwrite() {
        let store = InMemoryStore::new();
        store.set("k", "v1", None).await;
        store.set("k", "v2", None).await;
        assert_eq!(store.get("k").await.unwrap().value, "v2");
    }

    #[tokio::test]
    async fn in_memory_store_delete() {
        let store = InMemoryStore::new();
        store.set("k", "v", None).await;
        store.delete("k").await;
        assert!(store.get("k").await.is_none());
    }

    #[tokio::test]
    async fn in_memory_store_delete_missing_is_noop() {
        let store = InMemoryStore::new();
        store.delete("nonexistent").await; // must not panic
    }

    #[tokio::test]
    async fn in_memory_store_clear() {
        let store = InMemoryStore::new();
        store.set("a", "1", None).await;
        store.set("b", "2", None).await;
        store.clear().await;
        assert!(store.list().await.is_empty());
    }

    #[tokio::test]
    async fn in_memory_store_list() {
        let store = InMemoryStore::new();
        store.set("a", "1", None).await;
        store.set("b", "2", None).await;
        let entries = store.list().await;
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn in_memory_store_metadata() {
        let store = InMemoryStore::new();
        let mut meta = std::collections::HashMap::new();
        meta.insert("source".to_string(), serde_json::json!("agent-1"));
        store.set("k", "v", Some(meta)).await;
        let entry = store.get("k").await.unwrap();
        assert!(entry.metadata.is_some());
        assert_eq!(entry.metadata.unwrap()["source"], serde_json::json!("agent-1"));
    }

    // -------------------------------------------------------------------------
    // SharedMemory
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn shared_memory_namespaces_keys() {
        let store = Arc::new(InMemoryStore::new());
        let sm = SharedMemory::new(Arc::clone(&store) as Arc<dyn MemoryStore>);
        sm.write("agent1", "result", "output A").await;
        sm.write("agent2", "result", "output B").await;

        let a = sm.read("agent1", "result").await.unwrap();
        let b = sm.read("agent2", "result").await.unwrap();
        assert_eq!(a.value, "output A");
        assert_eq!(b.value, "output B");
    }

    #[tokio::test]
    async fn shared_memory_read_missing() {
        let store = Arc::new(InMemoryStore::new());
        let sm = SharedMemory::new(Arc::clone(&store) as Arc<dyn MemoryStore>);
        assert!(sm.read("nobody", "key").await.is_none());
    }

    #[tokio::test]
    async fn shared_memory_read_all() {
        let store = Arc::new(InMemoryStore::new());
        let sm = SharedMemory::new(Arc::clone(&store) as Arc<dyn MemoryStore>);
        sm.write("a1", "k", "v1").await;
        sm.write("a2", "k", "v2").await;
        let all = sm.read_all().await;
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn shared_memory_to_markdown_empty() {
        let store = Arc::new(InMemoryStore::new());
        let sm = SharedMemory::new(Arc::clone(&store) as Arc<dyn MemoryStore>);
        assert!(sm.to_markdown().await.is_empty());
    }

    #[tokio::test]
    async fn shared_memory_to_markdown_contains_entries() {
        let store = Arc::new(InMemoryStore::new());
        let sm = SharedMemory::new(Arc::clone(&store) as Arc<dyn MemoryStore>);
        sm.write("agent", "task1", "great result").await;
        let md = sm.to_markdown().await;
        assert!(md.contains("great result"));
        assert!(md.contains("agent/task1"));
    }
}
