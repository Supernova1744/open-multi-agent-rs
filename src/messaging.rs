/// Inter-agent message bus.
///
/// Provides a lightweight pub/sub system so agents can exchange typed messages
/// without direct references to each other. All messages are retained in memory
/// for replay and audit; read-state is tracked per recipient.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A single message exchanged between agents (or broadcast to all).
#[derive(Debug, Clone)]
pub struct Message {
    /// Stable UUID for this message.
    pub id: String,
    /// Name of the sending agent.
    pub from: String,
    /// Recipient agent name, or `"*"` for broadcast to all except sender.
    pub to: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

fn is_addressed_to(message: &Message, agent_name: &str) -> bool {
    if message.to == "*" {
        message.from != agent_name
    } else {
        message.to == agent_name
    }
}

// ---------------------------------------------------------------------------
// Subscriber
// ---------------------------------------------------------------------------

type SubscriberFn = Arc<dyn Fn(Message) + Send + Sync>;

struct Subscriber {
    id: u64,
    callback: SubscriberFn,
}

// ---------------------------------------------------------------------------
// MessageBus
// ---------------------------------------------------------------------------

struct BusState {
    messages: Vec<Message>,
    /// Per-agent set of message IDs already marked as read.
    read_state: HashMap<String, std::collections::HashSet<String>>,
    /// Active subscribers keyed by agent name.
    subscribers: HashMap<String, Vec<Subscriber>>,
    next_sub_id: u64,
}

/// In-memory message bus for inter-agent communication.
///
/// Thread-safe — `clone()` returns a handle to the same underlying bus.
#[derive(Clone)]
pub struct MessageBus {
    state: Arc<Mutex<BusState>>,
}

impl MessageBus {
    pub fn new() -> Self {
        MessageBus {
            state: Arc::new(Mutex::new(BusState {
                messages: Vec::new(),
                read_state: HashMap::new(),
                subscribers: HashMap::new(),
                next_sub_id: 0,
            })),
        }
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Send a message from `from` to `to`.
    pub fn send(&self, from: &str, to: &str, content: &str) -> Message {
        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            from: from.to_string(),
            to: to.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
        };
        self.persist(message.clone());
        message
    }

    /// Broadcast a message from `from` to all other agents (`to == "*"`).
    pub fn broadcast(&self, from: &str, content: &str) -> Message {
        self.send(from, "*", content)
    }

    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------

    /// Returns messages not yet marked as read by `agent_name`.
    pub fn get_unread(&self, agent_name: &str) -> Vec<Message> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let read = state.read_state.get(agent_name);
        state
            .messages
            .iter()
            .filter(|m| {
                is_addressed_to(m, agent_name)
                    && read.map_or(true, |r| !r.contains(&m.id))
            })
            .cloned()
            .collect()
    }

    /// Returns every message (read or unread) addressed to `agent_name`.
    pub fn get_all(&self, agent_name: &str) -> Vec<Message> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .messages
            .iter()
            .filter(|m| is_addressed_to(m, agent_name))
            .cloned()
            .collect()
    }

    /// Mark a set of messages as read for `agent_name`.
    pub fn mark_read(&self, agent_name: &str, message_ids: &[String]) {
        if message_ids.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let read = state
            .read_state
            .entry(agent_name.to_string())
            .or_default();
        for id in message_ids {
            read.insert(id.clone());
        }
    }

    /// Returns all messages exchanged between `agent1` and `agent2` in either direction.
    pub fn get_conversation(&self, agent1: &str, agent2: &str) -> Vec<Message> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .messages
            .iter()
            .filter(|m| {
                (m.from == agent1 && m.to == agent2)
                    || (m.from == agent2 && m.to == agent1)
            })
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Subscriptions
    // -----------------------------------------------------------------------

    /// Subscribe to new messages addressed to `agent_name`.
    ///
    /// Returns an unsubscribe function; calling it is idempotent.
    pub fn subscribe(
        &self,
        agent_name: &str,
        callback: impl Fn(Message) + Send + Sync + 'static,
    ) -> impl FnOnce() {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = state.next_sub_id;
        state.next_sub_id += 1;
        let subs = state
            .subscribers
            .entry(agent_name.to_string())
            .or_default();
        subs.push(Subscriber {
            id,
            callback: Arc::new(callback),
        });

        let bus_state = Arc::clone(&self.state);
        let agent = agent_name.to_string();
        move || {
            let mut s = bus_state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(subs) = s.subscribers.get_mut(&agent) {
                subs.retain(|sub| sub.id != id);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn persist(&self, message: Message) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.messages.push(message.clone());
        // Notify subscribers while holding the lock (callbacks must not call
        // back into the bus to avoid deadlocks — same rule as TS version).
        let callbacks: Vec<SubscriberFn> = if message.to != "*" {
            state
                .subscribers
                .get(&message.to)
                .map(|subs| subs.iter().map(|s| Arc::clone(&s.callback)).collect())
                .unwrap_or_default()
        } else {
            let from = message.from.clone();
            state
                .subscribers
                .iter()
                .filter(|(name, _)| *name != &from)
                .flat_map(|(_, subs)| subs.iter().map(|s| Arc::clone(&s.callback)))
                .collect()
        };
        drop(state); // release lock before calling callbacks
        for cb in callbacks {
            cb(message.clone());
        }
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn send_and_get_all() {
        let bus = MessageBus::new();
        bus.send("alice", "bob", "hello bob");
        bus.send("bob", "alice", "hello alice");
        let bobs = bus.get_all("bob");
        assert_eq!(bobs.len(), 1);
        assert_eq!(bobs[0].content, "hello bob");
    }

    #[test]
    fn broadcast_reaches_all_except_sender() {
        let bus = MessageBus::new();
        bus.broadcast("alice", "everyone listen up");
        let bobs = bus.get_all("bob");
        let charlies = bus.get_all("charlie");
        let alices = bus.get_all("alice");
        assert_eq!(bobs.len(), 1);
        assert_eq!(charlies.len(), 1);
        assert_eq!(alices.len(), 0); // sender doesn't receive own broadcast
    }

    #[test]
    fn mark_read_hides_from_get_unread() {
        let bus = MessageBus::new();
        let msg = bus.send("alice", "bob", "hello");
        assert_eq!(bus.get_unread("bob").len(), 1);
        bus.mark_read("bob", &[msg.id.clone()]);
        assert_eq!(bus.get_unread("bob").len(), 0);
        // get_all still returns it
        assert_eq!(bus.get_all("bob").len(), 1);
    }

    #[test]
    fn get_conversation_bidirectional() {
        let bus = MessageBus::new();
        bus.send("alice", "bob", "msg1");
        bus.send("bob", "alice", "msg2");
        bus.send("alice", "charlie", "other");
        let conv = bus.get_conversation("alice", "bob");
        assert_eq!(conv.len(), 2);
    }

    #[test]
    fn subscribe_receives_messages() {
        let bus = MessageBus::new();
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        let _unsub = bus.subscribe("bob", move |msg| {
            received_clone.lock().unwrap().push(msg.content.clone());
        });
        bus.send("alice", "bob", "ping");
        bus.send("alice", "charlie", "other");
        let r = received.lock().unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], "ping");
    }

    #[test]
    fn unsubscribe_stops_delivery() {
        let bus = MessageBus::new();
        let count = Arc::new(Mutex::new(0u32));
        let count_clone = Arc::clone(&count);
        let unsub = bus.subscribe("bob", move |_| {
            *count_clone.lock().unwrap() += 1;
        });
        bus.send("alice", "bob", "first");
        unsub();
        bus.send("alice", "bob", "second");
        assert_eq!(*count.lock().unwrap(), 1);
    }

    #[test]
    fn multiple_messages_ordering() {
        let bus = MessageBus::new();
        for i in 0..5 {
            bus.send("alice", "bob", &format!("msg{}", i));
        }
        let all = bus.get_all("bob");
        assert_eq!(all.len(), 5);
        assert_eq!(all[0].content, "msg0");
        assert_eq!(all[4].content, "msg4");
    }

    #[test]
    fn bus_clone_shares_state() {
        let bus = MessageBus::new();
        let bus2 = bus.clone();
        bus.send("alice", "bob", "hello");
        assert_eq!(bus2.get_all("bob").len(), 1);
    }
}
