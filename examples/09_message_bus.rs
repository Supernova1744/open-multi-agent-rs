/// Example 09 — MessageBus: Agent-to-Agent Communication
///
/// Demonstrates the pub/sub MessageBus independently of the orchestrator.
/// Simulates two agents exchanging messages and one agent broadcasting
/// an announcement to all subscribers.
///
/// MessageBus is clone-safe: all clones share the same underlying state.
///
/// Run:
///   cargo run --example 09_message_bus
use open_multi_agent::messaging::MessageBus;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let bus = MessageBus::new();

    // ── Subscribe before sending so we capture all messages ─────────────────
    let alice_inbox: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let bob_inbox: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));

    let alice_inbox_ref = Arc::clone(&alice_inbox);
    let bob_inbox_ref = Arc::clone(&bob_inbox);

    let _unsub_alice = bus.subscribe("alice", move |msg| {
        alice_inbox_ref
            .lock()
            .unwrap()
            .push(format!("[from {}] {}", msg.from, msg.content));
    });

    let _unsub_bob = bus.subscribe("bob", move |msg| {
        bob_inbox_ref
            .lock()
            .unwrap()
            .push(format!("[from {}] {}", msg.from, msg.content));
    });

    // ── Point-to-point messages ──────────────────────────────────────────────
    println!("Sending point-to-point messages...");
    bus.send("alice", "bob", "Hey Bob, I finished the research.");
    bus.send("bob", "alice", "Great! I'll start writing now.");
    bus.send("alice", "bob", "Let me know if you need more facts.");

    // ── Broadcast ────────────────────────────────────────────────────────────
    println!("Broadcasting a system announcement...");
    bus.broadcast("system", "Pipeline will restart in 60 seconds.");

    // ── Read via get_unread (polling style) ──────────────────────────────────
    println!("\n── Polling unread messages ─────────────────────");

    let bob_msgs = bus.get_unread("bob");
    println!("Bob's unread messages ({}):", bob_msgs.len());
    let bob_ids: Vec<String> = bob_msgs.iter().map(|m| m.id.clone()).collect();
    for msg in &bob_msgs {
        println!("  [from {}] {}", msg.from, msg.content);
    }
    bus.mark_read("bob", &bob_ids);

    let alice_msgs = bus.get_unread("alice");
    println!("\nAlice's unread messages ({}):", alice_msgs.len());
    let alice_ids: Vec<String> = alice_msgs.iter().map(|m| m.id.clone()).collect();
    for msg in &alice_msgs {
        println!("  [from {}] {}", msg.from, msg.content);
    }
    bus.mark_read("alice", &alice_ids);

    // ── Subscriber callbacks ─────────────────────────────────────────────────
    println!("\n── Subscriber callbacks ────────────────────────");
    println!("Alice's inbox via subscribe:");
    for entry in alice_inbox.lock().unwrap().iter() {
        println!("  {}", entry);
    }

    println!("\nBob's inbox via subscribe:");
    for entry in bob_inbox.lock().unwrap().iter() {
        println!("  {}", entry);
    }

    // ── Conversation history ─────────────────────────────────────────────────
    println!("\n── Conversation: alice ↔ bob ───────────────────");
    let convo = bus.get_conversation("alice", "bob");
    for msg in convo {
        println!("  [{} → {}] {}", msg.from, msg.to, msg.content);
    }

    // ── Unsubscribe ──────────────────────────────────────────────────────────
    // _unsub_alice and _unsub_bob drop here (or call them explicitly).
    println!("\n(subscriptions cleaned up on drop)");
}
