# Guide: MessageBus

`MessageBus` is a lightweight, in-process pub/sub system that lets agents send
messages to each other without going through the LLM. It is useful for coordinating
workflows, broadcasting status updates, or implementing human-in-the-loop patterns.

## Creating a bus

```rust
use open_multi_agent::messaging::MessageBus;

let bus = MessageBus::new();
```

`MessageBus` is `Clone` — all clones share the same underlying state. Pass clones
freely to different threads or async tasks.

## Point-to-point messaging

```rust
// Agent "planner" sends to agent "worker"
let msg = bus.send("planner", "worker", "Please summarise chapter 3.");

println!("Message ID: {}", msg.id);
println!("Sent at: {}", msg.timestamp);
```

## Broadcast

```rust
// Sends to all agents except the sender
bus.broadcast("coordinator", "Pipeline starting. Stand by.");
```

## Reading the inbox

```rust
// Unread messages addressed to "worker"
let unread = bus.get_unread("worker");
for msg in &unread {
    println!("[{}→{}] {}", msg.from, msg.to, msg.content);
}

// Mark as read (prevents them showing in get_unread again)
let ids: Vec<String> = unread.iter().map(|m| m.id.clone()).collect();
bus.mark_read("worker", &ids);

// All messages (read + unread)
let all = bus.get_all("worker");
```

## Getting a conversation history

```rust
let history = bus.get_conversation("planner", "worker");
for msg in history {
    println!("[{}] {}: {}", msg.timestamp, msg.from, msg.content);
}
```

## Subscriptions (push model)

```rust
let unsubscribe = bus.subscribe("worker", |msg| {
    println!("[worker received] from={} content={}", msg.from, msg.content);
});

// Later, to stop receiving messages:
unsubscribe();
```

The callback is called synchronously whenever a message is delivered to `"worker"`.
It runs on the thread/task that called `bus.send` or `bus.broadcast`.

## Example: multi-agent coordination

```rust
use std::sync::Arc;
use open_multi_agent::messaging::MessageBus;

let bus = MessageBus::new();
let bus_worker = bus.clone();

// Worker subscribes before coordinator sends
let _unsub = bus.subscribe("worker", |msg| {
    println!("Worker got task: {}", msg.content);
});

// Coordinator sends tasks
bus.send("coordinator", "worker", "Analyse dataset A");
bus.send("coordinator", "worker", "Analyse dataset B");

// Worker reads its inbox
let tasks = bus_worker.get_unread("worker");
println!("Worker has {} tasks", tasks.len());

// Worker replies when done
bus_worker.send("worker", "coordinator", "Done with A");
bus_worker.send("worker", "coordinator", "Done with B");

// Coordinator checks replies
let replies = bus.get_unread("coordinator");
println!("Coordinator received {} replies", replies.len());
```

## Thread safety

All methods take `&self` and are safe to call from multiple threads simultaneously.
The internal state is protected by a `std::sync::Mutex`.

## `Message` struct

```rust
pub struct Message {
    pub id: String,             // UUID v4
    pub from: String,           // Sender agent name
    pub to: String,             // Recipient agent name or "*" for broadcast
    pub content: String,
    pub timestamp: DateTime<Utc>,
}
```

## MessageBus vs. SharedMemory

| | `MessageBus` | `SharedMemory` |
|---|---|---|
| Communication style | Push (send/subscribe) | Pull (write/read) |
| Persistence | In-memory only | Backed by `MemoryStore` |
| Typical use | Notifications, handshakes | Passing results between pipeline stages |
| Reading | `get_unread`, `mark_read` | `read(agent, key)` |
