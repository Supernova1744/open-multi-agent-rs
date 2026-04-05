use crate::task::create_task;
use crate::task::scheduler::{SchedulingStrategy, Scheduler};
use crate::types::{AgentConfig, Task, TaskStatus};

fn agent(name: &str, system_prompt: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        model: "test".to_string(),
        system_prompt: Some(system_prompt.to_string()),
        ..Default::default()
    }
}

fn pending_task(id: &str, title: &str, desc: &str) -> Task {
    let mut t = create_task(title, desc, None, vec![]);
    t.id = id.to_string();
    t
}

fn in_progress_task(id: &str, assignee: &str) -> Task {
    let mut t = create_task(id, id, Some(assignee.to_string()), vec![]);
    t.id = id.to_string();
    t.status = TaskStatus::InProgress;
    t
}

// -------------------------------------------------------------------------
// Empty guards
// -------------------------------------------------------------------------

#[test]
fn no_agents_returns_empty() {
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let tasks = vec![pending_task("t1", "T1", "do something")];
    let result = s.schedule(&tasks, &[]);
    assert!(result.is_empty());
}

#[test]
fn no_pending_tasks_returns_empty() {
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let agents = vec![agent("alice", "developer")];
    let result = s.schedule(&[], &agents);
    assert!(result.is_empty());
}

// -------------------------------------------------------------------------
// RoundRobin
// -------------------------------------------------------------------------

#[test]
fn round_robin_cycles_through_agents() {
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let agents = vec![agent("alice", "a"), agent("bob", "b")];
    let tasks = vec![
        pending_task("t1", "T1", "d"),
        pending_task("t2", "T2", "d"),
        pending_task("t3", "T3", "d"),
    ];
    let result = s.schedule(&tasks, &agents);
    let assigned: Vec<String> = result.iter().map(|(_, a)| a.clone()).collect();
    // With 2 agents and 3 tasks: alice, bob, alice
    assert_eq!(assigned[0], "alice");
    assert_eq!(assigned[1], "bob");
    assert_eq!(assigned[2], "alice");
}

#[test]
fn round_robin_single_agent_gets_all() {
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let agents = vec![agent("only", "a")];
    let tasks = vec![
        pending_task("t1", "T1", "d"),
        pending_task("t2", "T2", "d"),
    ];
    let result = s.schedule(&tasks, &agents);
    assert!(result.iter().all(|(_, a)| a == "only"));
}

#[test]
fn round_robin_skips_already_assigned() {
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let agents = vec![agent("alice", "a"), agent("bob", "b")];

    // t1 already has assignee, t2 does not
    let mut t1 = pending_task("t1", "T1", "d");
    t1.assignee = Some("alice".to_string());
    let t2 = pending_task("t2", "T2", "d");

    let result = s.schedule(&[t1, t2], &agents);
    // Only t2 should be scheduled
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, "t2");
}

// -------------------------------------------------------------------------
// LeastBusy
// -------------------------------------------------------------------------

#[test]
fn least_busy_prefers_idle_agent() {
    let mut s = Scheduler::new(SchedulingStrategy::LeastBusy);
    let agents = vec![agent("alice", "a"), agent("bob", "b")];

    // bob already has one in-progress task
    let existing = in_progress_task("old", "bob");
    let new_task = pending_task("t1", "T1", "d");

    let all_tasks = vec![existing, new_task.clone()];
    let result = s.schedule(&all_tasks, &agents);
    // alice is idle so should get it
    assert_eq!(result[0].1, "alice");
}

#[test]
fn least_busy_balances_load() {
    let mut s = Scheduler::new(SchedulingStrategy::LeastBusy);
    let agents = vec![agent("a1", "dev"), agent("a2", "dev")];
    let tasks = vec![
        pending_task("t1", "T1", "d"),
        pending_task("t2", "T2", "d"),
    ];
    let result = s.schedule(&tasks, &agents);
    // Both agents should each get one task
    let names: std::collections::HashSet<String> = result.iter().map(|(_, a)| a.clone()).collect();
    assert_eq!(names.len(), 2);
}

// -------------------------------------------------------------------------
// CapabilityMatch
// -------------------------------------------------------------------------

#[test]
fn capability_match_assigns_by_keyword() {
    let mut s = Scheduler::new(SchedulingStrategy::CapabilityMatch);
    let agents = vec![
        agent("rustdev", "You write Rust code and systems programming"),
        agent("pydev", "You write Python scripts and data analysis"),
    ];
    let tasks = vec![pending_task("t1", "Write Rust code", "Implement a Rust parser")];
    let result = s.schedule(&tasks, &agents);
    assert_eq!(result[0].1, "rustdev");
}

#[test]
fn capability_match_falls_back_when_no_match() {
    let mut s = Scheduler::new(SchedulingStrategy::CapabilityMatch);
    let agents = vec![agent("alice", "a"), agent("bob", "b")];
    let tasks = vec![pending_task("t1", "Xyz", "Zyx")];
    let result = s.schedule(&tasks, &agents);
    // Should still assign something
    assert_eq!(result.len(), 1);
}

// -------------------------------------------------------------------------
// DependencyFirst
// -------------------------------------------------------------------------

#[test]
fn dependency_first_prioritises_blocking_tasks() {
    let mut s = Scheduler::new(SchedulingStrategy::DependencyFirst);
    let agents = vec![agent("a1", "dev"), agent("a2", "dev")];

    // t1 blocks t2 and t3 (2 dependents); t4 blocks nothing
    let t1 = pending_task("t1", "Critical", "blocks others");
    let mut t2 = create_task("t2", "d", None, vec!["t1".to_string()]);
    t2.id = "t2".to_string();
    let mut t3 = create_task("t3", "d", None, vec!["t1".to_string()]);
    t3.id = "t3".to_string();
    let t4 = pending_task("t4", "Leaf", "no deps, no dependents");

    let all = vec![t1.clone(), t2, t3, t4.clone()];
    let result = s.schedule(&all, &agents);

    // t1 should be first because it has 2 blocked dependents
    assert_eq!(result[0].0, "t1");
}

// -------------------------------------------------------------------------
// Determinism over multiple calls
// -------------------------------------------------------------------------

#[test]
fn round_robin_cursor_advances_across_calls() {
    let mut s = Scheduler::new(SchedulingStrategy::RoundRobin);
    let agents = vec![agent("a", "x"), agent("b", "y")];

    let t1 = pending_task("t1", "T1", "d");
    let r1 = s.schedule(&[t1], &agents);
    assert_eq!(r1[0].1, "a"); // first call: cursor=0 → a

    let t2 = pending_task("t2", "T2", "d");
    let r2 = s.schedule(&[t2], &agents);
    assert_eq!(r2[0].1, "b"); // second call: cursor=1 → b
}
