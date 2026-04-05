// This file is included into queue.rs via #[cfg(test)] mod tests
// All TaskQueue unit tests live here.

use crate::task::queue::TaskQueue;
use crate::task::create_task;
use crate::types::{Task, TaskStatus};

fn task(id: &str, deps: Vec<&str>) -> Task {
    let mut t = create_task(id, id, None, deps.iter().map(|s| s.to_string()).collect());
    t.id = id.to_string();
    t
}

// -------------------------------------------------------------------------
// add / initial status
// -------------------------------------------------------------------------

#[test]
fn add_no_deps_is_pending() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    assert_eq!(q.get("a").unwrap().status, TaskStatus::Pending);
}

#[test]
fn add_with_unresolved_dep_is_blocked() {
    let mut q = TaskQueue::new();
    q.add(task("b", vec!["a"]));
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Blocked);
}

#[test]
fn add_batch_unresolved_dep_is_blocked() {
    let mut q = TaskQueue::new();
    q.add_batch(vec![task("a", vec![]), task("b", vec!["a"])]);
    assert_eq!(q.get("a").unwrap().status, TaskStatus::Pending);
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Blocked);
}

// -------------------------------------------------------------------------
// complete
// -------------------------------------------------------------------------

#[test]
fn complete_sets_result() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.complete("a", Some("done".to_string())).unwrap();
    let t = q.get("a").unwrap();
    assert_eq!(t.status, TaskStatus::Completed);
    assert_eq!(t.result.as_deref(), Some("done"));
}

#[test]
fn complete_unblocks_single_dependent() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec!["a"]));
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Blocked);
    q.complete("a", None).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Pending);
}

#[test]
fn complete_unblocks_chain() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec!["a"]));
    q.add(task("c", vec!["b"]));

    q.complete("a", None).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Pending);
    assert_eq!(q.get("c").unwrap().status, TaskStatus::Blocked);

    q.complete("b", None).unwrap();
    assert_eq!(q.get("c").unwrap().status, TaskStatus::Pending);
}

#[test]
fn complete_requires_all_deps() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("x", vec![]));
    q.add(task("b", vec!["a", "x"]));

    q.complete("a", None).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Blocked);

    q.complete("x", None).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Pending);
}

#[test]
fn complete_nonexistent_task_errors() {
    let mut q = TaskQueue::new();
    assert!(q.complete("ghost", None).is_err());
}

// -------------------------------------------------------------------------
// fail + cascade
// -------------------------------------------------------------------------

#[test]
fn fail_marks_task_failed() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.fail("a", "boom".to_string()).unwrap();
    assert_eq!(q.get("a").unwrap().status, TaskStatus::Failed);
}

#[test]
fn fail_cascades_to_direct_dependent() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec!["a"]));
    q.fail("a", "err".to_string()).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Failed);
}

#[test]
fn fail_cascades_transitively() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec!["a"]));
    q.add(task("c", vec!["b"]));
    q.fail("a", "err".to_string()).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Failed);
    assert_eq!(q.get("c").unwrap().status, TaskStatus::Failed);
}

#[test]
fn fail_does_not_affect_independent_task() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("independent", vec![]));
    q.fail("a", "err".to_string()).unwrap();
    assert_eq!(q.get("independent").unwrap().status, TaskStatus::Pending);
}

// -------------------------------------------------------------------------
// skip + cascade
// -------------------------------------------------------------------------

#[test]
fn skip_marks_task_skipped() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.skip("a", "not needed".to_string()).unwrap();
    assert_eq!(q.get("a").unwrap().status, TaskStatus::Skipped);
}

#[test]
fn skip_cascades_to_dependents() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec!["a"]));
    q.skip("a", "skipped".to_string()).unwrap();
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Skipped);
}

// -------------------------------------------------------------------------
// skip_remaining
// -------------------------------------------------------------------------

#[test]
fn skip_remaining_marks_all_non_terminal() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec![]));
    q.add(task("c", vec![]));
    q.complete("a", None).unwrap();
    q.skip_remaining("approval rejected");
    assert_eq!(q.get("a").unwrap().status, TaskStatus::Completed);
    assert_eq!(q.get("b").unwrap().status, TaskStatus::Skipped);
    assert_eq!(q.get("c").unwrap().status, TaskStatus::Skipped);
}

// -------------------------------------------------------------------------
// is_complete
// -------------------------------------------------------------------------

#[test]
fn is_complete_empty_queue() {
    let q = TaskQueue::new();
    assert!(q.is_complete());
}

#[test]
fn is_complete_all_terminal() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec![]));
    q.complete("a", None).unwrap();
    q.fail("b", "x".to_string()).unwrap();
    assert!(q.is_complete());
}

#[test]
fn is_complete_with_pending() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    assert!(!q.is_complete());
}

// -------------------------------------------------------------------------
// pending_tasks
// -------------------------------------------------------------------------

#[test]
fn pending_tasks_returns_only_pending() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec!["a"]));
    q.add(task("c", vec![]));

    let pending: Vec<String> = q.pending_tasks().iter().map(|t| t.id.clone()).collect();
    assert!(pending.contains(&"a".to_string()));
    assert!(pending.contains(&"c".to_string()));
    assert!(!pending.contains(&"b".to_string()));
}

// -------------------------------------------------------------------------
// assignee / in_progress
// -------------------------------------------------------------------------

#[test]
fn set_assignee() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.set_assignee("a", "alice").unwrap();
    assert_eq!(q.get("a").unwrap().assignee.as_deref(), Some("alice"));
}

#[test]
fn set_in_progress() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.set_in_progress("a").unwrap();
    assert_eq!(q.get("a").unwrap().status, TaskStatus::InProgress);
}

#[test]
fn set_assignee_nonexistent_errors() {
    let mut q = TaskQueue::new();
    assert!(q.set_assignee("ghost", "bob").is_err());
}

// -------------------------------------------------------------------------
// list
// -------------------------------------------------------------------------

#[test]
fn list_returns_all_tasks() {
    let mut q = TaskQueue::new();
    q.add(task("a", vec![]));
    q.add(task("b", vec![]));
    assert_eq!(q.list().len(), 2);
}
