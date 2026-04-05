pub mod queue;
pub mod scheduler;

use chrono::Utc;
use uuid::Uuid;

use crate::types::{Task, TaskStatus};

// ---------------------------------------------------------------------------
// Task factory
// ---------------------------------------------------------------------------

pub fn create_task(
    title: impl Into<String>,
    description: impl Into<String>,
    assignee: Option<String>,
    depends_on: Vec<String>,
) -> Task {
    let now = Utc::now();
    Task {
        id: Uuid::new_v4().to_string(),
        title: title.into(),
        description: description.into(),
        status: TaskStatus::Pending,
        assignee,
        depends_on,
        result: None,
        created_at: now,
        updated_at: now,
        max_retries: None,
        retry_delay_ms: None,
        retry_backoff: None,
    }
}

/// Check whether all dependencies of `task` are completed.
pub fn is_task_ready(task: &Task, all_tasks: &[Task]) -> bool {
    if task.depends_on.is_empty() {
        return true;
    }
    for dep_id in &task.depends_on {
        let dep = all_tasks.iter().find(|t| &t.id == dep_id);
        match dep {
            None => return false,
            Some(d) => {
                if d.status != TaskStatus::Completed {
                    return false;
                }
            }
        }
    }
    true
}

/// Topologically sort tasks by dependency order.
/// Returns sorted Vec; Err if a cycle is detected.
pub fn topological_sort(tasks: &[Task]) -> Result<Vec<Task>, String> {
    let mut result = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut in_progress = std::collections::HashSet::new();

    fn visit(
        task_id: &str,
        all: &[Task],
        visited: &mut std::collections::HashSet<String>,
        in_progress: &mut std::collections::HashSet<String>,
        result: &mut Vec<Task>,
    ) -> Result<(), String> {
        if visited.contains(task_id) {
            return Ok(());
        }
        if in_progress.contains(task_id) {
            return Err(format!("Cycle detected at task '{}'", task_id));
        }
        in_progress.insert(task_id.to_string());

        if let Some(task) = all.iter().find(|t| t.id == task_id) {
            for dep in &task.depends_on {
                visit(dep, all, visited, in_progress, result)?;
            }
            result.push(task.clone());
        }

        in_progress.remove(task_id);
        visited.insert(task_id.to_string());
        Ok(())
    }

    for task in tasks {
        visit(&task.id, tasks, &mut visited, &mut in_progress, &mut result)?;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TaskStatus;

    fn make_task(id: &str, deps: Vec<&str>) -> Task {
        let mut t = create_task(id, id, None, deps.iter().map(|s| s.to_string()).collect());
        t.id = id.to_string();  // override UUID with readable id
        t
    }

    fn completed(id: &str) -> Task {
        let mut t = make_task(id, vec![]);
        t.status = TaskStatus::Completed;
        t
    }

    #[test]
    fn create_task_defaults() {
        let t = create_task("Do something", "desc", None, vec![]);
        assert_eq!(t.status, TaskStatus::Pending);
        assert!(t.assignee.is_none());
        assert!(t.depends_on.is_empty());
        assert!(!t.id.is_empty());
    }

    #[test]
    fn create_task_with_assignee() {
        let t = create_task("title", "desc", Some("alice".to_string()), vec![]);
        assert_eq!(t.assignee, Some("alice".to_string()));
    }

    #[test]
    fn is_task_ready_no_deps() {
        let t = make_task("a", vec![]);
        assert!(is_task_ready(&t, &[]));
    }

    #[test]
    fn is_task_ready_all_deps_completed() {
        let dep = completed("dep1");
        let t   = make_task("child", vec!["dep1"]);
        assert!(is_task_ready(&t, &[dep]));
    }

    #[test]
    fn is_task_ready_dep_still_pending() {
        let dep = make_task("dep1", vec![]);  // Pending
        let t   = make_task("child", vec!["dep1"]);
        assert!(!is_task_ready(&t, &[dep]));
    }

    #[test]
    fn is_task_ready_dep_missing() {
        let t = make_task("child", vec!["nonexistent"]);
        assert!(!is_task_ready(&t, &[]));
    }

    #[test]
    fn is_task_ready_multiple_deps_one_pending() {
        let d1 = completed("d1");
        let d2 = make_task("d2", vec![]);  // still pending
        let t  = make_task("child", vec!["d1", "d2"]);
        assert!(!is_task_ready(&t, &[d1, d2]));
    }

    #[test]
    fn topological_sort_linear_chain() {
        let a = make_task("a", vec![]);
        let b = make_task("b", vec!["a"]);
        let c = make_task("c", vec!["b"]);

        let sorted = topological_sort(&[c.clone(), a.clone(), b.clone()]).unwrap();
        let ids: Vec<&str> = sorted.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn topological_sort_independent_tasks() {
        let a = make_task("a", vec![]);
        let b = make_task("b", vec![]);

        let sorted = topological_sort(&[a, b]).unwrap();
        assert_eq!(sorted.len(), 2);
    }

    #[test]
    fn topological_sort_diamond() {
        // a → b, a → c, b+c → d
        let a = make_task("a", vec![]);
        let b = make_task("b", vec!["a"]);
        let c = make_task("c", vec!["a"]);
        let d = make_task("d", vec!["b", "c"]);

        let sorted = topological_sort(&[d, b, c, a]).unwrap();
        let pos: std::collections::HashMap<&str, usize> = sorted.iter()
            .enumerate().map(|(i, t)| (t.id.as_str(), i)).collect();
        assert!(pos["a"] < pos["b"]);
        assert!(pos["a"] < pos["c"]);
        assert!(pos["b"] < pos["d"]);
        assert!(pos["c"] < pos["d"]);
    }

    #[test]
    fn topological_sort_detects_cycle() {
        let a = make_task("a", vec!["b"]);
        let b = make_task("b", vec!["a"]);
        let err = topological_sort(&[a, b]);
        assert!(err.is_err());
    }

    #[test]
    fn topological_sort_empty() {
        let sorted = topological_sort(&[]).unwrap();
        assert!(sorted.is_empty());
    }
}
