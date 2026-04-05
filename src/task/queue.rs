use chrono::Utc;
use std::collections::HashMap;

use crate::error::{AgentError, Result};
use crate::types::{Task, TaskStatus};
use super::is_task_ready;

// ---------------------------------------------------------------------------
// TaskQueue — dependency-aware mutable task store
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct TaskQueue {
    tasks: HashMap<String, Task>,
}

impl TaskQueue {
    pub fn new() -> Self {
        TaskQueue {
            tasks: HashMap::new(),
        }
    }

    /// Add a task. Promotes to 'blocked' if dependencies aren't all completed yet.
    pub fn add(&mut self, mut task: Task) {
        if !task.depends_on.is_empty() {
            let all: Vec<Task> = self.tasks.values().cloned().collect();
            if !is_task_ready(&task, &all) {
                task.status = TaskStatus::Blocked;
                task.updated_at = Utc::now();
            }
        }
        self.tasks.insert(task.id.clone(), task);
    }

    pub fn add_batch(&mut self, tasks: Vec<Task>) {
        for task in tasks {
            self.add(task);
        }
    }

    pub fn update(&mut self, task_id: &str, status: Option<TaskStatus>, result: Option<String>, assignee: Option<Option<String>>) -> Result<Task> {
        let task = self.tasks.get_mut(task_id)
            .ok_or_else(|| AgentError::TaskNotFound(task_id.to_string()))?;

        if let Some(s) = status {
            task.status = s;
        }
        if let Some(r) = result {
            task.result = Some(r);
        }
        if let Some(a) = assignee {
            task.assignee = a;
        }
        task.updated_at = Utc::now();
        Ok(task.clone())
    }

    pub fn complete(&mut self, task_id: &str, result: Option<String>) -> Result<Task> {
        let completed = self.update(task_id, Some(TaskStatus::Completed), result, None)?;
        self.unblock_dependents(task_id);
        Ok(completed)
    }

    pub fn fail(&mut self, task_id: &str, error: String) -> Result<Task> {
        let failed = self.update(task_id, Some(TaskStatus::Failed), Some(error.clone()), None)?;
        self.cascade_fail(task_id, &error);
        Ok(failed)
    }

    pub fn skip(&mut self, task_id: &str, reason: String) -> Result<Task> {
        let skipped = self.update(task_id, Some(TaskStatus::Skipped), Some(reason.clone()), None)?;
        self.cascade_skip(task_id, &reason);
        Ok(skipped)
    }

    /// Mark all non-terminal tasks as skipped.
    pub fn skip_remaining(&mut self, reason: &str) {
        let ids: Vec<String> = self.tasks.values()
            .filter(|t| !t.status.is_terminal())
            .map(|t| t.id.clone())
            .collect();
        for id in ids {
            let _ = self.update(&id, Some(TaskStatus::Skipped), Some(reason.to_string()), None);
        }
    }

    fn cascade_fail(&mut self, failed_id: &str, error: &str) {
        let dependent_ids: Vec<String> = self.tasks.values()
            .filter(|t| {
                !t.status.is_terminal()
                    && t.depends_on.iter().any(|d| d == failed_id)
            })
            .map(|t| t.id.clone())
            .collect();

        for id in dependent_ids {
            let msg = format!("Cancelled: dependency '{}' failed. {}", failed_id, error);
            let _ = self.update(&id, Some(TaskStatus::Failed), Some(msg.clone()), None);
            self.cascade_fail(&id, &msg);
        }
    }

    fn cascade_skip(&mut self, skipped_id: &str, _reason: &str) {
        let dependent_ids: Vec<String> = self.tasks.values()
            .filter(|t| {
                !t.status.is_terminal()
                    && t.depends_on.iter().any(|d| d == skipped_id)
            })
            .map(|t| t.id.clone())
            .collect();

        for id in dependent_ids {
            let msg = format!("Skipped: dependency '{}' was skipped.", skipped_id);
            let _ = self.update(&id, Some(TaskStatus::Skipped), Some(msg.clone()), None);
            self.cascade_skip(&id, &msg);
        }
    }

    fn unblock_dependents(&mut self, completed_id: &str) {
        let all: Vec<Task> = self.tasks.values().cloned().collect();

        let to_unblock: Vec<String> = all.iter()
            .filter(|t| {
                t.status == TaskStatus::Blocked
                    && t.depends_on.iter().any(|d| d == completed_id)
            })
            .filter(|t| {
                // Check if all deps are now completed.
                let mut check: Task = (*t).clone();
                check.status = TaskStatus::Pending;
                is_task_ready(&check, &all)
            })
            .map(|t| t.id.clone())
            .collect();

        for id in to_unblock {
            if let Some(task) = self.tasks.get_mut(&id) {
                task.status = TaskStatus::Pending;
                task.updated_at = Utc::now();
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Queries
    // ---------------------------------------------------------------------------

    pub fn list(&self) -> Vec<Task> {
        self.tasks.values().cloned().collect()
    }

    pub fn get(&self, task_id: &str) -> Option<&Task> {
        self.tasks.get(task_id)
    }

    pub fn pending_tasks(&self) -> Vec<Task> {
        self.tasks.values()
            .filter(|t| t.status == TaskStatus::Pending)
            .cloned()
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        self.tasks.values().all(|t| t.status.is_terminal())
    }

    pub fn set_assignee(&mut self, task_id: &str, assignee: &str) -> Result<()> {
        let task = self.tasks.get_mut(task_id)
            .ok_or_else(|| AgentError::TaskNotFound(task_id.to_string()))?;
        task.assignee = Some(assignee.to_string());
        task.updated_at = Utc::now();
        Ok(())
    }

    pub fn set_in_progress(&mut self, task_id: &str) -> Result<()> {
        let task = self.tasks.get_mut(task_id)
            .ok_or_else(|| AgentError::TaskNotFound(task_id.to_string()))?;
        task.status = TaskStatus::InProgress;
        task.updated_at = Utc::now();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    include!("queue_tests.rs");
}
