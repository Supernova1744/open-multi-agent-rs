use crate::types::{AgentConfig, Task, TaskStatus};

/// Scheduling strategies for mapping tasks to agents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulingStrategy {
    RoundRobin,
    LeastBusy,
    CapabilityMatch,
    DependencyFirst,
}

pub struct Scheduler {
    strategy: SchedulingStrategy,
    round_robin_cursor: usize,
}

impl Scheduler {
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Scheduler {
            strategy,
            round_robin_cursor: 0,
        }
    }

    /// Assign unassigned pending tasks to agents.
    /// Returns a Vec of (task_id, agent_name) pairs.
    pub fn schedule(&mut self, tasks: &[Task], agents: &[AgentConfig]) -> Vec<(String, String)> {
        if agents.is_empty() {
            return Vec::new();
        }

        let unassigned: Vec<&Task> = tasks.iter()
            .filter(|t| t.status == TaskStatus::Pending && t.assignee.is_none())
            .collect();

        match self.strategy {
            SchedulingStrategy::RoundRobin => self.round_robin(&unassigned, agents),
            SchedulingStrategy::LeastBusy => self.least_busy(&unassigned, agents, tasks),
            SchedulingStrategy::CapabilityMatch => self.capability_match(&unassigned, agents),
            SchedulingStrategy::DependencyFirst => self.dependency_first(&unassigned, agents, tasks),
        }
    }

    fn round_robin(&mut self, unassigned: &[&Task], agents: &[AgentConfig]) -> Vec<(String, String)> {
        let mut result = Vec::new();
        for task in unassigned {
            let agent = &agents[self.round_robin_cursor % agents.len()];
            result.push((task.id.clone(), agent.name.clone()));
            self.round_robin_cursor = (self.round_robin_cursor + 1) % agents.len();
        }
        result
    }

    fn least_busy(&self, unassigned: &[&Task], agents: &[AgentConfig], all_tasks: &[Task]) -> Vec<(String, String)> {
        let mut load: std::collections::HashMap<String, usize> =
            agents.iter().map(|a| (a.name.clone(), 0)).collect();

        for task in all_tasks {
            if task.status == TaskStatus::InProgress {
                if let Some(assignee) = &task.assignee {
                    *load.entry(assignee.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut result = Vec::new();
        for task in unassigned {
            let Some(best) = agents.iter()
                .min_by_key(|a| load.get(&a.name).copied().unwrap_or(0))
            else {
                continue; // no agents available — skip this task
            };
            result.push((task.id.clone(), best.name.clone()));
            *load.entry(best.name.clone()).or_insert(0) += 1;
        }
        result
    }

    fn capability_match(&self, unassigned: &[&Task], agents: &[AgentConfig]) -> Vec<(String, String)> {
        let mut result = Vec::new();
        for task in unassigned {
            let task_text = format!("{} {}", task.title, task.description).to_lowercase();
            let best = agents.iter().max_by_key(|a| {
                let agent_text = format!("{} {}", a.name, a.system_prompt.as_deref().unwrap_or("")).to_lowercase();
                task_text.split_whitespace()
                    .filter(|w| w.len() > 3 && agent_text.contains(*w))
                    .count()
            }).unwrap_or(&agents[0]);
            result.push((task.id.clone(), best.name.clone()));
        }
        result
    }

    fn dependency_first(&mut self, unassigned: &[&Task], agents: &[AgentConfig], all_tasks: &[Task]) -> Vec<(String, String)> {
        let mut ranked: Vec<&Task> = unassigned.to_vec();
        ranked.sort_by(|a, b| {
            count_blocked_dependents(&b.id, all_tasks)
                .cmp(&count_blocked_dependents(&a.id, all_tasks))
        });

        let mut result = Vec::new();
        for task in ranked {
            let agent = &agents[self.round_robin_cursor % agents.len()];
            result.push((task.id.clone(), agent.name.clone()));
            self.round_robin_cursor = (self.round_robin_cursor + 1) % agents.len();
        }
        result
    }
}

fn count_blocked_dependents(task_id: &str, all_tasks: &[Task]) -> usize {
    let mut visited = std::collections::HashSet::new();
    let mut queue = vec![task_id.to_string()];

    // Build reverse adjacency.
    let mut dependents: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for t in all_tasks {
        for dep in &t.depends_on {
            dependents.entry(dep.clone()).or_default().push(t.id.clone());
        }
    }

    while let Some(current) = queue.first().cloned() {
        queue.remove(0);
        for dep_id in dependents.get(&current).cloned().unwrap_or_default() {
            if !visited.contains(&dep_id) {
                visited.insert(dep_id.clone());
                queue.push(dep_id);
            }
        }
    }
    visited.len()
}

#[cfg(test)]
mod tests {
    include!("scheduler_tests.rs");
}
