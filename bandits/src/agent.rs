use super::bandit::{Action, Reward};
use rand::prelude::*;
use rand::thread_rng;
use rand_distr::{Standard, Uniform};
use std::f64::NEG_INFINITY;


pub trait Agent {
    fn act(&self) -> Action;
    fn observe(&mut self, a: Action, r: Reward);
}


pub struct RandomAgent {
    n_arms: u64
}


impl RandomAgent {
    pub fn new(n_arms: u64) -> Self {
        Self { n_arms }
    }
}


impl Agent for RandomAgent {
    fn act(&self) -> Action {
        thread_rng().sample(Uniform::from(0..self.n_arms))
    }
    fn observe(&mut self, _: Action, _: Reward) {}
}

pub struct EpsilonGreedyAgent {
    n_arms: u64,
    epsilon: f64,
    N: Vec<u64>,
    Q: Vec<f64>
}

impl EpsilonGreedyAgent {
    pub fn new(n_arms: u64, epsilon: f64) -> Self {
        let N = vec![0; n_arms as usize];
        let Q = vec![0.; n_arms as usize];
        Self { n_arms, epsilon, N, Q }
    }
}

impl Agent for EpsilonGreedyAgent {
    fn act(&self) -> Action {
        let sample: f64 = thread_rng().sample(Standard);
        if sample < self.epsilon {
            return thread_rng().sample(Uniform::from(0..self.n_arms))
        }
        let mut argmax = 0;
        let mut max = NEG_INFINITY;
        for (i, q) in self.Q.iter().enumerate() {
            if q > &max {
                max = *q;
                argmax = i;
            }
        }
        return argmax as Action;
    }
    fn observe(&mut self, a: Action, r: Reward) {
        self.N[a as usize] += 1;
        self.Q[a as usize] += 1. / self.N[a as usize] as f64 * (r - self.Q[a as usize]);
    }
}