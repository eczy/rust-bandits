use super::agent::Agent;
use std::vec::Vec;
use rand::prelude::*;
use rand::thread_rng;
use rand_distr::StandardNormal;
use std::f64::NEG_INFINITY;

pub type Reward = f64;
pub type Action = u64;

pub trait Bandit {
    fn sample(&self) -> Reward;
}

struct GaussianBandit {
    mean: f64
}

impl GaussianBandit {
    pub fn new(mean: f64) -> Self {
        Self { mean }
    }
}

impl Bandit for GaussianBandit {
    fn sample(&self) -> Reward {
        thread_rng().sample::<Reward, _>(StandardNormal) + self.mean
    }
}

struct KArmedBandit<T> where T: Bandit {
    bandits: Vec<Box<T>>,
}

impl<T> KArmedBandit<T> where T: Bandit{
    pub fn new(bandits: Vec<Box<T>>) -> Self {
        Self { bandits }
    }

    pub fn step(&self, a: Action) -> Reward {
        self.bandits[a as usize].sample()
    }
}

pub struct TenArmedTestbed {
    mabs: Vec<KArmedBandit<GaussianBandit>>,
    optimal: Vec<Action>
}

impl TenArmedTestbed {
    pub fn new() -> Self {
        let mut mabs = Vec::with_capacity(2000);
        let mut optimal = Vec::with_capacity(2000);
        for _ in 0..2000 {
            let mut bandits = Vec::with_capacity(10);
            let mut optimal_action = NEG_INFINITY;
            let mut optimal_index = 0;
            for j in 0..10 {
                let mean = thread_rng().sample(StandardNormal);
                bandits.push(Box::new(GaussianBandit::new(mean)));
                if mean > optimal_action {
                    optimal_index = j;
                    optimal_action = mean;
                }
            }
            mabs.push(KArmedBandit::new(bandits));
            optimal.push(optimal_index as Action);
        }
        Self { mabs, optimal }
    }

    pub fn run(&self, agent: &mut impl Agent) -> (Vec<Vec<f64>>, Vec<Vec<bool>>) {
        let mut rewards: Vec<Vec<f64>> = Vec::with_capacity(self.mabs.len());
        let mut optimality: Vec<Vec<bool>> = Vec::with_capacity(self.mabs.len());
        for (i, mab) in self.mabs.iter().enumerate() {
            rewards.push(Vec::with_capacity(1000));
            optimality.push(Vec::with_capacity(1000));
            for _ in 0..1000 {
                let action = agent.act();
                let reward = mab.step(action);
                agent.observe(action, reward);
                rewards[i].push(reward);
                optimality[i].push(action == self.optimal[i]);
            }
        }
        (rewards, optimality)
    }
}