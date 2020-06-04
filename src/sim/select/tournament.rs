
// file: tournament.rs
//
// Copyright 2015-2017 The RsGenetic Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::mem::swap;
use std::sync::atomic::Ordering::*;
use std::cmp::max;
use super::*;
use pheno::{Fitness, Phenotype};
use rand::Rng;
use rayon::prelude::*;
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::iter::Iterator;


/// Runs several tournaments, and selects best performing phenotypes from each tournament.
#[derive(Copy, Clone, Debug)]
pub struct TournamentSelector {
    count: usize,
    participants: usize,
    head_to_head_kind: Option<HeadToHeadKind>,
}

/// The kind of tournament to use for head to head tournaments
#[derive(Copy, Clone, Debug)]
pub enum HeadToHeadKind {
    /// A single-elimination tournament
    Elimination,

    /// A swiss-style tournamnet (with random matchups)
    Swiss(usize),
}

// #![feature(vec_remove_item)]
impl TournamentSelector {
    /// Create and return a tournament selector.
    ///
    /// Such a selector runs `count / 2` tournaments, each with `participants` participants.
    /// From each tournament, the best 2 phenotypes are selected, yielding
    /// `count` parents.
    ///
    /// * `count`: must be larger than zero, a multiple of two and less than the population size.
    /// * `participants`: must be larger than one and less than the population size.
    #[deprecated(
        note = "The `TournamentSelector` requires at least 2 participants. This is not enforced
                       by the `new` function. You should use `new_checked` instead.",
        since = "1.7.11"
    )]
    pub fn new(count: usize, participants: usize) -> TournamentSelector {
        TournamentSelector {
            count,
            participants,
            head_to_head_kind: None,
        }
    }

    /// Create and return a tournament selector.
    ///
    /// Such a selector runs `count / 2` tournaments, each with `participants` participants.
    /// From each tournament, the best 2 phenotypes are selected, yielding
    /// `count` parents.
    ///
    /// * `count`: must be larger than zero, a multiple of two and less than the population size.
    /// * `participants`: must be larger than one and less than the population size.
    pub fn new_checked(count: usize, participants: usize) -> Result<TournamentSelector, String> {
        Self::_new(count, participants, None)
    }

    /// Create and return a tournament selector.
    ///
    /// Such a selector runs `count / 2` tournaments, each with `participants` participants.
    /// From each tournament, the best 2 phenotypes are selected, yielding
    /// `count` parents.
    ///
    /// * `count`: must be larger than zero, a multiple of two and less than the population size.
    /// * `participants`: must be larger than one and less than the population size.
    pub fn new_head_to_head(kind: HeadToHeadKind, count: usize, participants: usize) -> Result<TournamentSelector, String> {
        Self::_new(count, participants, Some(kind))
    }

    fn _new(count: usize, participants: usize, head_to_head_kind: Option<HeadToHeadKind>) -> Result<TournamentSelector, String> {
        if count == 0 || count % 2 != 0 || participants < 2 {
            Err(String::from(
                "count must be larger than zero and a multiple of two; participants must be larger than one",
            ))
        } else {
            Ok(TournamentSelector {
                count,
                participants,
                head_to_head_kind: head_to_head_kind
            })
        }
    }

    fn round_robin<T: Phenotype<F>, F: Fitness>(participants: &[&T]) -> usize {
        let mut scores = vec![0.0; participants.len()];
        for p1 in 0..participants.len(){
            for p2 in (p1+1)..participants.len(){
                let matchup = participants[p1].relative_fitness(participants[p2]);
                scores[p1] += matchup;
                scores[p2] -= matchup;
            }
        }
        scores.iter().enumerate().max_by(|(_,x1), (_,x2)| x1.partial_cmp(x2).unwrap()).map(|(i,_x)| i).unwrap()
    }

    fn elimination<T: Phenotype<F>, F: Fitness>(participants: &[&T]) -> usize {
        if participants.len() == 1 {return 0;}
        
        let middle = participants.len() / 2;
        let left = &participants[..middle];
        let right = &participants[middle..];
        
        let finalist1 = Self::elimination(left);
        let finalist2 = Self::elimination(right);

        let mut result = left[finalist1].relative_fitness(right[finalist2]);
        if result == 0.0 {result = {if thread_rng().gen() {1.0} else {-1.0}}};
        if  result > 0.0 {
            finalist1
        } else {
            finalist2 + middle
        }

    }

    fn elimination2<'a, T: Phenotype<F>, F: Fitness>(participants: &[&'a T]) -> (usize, usize, Vec<f64>) {
        
        fn elim<T: Phenotype<F>, F: Fitness>(participants: &[&T], scores: &mut [f64], win_bonus: f64) -> usize {
            if participants.len() == 1 {
                return 0;
            }
            
            let middle = participants.len() / 2;
            let left = &participants[..middle];
            
            let right = & participants[middle..];
            let (left_scores, right_scores) = scores.split_at_mut(middle);
            
            let finalist1 = elim(left, left_scores, win_bonus/2.0);
            let finalist2 = elim(right, right_scores, win_bonus/2.0);

            let mut result = left[finalist1].relative_fitness(right[finalist2]);
            if result == 0.0 {result = {if thread_rng().gen() {1.0} else {-1.0}}};
            if result > 0.0 { left_scores[finalist1] += win_bonus;}
            else if result < 0.0 {right_scores[finalist2] += win_bonus;}
            if  result > 0.0 {
                finalist1
            } else {
                finalist2 + middle
            }
        }

        let mut scores = vec![0.0; participants.len()];
        let mid = scores.len() / 2;
        let best1 = elim(participants, &mut scores[0..mid], participants.len() as f64);
        let best2 = elim(participants, &mut scores[mid..], participants.len() as f64);

        (best1, best2, scores)
    }

    fn generate_round_robin_matchup(teams: usize) -> Vec<Vec<(usize, usize)>>{
        
        fn seq(i: usize, N: usize) -> usize {
            if i < N /2 - 1 { i + 2} else {N + N /2 - i - 1}
        }
        fn rotate_index_left(i: usize, rotation: usize, seq_len: usize) -> usize {
            if i >= rotation {i - rotation} else {seq_len + i - rotation}
        }
        
        fn row1(i: usize, round: usize, N: usize) -> usize {
            if i==0 {1}
            else { seq(rotate_index_left(i - 1,round, N - 1), N) }
        }
        
        fn row2(i: usize, round: usize, N: usize) -> usize {
            seq(rotate_index_left(N - 2 - i, round, N - 1), N)
        }
        
        let mut res = vec![Vec::new(); teams - 1];
        for round in 0..teams-1 {
            for i in 0..teams/2 {
                res[round].push((row1(i,round, teams), row2(i,round, teams)));
            }
        }
        res
    }

    fn swiss_tournament<'a, T: Phenotype<F>, F: Fitness>(participants: &[&'a T], rounds: usize) -> (usize, usize, Vec<f64>) {
        assert!(rounds < participants.len());
        assert!(participants.len() > 1);

        // println!("running random matchups tournament. participants size: {}, count_per_participant: {}", participants.len(), count_per_participant);
        let round_robin = Self::generate_round_robin_matchup(participants.len());
        let picked_rounds = rand::seq::index::sample(&mut thread_rng(), participants.len()-1, rounds);
        
        let mut scores = vec![0.0; participants.len()];
        let mut best = 0; let mut second_best = 0;
        for r in picked_rounds.into_iter(){
            for (i,j) in round_robin[r].iter(){
                let i= i-1; let j= j-1;

                let i_first: bool = rand::random();
                let i_first_mul = if i_first {1.0} else {-1.0};
                let matchup = if i_first {participants[i].relative_fitness(participants[j])}
                              else {participants[j].relative_fitness(participants[i])};
                scores[i] += i_first_mul * matchup;
                scores[j] -= i_first_mul * matchup;

                if scores[i] > scores[second_best] {second_best = i;}
                if scores[second_best] > scores[best] {swap(&mut best, &mut second_best);}
                if scores[j] > scores[second_best] {second_best = j;}
                if scores[second_best] > scores[best] {swap(&mut best, &mut second_best);}
            }
        }

        (best, second_best, scores)

    }

    fn run_tournament_for_kind<'a, T: Phenotype<F>, F: Fitness>(kind: HeadToHeadKind, participants: &[&'a T]) -> (usize, usize, Vec<f64>){
        match kind {
            HeadToHeadKind::Elimination => Self::elimination2(participants),
            HeadToHeadKind::Swiss(rounds) => Self::swiss_tournament(participants, rounds)
        }
    }
}

impl<T, F> Selector<T, F> for TournamentSelector
where
    T: Phenotype<F>,
    F: Fitness,
{
    fn fittest<'a>(&self, population: &'a [T]) -> &'a T {
        if let Some(h2h_kind) = self.head_to_head_kind {
            let res = TournamentSelector::run_tournament_for_kind(h2h_kind, &population.iter().collect::<Vec<_>>()).0;
            &population[res]
        } else {
            population.iter().max_by_key(|f| f.fitness()).unwrap()
        }
    }

    fn select_and_rank<'a>(&self, population: &'a [T]) -> Result<(Parents<&'a T>, Vec<f64>), String> {
        if self.count == 0 || self.count % 2 != 0 || self.count * 2 >= population.len() {
            return Err(format!(
                "Invalid parameter `count`: {}. Should be larger than zero, a \
                 multiple of two and less than half the population size.",
                self.count
            ));
        }
        if self.participants == 0 || self.participants >= population.len() {
            return Err(format!(
                "Invalid parameter `participants`: {}. Should be larger than \
                 zero and less than the population size.",
                self.participants
            ));
        }

        let mut fitness_scores = Vec::new();
        for _ in 0..population.len() { fitness_scores.push(atomic::Atomic::<f64>::new(0.0))};

        let result = (0..(self.count/2)).into_par_iter().map(|_| {
            let mut rng = ::rand::thread_rng();
            let mut tournament: Vec<&T> = Vec::with_capacity(self.participants);
            let mut indices_in_tournamnet = Vec::with_capacity(self.participants);
            for _ in 0..self.participants {
                let index = rng.gen_range(0, population.len());
                tournament.push(&population[index]);
                indices_in_tournamnet.push(index);
            }
            // println!("... participants selected!");

            if let Some(h2h_kind) = self.head_to_head_kind {
                let (fst, snd, tournament_scores) = TournamentSelector::run_tournament_for_kind(h2h_kind, &tournament);

                for ind in 0..tournament.len(){
                    let score_cell = &fitness_scores[indices_in_tournamnet[ind]];
                    score_cell.store( tournament_scores[ind] + score_cell.load(Acquire), Relaxed);
                }
                (tournament[fst], tournament[snd])
            } else {
                tournament.sort_by(|x, y| y.fitness().cmp(&x.fitness()));
                (tournament[0], tournament[1])
            }
        }).collect::<Vec<_>>();
        let fitness_scores = fitness_scores.iter().map(|s| s.load(Relaxed)).collect::<Vec<_>>();
        Ok((result, fitness_scores))
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use sim::select::*;
    use test::Test;

    #[test]
    fn test_count_zero() {
        let selector = TournamentSelector::new(0, 1);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_participants_zero() {
        let selector = TournamentSelector::new(2, 0);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_count_odd() {
        let selector = TournamentSelector::new(5, 1);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_count_too_large() {
        let selector = TournamentSelector::new(100, 1);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_participants_too_large() {
        let selector = TournamentSelector::new(2, 100);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_result_size() {
        let selector = TournamentSelector::new(20, 5);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert_eq!(20, selector.select(&population).unwrap().len() * 2);
    }

    #[test]
    fn test_new_checked_count_0() {
        let selector = TournamentSelector::new_checked(0, 2);
        assert!(selector.is_err());
    }

    #[test]
    fn test_new_checked_count_odd() {
        let selector = TournamentSelector::new_checked(3, 2);
        assert!(selector.is_err());
    }

    #[test]
    fn test_new_checked_participants() {
        let selector = TournamentSelector::new_checked(2, 1);
        assert!(selector.is_err());
    }

    #[test]
    fn test_new_checked_ok() {
        let selector = TournamentSelector::new_checked(2, 2);
        assert!(selector.is_ok());
    }
    
    #[test]
    fn generate_round_robin_matchup() {
        let res = TournamentSelector::generate_round_robin_matchup(4);
        println!("{:?}", res);
    }
}
