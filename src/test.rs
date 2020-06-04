// file: test.rs
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

// This is a private module containing code used in
// several tests across the library.

use pheno::*;
use std::cmp;
use sim::select::TournamentSelector;
use sim::select::HeadToHeadKind;
use sim::Simulation;
use sim::Builder;

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct MyFitness {
    pub f: i64,
}

impl Fitness for MyFitness {
    fn zero() -> Self {
        MyFitness { f: 0 }
    }

    fn abs_diff(&self, other: &Self) -> Self {
        MyFitness {
            f: (self.f - other.f).abs(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Test {
    pub f: i64,
}

impl Phenotype<MyFitness> for Test {
    fn fitness(&self) -> MyFitness {
        MyFitness { f: self.f.abs() }
    }

    fn crossover(&self, t: &Test) -> Test {
        Test {
            f: cmp::min(self.f, t.f),
        }
    }

    fn mutate(&self) -> Test {
        if self.f < 0 {
            Test { f: self.f + 1 }
        } else if self.f > 0 {
            Test { f: self.f - 1 }
        } else {
            *self
        }
    }
}

#[derive(PartialEq, Eq, Clone)]
struct IntW(pub i64);

impl Phenotype<i64> for IntW{
    fn fitness(&self) -> i64 {
        self.0
    }

    fn crossover(&self, other: &Self) -> IntW{
        IntW(std::cmp::max(self.0, other.0))
    }

    fn mutate(&self) -> IntW{
        self.clone()
    }

    fn relative_fitness(&self, other: &Self) -> f64 {
        (self.0 - other.0) as f64
    }
}

#[test]
fn test_head_to_head() {

    for iter in 0..1 {
        let mut population: Vec<IntW> = (0..1001).map(|i| IntW(i)).collect();
        let mut s = ::sim::seq::Simulator::builder(&mut population);

        s.with_selector(Box::new(TournamentSelector::new_head_to_head(HeadToHeadKind::Elimination, 20, 100).unwrap())).with_max_iters(100);
        let mut s = s.build();
        assert_eq!(s.run() , ::sim::RunResult::Done);
        let res = s.get().unwrap();
        
        assert_eq!(res.0, 1000, "iteration: {}", iter);
    }

}

#[test]
fn test_head_to_head_swiss() {

    for iter in 0..1 {
        let mut population: Vec<IntW> = (0..1001).map(|i| IntW(i)).collect();
        let mut s = ::sim::seq::Simulator::builder(&mut population);

        s.with_selector(Box::new(TournamentSelector::new_head_to_head(HeadToHeadKind::Swiss(10), 50, 40).unwrap())).with_max_iters(100);
        let mut s = s.build();
        assert_eq!(s.run() , ::sim::RunResult::Done);
        let res = s.get().unwrap();
        
        assert_eq!(res.0, 1000, "iteration: {}", iter);
    }

}