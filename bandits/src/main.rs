mod bandit;
mod agent;

extern crate csv;
use clap::{Arg, App, ArgMatches};
use std::error::Error;
use bandit::{TenArmedTestbed};
use agent::*;
use csv::Writer;

fn run(matches: ArgMatches) -> Result<(), Box<dyn Error>> {
    let output_filename = matches.value_of("output").unwrap();

    let testbed = TenArmedTestbed::new();
    let mut agent = EpsilonGreedyAgent::new(10, 0.1);
    let (rewards, optimal) = testbed.run(&mut agent);
    let mut writer = Writer::from_path(output_filename)?;
    writer.write_record(&["trial", "step", "reward", "optimal"])?;
    for (i, (trial_r, trial_o)) in rewards.iter().zip(optimal.iter()).enumerate() {
        for (j, (r, o)) in trial_r.iter().zip(trial_o.iter()).enumerate() {
            writer.serialize((i, j, r, o))?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn main() {
    let matches = App::new("Bandits")
        .arg(Arg::with_name("output")
            .required(true)
            .index(1)
            .takes_value(true)
            .help("Output csv path."))
        .get_matches();
        
    run(matches).unwrap();
}
