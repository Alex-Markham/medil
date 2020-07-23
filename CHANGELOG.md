# Changelog #

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased ##
- add HSIC test
- implement AM clique cover
### In progress###
- causal clustering

## [0.5.0] - 2020-0X-XX ##
### Added ###
- can now learn functional MCM instead of just structure (using GAN architecture)
- data simulation (using same GAN architecture)
- module for visualizing graphs and making various other plots

## [0.4.0] - 2020-07-23 ##
### Added ###
- package now available on pip

### Fixed ###
- bug in max\_intersection\_num calculation
- exception handling for edgeless graph
- bug with graph.cover\_edges() use in ecc.find\_cm.branch()
- bug with using graph copy outside of ecc.find\_cm.branch() rather than within
- bug where graph.choose\_nbrhood() picked wrong edge (and gave edgeless nbrhood)

### Removed ###
- hack to stop while loop in find\_clique\_min\_cover()

### Changed ###
- default verbosity of output
- reducible\_graph var name to branch\_graph
- package requirements
- updated README

## [0.3.0] - 2020-04-13 ##
### Added ###
- installation via pip and git
- rudimentary output to update on progress
- doc generating script and initial (incomplete) documentation website
- `tests/` directory and various files for more detailed integration and unit testing

### Fixed ###
- bug in reduction rule 2 of find_cm ecc alg that shows up on fully connected graphs
- bug in reduction rule 3 of the find_cm ecc alg
- incorrect dates (2018 instead of 2019) in changelog

### Removed ###
- `utils.py`---distributed functions to relevant modules

### Changed ###
- `indpendence_test.py` --> `independence_testing.py`
- split up `ecc_algs.reducee()` internals for better testing/readability

## [0.2.0] - 2019-11-04 ##
### Added ###
- references and better example to README
- `graph.py` and `ecc_algorithms.py` modules---makes package more maintainable and extensible
- test script

### Changed ###
- organization more in line with python package guidelines
- changed from using distance covariance to distance correlation
- many functions renamed according to new modules and organization

### Removed ###
- need for importing networkx to find minMCM
- junk code in misc dir

## [0.1.0] - 2019-10-07 ##
### Added ###
- code in its raw, disorganized, poorly documented, inefficient state
