# Changelog #

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased ##
- implement AM clique cover
- implement ECC heuristics

### In progress ###
- causal clustering
- causal abstraction
- GUES: improved GES using unconditional equivalence classes (UECs)
- improving GAN performance and theory

## [0.7.0] - 2022-10-18 ##
### Changed ###
- updated licence to CNPLv7+
- updated description in `setup.py` and readme to reflect expanded focus of the package

### Added ###
- `grues` submodule implementing an MCMC-based method for UEC learning

## [0.6.0] - 2021-06-07 ##
### Changed ###
- licences from AGPLv3+ to CNPLv6+	

### Fixed ###
- name in `setup.py`
- numpy depricated boolean subtraction graph.py l269

## [0.5.0] - 2020-09-22 ##
### Added ###
- can now learn functional MCM instead of just structure (using GAN architecture)
- data simulation (using same GAN architecture)
- module for visualizing graphs and making various other plots
- vastly improved documentation site
- `examples` submodule, with some sample structures and data sets
- PGM conference demo script
- more metadata to setup.py

### Changed ###
- `independence_testing.dependencies()` subsumed by `independence_testing.hypothesis_test()`
- license from GPLv3 to AGPLv3+
- using [Black](https://black.readthedocs.io/en/stable/?badge=stable) code formatter

### Fixed ###
- various small typo and bug fixes

## [0.4.0] - 2020-07-23 ##
### Added ###
- package now available on pip

### Fixed ###
- bug in `max_intersection_num` calculation
- exception handling for edgeless graph
- bug with `graph.cover_edges()` use in `ecc.find_cm.branch()`
- bug with using graph copy outside of `ecc.find_cm.branch()` rather than within
- bug where `graph.choose_nbrhood()` picked wrong edge (and gave edgeless nbrhood)

### Removed ###
- hack to stop while loop in `find_clique_min_cover()`

### Changed ###
- default verbosity of output
- `reducible_graph` var name to `branch_graph`
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
