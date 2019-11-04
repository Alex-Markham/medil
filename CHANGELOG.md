# Changelog #

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased ##
- make proper python package and add figure out unit testing
- add HSIC perm test
- implement AM clique cover
- learn actual functions corresponding to minMCM edges

## [0.2.0] - 2018-11-04 ##
### Added ###
- references and better example to README
- graph and find_ecc modules---makes package more maintainable and extensible
- test script

### Changed ###
- organization more in line with python package guidelines
- changed from using distance covariance to distance correlation
- many functions renamed according to new modules and organization

### Removed ###
- need for importing networkx to find minMCM
- junk code in misc dir

## [0.1.0] - 2018-10-07 ##
### Added ###
- code in its raw, disorganized, poorly documented, inefficient state
