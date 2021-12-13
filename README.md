# YAstarMM (or YA*MM)

The suite name is the acronym of: Yet Another ( __*__ | Continuous
Time | Hidden ) Markov Model

## Installation

### Create the virtual environment

https://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation

```
mkvirtualenv -a ~/YAstarMM/                        \
             -r ~/YAstarMM/requirements.txt        \
             --verbose                             \
             --python=/usr/bin/python3             \
             --prompt="(YastarMM) "                \
             YAstarMM
```

### Replace the YMMV in the codebase

```grep -Hinre "YOUR_MILAGE_MAY_VARY" ~/YAstarMM```

## Dependencies

* [pandas](https://pandas.pydata.org) ([BSD 3-Clause
  License](https://github.com/pandas-dev/pandas/blob/master/LICENSE))

* [PyYAML](https://pyyaml.org) ([MIT
  License](https://github.com/yaml/pyyaml/blob/master/LICENSE))

## Usage

```
from YAstarMM.composer import run as run_composer
from YAstarMM.flavoured_parser import parsed_args
from YAstarMM.hmm import run as run_hmm_training

if __name__ == "__main__":
   if getattr(parsed_args(), "composer_input_dir", None) is not None:
        run_composer()
    else:
        run_hmm_training()
```

## Credits

* Davide Ferrari (UNIMORE, now Ph.D. at King's College)

  who followed the early stages of the project

* Federico Motta (UNIMORE)

* Francesco Ghinelli (UNIMORE)

  who undirectly helped preprocessing the dataset

* Jonathan Law (NICD)

  who followed the early stages of the project and provided a
  mathematical model and a first implementation in R

## LICENSE

See full text [license](LICENSE) here; what follows are the copyright
and license notices.

```
Copyright (C) 2021 Federico Motta <federico.motta@unimore.it>
              2020 Federico Motta <191685@studenti.unimore.it>

YAstarMM is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

YAstarMM is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with YAstarMM.  If not, see
<[https://www.gnu.org/licenses/](https://www.gnu.org/licenses/)>.
```