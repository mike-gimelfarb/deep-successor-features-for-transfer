# -*- coding: UTF-8 -*-
from ast import literal_eval
from collections import defaultdict
import configparser
import os
from pathlib import Path


def parse_config_file(name):
    raw_path = Path(__file__).parent.parent
    config_path = os.path.join(raw_path, 'configs', name)
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read(config_path)
    section_dict = defaultdict()
    for section in config.sections():
        section_dict[section] = {k: literal_eval(v) for k, v in config.items(section)}
    return section_dict
