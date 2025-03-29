# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

from data.utils_strokerehab import strokerehab_load_dataset, HUMAN_INPUT_JSON_PATH

dataset = strokerehab_load_dataset(filter_for_testset=True)
