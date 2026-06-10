# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core optimization engine for Spark Optima.

This module contains the core components for Spark configuration optimization,
including the main Optimizer class, result handling, and base interfaces.
"""

from spark_optima.core.history import HistoryEntry, OptimizationHistory
from spark_optima.core.optimizer import Optimizer
from spark_optima.core.result import OptimizationResult

__all__ = ["HistoryEntry", "OptimizationHistory", "Optimizer", "OptimizationResult"]
