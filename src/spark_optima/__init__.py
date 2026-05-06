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

"""Spark Optima - Intelligent Apache Spark Configuration Optimization Tool.

This package provides a professional tool for automatically finding optimal
Apache Spark configurations using hybrid heuristic and Bayesian optimization.

Example:
    >>> from spark_optima import Optimizer
    >>> optimizer = Optimizer(platform="databricks")
    >>> result = optimizer.optimize(code_path="./job.py")
    >>> print(result.configuration)

Attributes:
    __version__: The version string of the package.
    __author__: The author of the package.

"""

__version__ = "0.1.0"
__author__ = "Spark Optima Contributors"
__email__ = "your-email@example.com"
__license__ = "Apache-2.0"

from spark_optima.core.optimizer import Optimizer
from spark_optima.core.result import OptimizationResult

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "__version__",
]
