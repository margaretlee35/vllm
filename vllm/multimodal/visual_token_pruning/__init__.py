# SPDX-License-Identifier: Apache-2.0
"""Visual token pruning strategies for multimodal models.

This package provides a lightweight registry for visual-token pruning
strategies that can be plugged into multimodal model wrappers.

Strategies are registered at import time (via module-level calls to
``register_visual_pruning_strategy``) and looked up by name at runtime.
"""
