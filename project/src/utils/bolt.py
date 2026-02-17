"""Bolt utilities for Apple TuriBolt integration.

This module provides utilities for detecting and interacting with TuriBolt.
All functions are robust and work even when apple_bolt is not installed.
"""

from typing import Optional

# Try to import apple_bolt, but don't fail if it's not available
try:
    import apple_bolt as bolt
    from apple_bolt import ARTIFACT_DIR
    _BOLT_AVAILABLE = True
except ImportError:
    bolt = None
    ARTIFACT_DIR = None
    _BOLT_AVAILABLE = False


def is_bolt_available() -> bool:
    """Check if apple_bolt is installed and available.

    Returns:
        True if apple_bolt is installed, False otherwise.
    """
    return _BOLT_AVAILABLE


def is_on_bolt() -> bool:
    """Check if currently running on TuriBolt.

    Returns:
        True if running on Bolt (has a current task ID), False otherwise.
    """
    if not _BOLT_AVAILABLE:
        return False
    return bool(bolt.get_current_task_id())


def is_interactive() -> bool:
    """Check if running in interactive mode on Bolt.

    Returns:
        True if running interactively on Bolt, False otherwise.
    """
    if not is_on_bolt():
        return False
    return bolt.get_task(bolt.get_current_task_id()).is_interactive


def get_artifact_dir() -> Optional[str]:
    """Get the Bolt artifact directory if available.

    Returns:
        The ARTIFACT_DIR path if on Bolt, None otherwise.
    """
    if is_on_bolt():
        return ARTIFACT_DIR
    return None


def get_parent_id() -> Optional[str]:
    """Get the parent task ID if running on Bolt.

    Returns:
        Parent task ID if it exists, current task ID if no parent, None if not on Bolt.
    """
    if not is_on_bolt():
        return None

    task = bolt.get_task(bolt.get_current_task_id())
    if task.parent_id:
        return task.parent_id
    return task.id
