"""Simulated governance platform: read tools, one draft-only write tool, and a reference validator."""
from .tools import Platform, ToolError, TOOL_SPECS  # noqa: F401
from .validator import Validator  # noqa: F401
