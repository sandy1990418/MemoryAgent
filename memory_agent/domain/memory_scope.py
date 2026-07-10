from enum import Enum


class MemoryScope(str, Enum):
    USER = "user"
    SESSION = "session"
    TASK = "task"
    AGENT = "agent"
    PROJECT = "project"
    GLOBAL = "global"
