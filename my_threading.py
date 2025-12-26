from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import threading
from typing_extensions import override


class ContextThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        super().__init__(group=group, target=copy_context().run, name=name, args=(target, *args), kwargs=kwargs, daemon=daemon)


class ContextThreadExecutor(ThreadPoolExecutor):
    @override
    def submit(self, fn, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)
