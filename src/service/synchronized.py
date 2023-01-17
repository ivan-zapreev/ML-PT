# For safe execution of synchronized blocks
class safe_exec():
    def __init__(self, _lock):
        self._lock = _lock
    
    def execute(self, operation, *argv, **kwargs):
        # Start safe section
        self._lock.acquire()
        try:
            return operation(*argv, **kwargs)
        finally:
            # Finish safe section
            self._lock.release()
