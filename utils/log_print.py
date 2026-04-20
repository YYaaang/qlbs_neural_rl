import os
from datetime import datetime


class LogPrint(object):
    """
    Logger that:
    1) accumulates log strings
    2) optionally prints to stdout
    3) saves logs under model/experiment directory with timestamp
    4) can raise exceptions WITH logging
    """

    def __init__(
        self,
        print_debug: bool,
        *,
        save_dir: str | None = None,
        prefix: str = "train",
        auto_flush: bool = True,
    ):
        self.print_debug = print_debug
        self.log_str: str = ""
        self.auto_flush = auto_flush

        self.log_path = None
        self._fh = None
        self._closed = False

        if save_dir is not None:
            if not isinstance(save_dir, str) or not save_dir.strip():
                raise ValueError("save_dir must be a non-empty string or None")

            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{prefix}_{timestamp}.log"
            self.log_path = os.path.join(save_dir, filename)

            self._fh = open(self.log_path, "w", encoding="utf-8")

    # ======================================================
    # internal helpers
    # ======================================================
    def _check_open(self):
        if self._closed:
            raise ValueError("LogPrint is already closed")

    def _write_raw(self, line: str):
        self.log_str += line + "\n"

        if self.print_debug:
            print(line)

        if self._fh is not None:
            self._fh.write(line + "\n")
            if self.auto_flush:
                self._fh.flush()

    # ======================================================
    # public logging API
    # ======================================================
    def write(self, new_str: str):
        if not isinstance(new_str, str):
            raise ValueError(
                f"LogPrint.write expects a string, got {type(new_str).__name__}"
            )

        self._check_open()
        self._write_raw(new_str)

    # ======================================================
    # ✨ NEW: error + raise interface
    # ======================================================
    def raise_value_error(self, message: str):
        """
        Log an error message and raise ValueError.

        This guarantees:
        - the error is written into log file
        - the error is visible in log_str
        - the program still fails fast
        """
        if not isinstance(message, str):
            message = repr(message)

        self._check_open()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        err_line = f"[ERROR {timestamp}] {message}"

        # always log errors
        self._write_raw(err_line)

        # flush before raising
        if self._fh is not None:
            self._fh.flush()

        raise ValueError(message)

    # ======================================================
    # lifecycle
    # ======================================================
    def close(self):
        if not self._closed:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    # ======================================================
    # ✨ file-not-found + raise interface
    # ======================================================
    def raise_file_not_found_error(self, message: str, *, path: str | None = None):
        """
        Log a file-not-found error and raise FileNotFoundError.

        Args:
            message: human-readable error description
            path: optional file path that was not found
        """
        if not isinstance(message, str):
            message = repr(message)

        self._check_open()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if path is not None:
            err_line = f"[FILE NOT FOUND {timestamp}] {message} | path={path}"
        else:
            err_line = f"[FILE NOT FOUND {timestamp}] {message}"

        # write to log
        self._write_raw(err_line)

        # flush before raising
        if self._fh is not None:
            self._fh.flush()

        if path is not None:
            raise FileNotFoundError(f"{message} (path={path})")
        else:
            raise FileNotFoundError(message)