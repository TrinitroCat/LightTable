import sys

from PySide6.QtWidgets import QApplication

from console_backend import ConsoleBackend
from data_backend import DataBackend
from display import MainWindow

__version__ = 'v0.1'

def main() -> int:
    app = QApplication(sys.argv)

    backend = DataBackend()
    console_backend = ConsoleBackend(backend)
    window = MainWindow(backend)

    window.console.submitted.connect(
        lambda code: window.console.append_output(console_backend.execute(code))
    )
    window.console.live_text_changed.connect(
        lambda text: console_backend.preview_selection_from_code(text, emit_warning=False)
    )
    window.console.clear_requested.connect(window.console.clear_output)
    window.console.clear_requested.connect(window.console.clear_input)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())