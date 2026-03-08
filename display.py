"""
The GUI visualization part.
"""

from __future__ import annotations

import ast
from typing import Optional
import re
import os

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, QEvent, QSignalBlocker
from PySide6.QtGui import (
    QAction, QColor, QBrush, QTextCursor, QKeyEvent,
    QTextCharFormat, QSyntaxHighlighter, QFontMetrics,
    QShortcut, QKeySequence
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableView,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QComboBox,
    QFrame,
    QApplication,
    QFileDialog,
    QMenu,
)

import numpy as np

from data_backend import DataBackend


def blend_colors(colors: list[QColor]) -> QColor:
    if not colors:
        return QColor()
    r = g = b = a = 0
    for c in colors:
        r += c.red()
        g += c.green()
        b += c.blue()
        a += c.alpha()
    n = len(colors)
    return QColor(r // n, g // n, b // n, min(255, a // n + 40))


class DataExprCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[ast.Subscript] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "data":
            self.nodes.append(node)
        self.generic_visit(node)


class ConsoleHighlighter(QSyntaxHighlighter):
    def __init__(self, doc, palette: list[QColor]) -> None:
        super().__init__(doc)
        self.palette = palette

    def highlightBlock(self, text: str) -> None:
        try:
            tree = ast.parse(text, mode="exec")
        except Exception:
            return

        collector = DataExprCollector()
        collector.visit(tree)

        for i, node in enumerate(collector.nodes):
            color = self.palette[i % len(self.palette)]
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            if hasattr(node, "col_offset") and hasattr(node, "end_col_offset"):
                start = node.col_offset
                length = node.end_col_offset - node.col_offset
                if length > 0:
                    self.setFormat(start, length, fmt)


class SelectionTableView(QTableView):
    """
    To show the selection zone on the table.
    """
    space_pressed = Signal()
    copy_requested = Signal()  # ctrl+c
    paste_requested = Signal()  # ctrl+v
    viewport_resized = Signal(object, object)  # old_size, new_size

    def resizeEvent(self, event) -> None:
        old_size = self.viewport().size()
        super().resizeEvent(event)
        new_size = self.viewport().size()

        if old_size.width() > 0 and old_size.height() > 0 and old_size != new_size:
            self.viewport_resized.emit(old_size, new_size)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space:
            event.accept()
            self.space_pressed.emit()
            return

        if event.matches(QKeySequence.Paste):
            event.accept()
            self.paste_requested.emit()
            return

        if event.matches(QKeySequence.Copy):
            event.accept()
            self.copy_requested.emit()
            return

        super().keyPressEvent(event)


class NumpyTableModel(QAbstractTableModel):
    """
    To show the numpy array as a QTableView.
    """
    def __init__(self, backend: DataBackend) -> None:
        super().__init__()
        self.backend = backend

        self.mouse_cells: set[tuple[int, int]] = set()
        self.preview_groups: list[set[tuple[int, int]]] = []

        self.preview_palette = [
            QColor(255, 120, 120, 110),
            QColor(120, 170, 255, 110),
            QColor(120, 220, 160, 110),
            QColor(255, 190, 100, 110),
            QColor(200, 140, 255, 110),
            QColor(120, 220, 220, 110),
        ]
        self.text_palette = [
            QColor(190, 50, 50),
            QColor(50, 90, 210),
            QColor(30, 140, 90),
            QColor(180, 110, 30),
            QColor(130, 70, 180),
            QColor(40, 150, 150),
        ]
        self.mouse_color = QColor(255, 235, 140, 90)

        self.backend.data_changed.connect(self._reset_model)
        self.backend.mouse_selection_changed.connect(self._on_mouse_selection_changed)
        self.backend.preview_selections_changed.connect(self._on_preview_groups_changed)

    def _reset_model(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def _full_refresh_bg(self) -> None:
        if self.rowCount() > 0 and self.columnCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
                [Qt.BackgroundRole],
            )

    def _on_mouse_selection_changed(self, cells: set[tuple[int, int]]) -> None:
        self.mouse_cells = set(cells)
        self._full_refresh_bg()

    def _on_preview_groups_changed(self, groups: list[set[tuple[int, int]]]) -> None:
        self.preview_groups = [set(g) for g in groups]
        self._full_refresh_bg()

    def rowCount(self, parent=QModelIndex()) -> int:
        return self.backend.shape[0]

    def columnCount(self, parent=QModelIndex()) -> int:
        return self.backend.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        r, c = index.row(), index.column()

        if role == Qt.BackgroundRole:
            layers: list[QColor] = []

            if (r, c) in self.mouse_cells:
                layers.append(self.mouse_color)

            for i, group in enumerate(self.preview_groups):
                if (r, c) in group:
                    layers.append(self.preview_palette[i % len(self.preview_palette)])

            if layers:
                return QBrush(blend_colors(layers))

        if role in (Qt.DisplayRole, Qt.EditRole):
            value = self.backend.data[r, c]
            if isinstance(value, (float, np.floating)):
                return f"{value:.6g}"
            return str(value)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return f"C{section}" if orientation == Qt.Horizontal else f"R{section}"

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False
        try:
            self.backend.set_value(index.row(), index.column(), value)
            return True
        except Exception:
            return False


class ConsoleInput(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and event.modifiers() & Qt.ShiftModifier
        ):
            event.accept()
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class ConsolePanel(QWidget):
    submitted = Signal(str)
    live_text_changed = Signal(str)
    clear_requested = Signal()

    def __init__(self, text_palette: list[QColor], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # 新增的工具栏区域
        top = QHBoxLayout()
        self.title = QLabel("Console")
        self.run_button = QPushButton("执行")
        self.clear_after_submit = QCheckBox("提交后清空输入")
        self.clear_after_submit.setChecked(True)  # default: true
        self.clear_input_button = QPushButton("清空输入")
        self.clear_output_button = QPushButton("清空输出")

        # 添加调整字号的 QComboBox 控件
        self.font_size_combobox = QComboBox()
        self.font_size_combobox.addItems(["8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "32", "36", "40"])
        self.font_size_combobox.setCurrentText("16")  # Default font size: 16

        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.run_button)
        top.addWidget(self.clear_after_submit)
        top.addWidget(self.clear_input_button)
        top.addWidget(self.clear_output_button)
        top.addWidget(QLabel("字号: "))
        top.addWidget(self.font_size_combobox)  # Font config

        self.input_edit = ConsoleInput()
        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.adjust_font_size("16")
        self.input_edit.setPlaceholderText(
            "Shift+Enter 提交执行表达式；在表格中选中区域后，按 Space 写入所选区域的表达式 data[...]。\n\n"
            "示例：\n"
            "data[1:4, [0, 2, 4]] + data[:, [1, 3]]\n"
            "data[8:10, 6:8] = np.ones((2, 2))\n"
            "np.mean(data)\n"
            "addr(2, 3); delr(2, 1)\n"
            "addc(1, 2); delc(1, 1)"
        )
        self.highlighter = ConsoleHighlighter(self.input_edit.document(), text_palette)

        layout.addLayout(top)
        layout.addWidget(QLabel("输入"))
        layout.addWidget(self.input_edit, 2)
        layout.addWidget(QLabel("输出"))
        layout.addWidget(self.output_edit, 3)

        self.run_button.clicked.connect(self._submit)
        self.clear_input_button.clicked.connect(self.clear_requested.emit)
        self.clear_output_button.clicked.connect(self.clear_requested.emit)
        self.input_edit.textChanged.connect(self._emit_live_text)
        self.input_edit.submitted.connect(self._submit)
        # 新增事件连接，用于调整字号
        self.font_size_combobox.currentTextChanged.connect(self.adjust_font_size)

    def adjust_font_size(self, size: str) -> None:
        """
        adjust font size to input `size`
        """
        font = self.input_edit.font()
        font.setPointSize(int(size))
        self.input_edit.setFont(font)
        self.output_edit.setFont(font)

    def _submit(self) -> None:
        code = self.input_edit.toPlainText().strip()
        if code:
            self.submitted.emit(code)
            if self.clear_after_submit.isChecked():
                self.input_edit.clear()

    def _emit_live_text(self) -> None:
        self.live_text_changed.emit(self.input_edit.toPlainText())

    def append_output(self, text: str) -> None:
        self.output_edit.moveCursor(QTextCursor.End)
        self.output_edit.insertPlainText(text)
        if not text.endswith("\n"):
            self.output_edit.insertPlainText("\n")
        self.output_edit.moveCursor(QTextCursor.End)

    def append_input(self, text: str) -> None:
        """
        append `text` to `self.input_edit.toPlainText()`, and then reset focues to the console
        Args:
            text:

        Returns:

        """
        self.input_edit.setFocus(Qt.OtherFocusReason)
        cursor = self.input_edit.textCursor()
        cursor.insertText(text)
        self.input_edit.setTextCursor(cursor)
        self.input_edit.ensureCursorVisible()

    def clear_output(self) -> None:
        self.output_edit.clear()

    def clear_input(self) -> None:
        self.input_edit.clear()


class RibbonGroup(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)

        root.addLayout(self.content_layout)
        root.addWidget(self.title_label)


class RibbonToolbar(QWidget):
    """
    Toolbar widget for Ribbon.
    just like MS Office-style
    """
    font_size_changed = Signal(int)
    dtype_selected = Signal(str)
    cell_size_changed = Signal(int, int)  # col_width, row_height

    def __init__(self, dtype_names: list[str], parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Title
        self.file_group = RibbonGroup("文件")
        self.file_label = QLabel("Untitled1.csv")
        self.file_group.content_layout.addWidget(QLabel("当前:"))
        self.file_group.content_layout.addWidget(self.file_label)
        layout.addWidget(self.file_group)

        # 字号组
        self.font_group = RibbonGroup("字体")
        self.font_dec_btn = QPushButton("A-")
        self.font_inc_btn = QPushButton("A+")
        self.font_size_box = QComboBox()
        self.font_size_box.setEditable(True)
        self.font_size_box.addItems(["8", "9", "10", "11", "12", "14", "16", "18", "20", "24", "28", "32"])
        self.font_size_box.setCurrentText("12")
        self.font_size_box.setMaximumWidth(72)

        self.font_group.content_layout.addWidget(self.font_dec_btn)
        self.font_group.content_layout.addWidget(self.font_size_box)
        self.font_group.content_layout.addWidget(self.font_inc_btn)

        # Change dtype
        self.dtype_group = RibbonGroup("数据类型")
        self.dtype_label_caption = QLabel("当前:")
        self.dtype_label = QLabel("float64")
        self.dtype_box = QComboBox()
        self.dtype_box.setEditable(False)
        self.dtype_box.addItems(dtype_names)
        self.dtype_box.setCurrentText("float64")

        self.dtype_group.content_layout.addWidget(self.dtype_label_caption)
        self.dtype_group.content_layout.addWidget(self.dtype_label)
        self.dtype_group.content_layout.addWidget(self.dtype_box)

        layout.addWidget(self.dtype_group)

        self.view_group = RibbonGroup("视图")
        self.cell_width_box = QComboBox()
        self.cell_width_box.setEditable(True)
        self.cell_width_box.addItems(["40", "60", "80", "100", "120", "160", "200"])
        self.cell_width_box.setCurrentText("100")
        self.cell_width_box.setMaximumWidth(72)

        self.cell_height_box = QComboBox()
        self.cell_height_box.setEditable(True)
        self.cell_height_box.addItems(["18", "22", "26", "30", "36", "44", "52"])
        self.cell_height_box.setCurrentText("30")
        self.cell_height_box.setMaximumWidth(72)

        self.view_group.content_layout.addWidget(QLabel("列宽"))
        self.view_group.content_layout.addWidget(self.cell_width_box)
        self.view_group.content_layout.addWidget(QLabel("行高"))
        self.view_group.content_layout.addWidget(self.cell_height_box)

        self.cell_width_box.lineEdit().editingFinished.connect(self._emit_cell_size)
        self.cell_height_box.lineEdit().editingFinished.connect(self._emit_cell_size)
        self.cell_width_box.currentTextChanged.connect(lambda _: self._emit_cell_size())
        self.cell_height_box.currentTextChanged.connect(lambda _: self._emit_cell_size())

        # 预留空组
        self.edit_group = RibbonGroup("编辑(未实装)")

        layout.addWidget(self.font_group)
        layout.addWidget(self.edit_group)
        layout.addWidget(self.view_group)
        layout.addStretch()

        # set connections
        self.font_dec_btn.clicked.connect(lambda: self._emit_relative(-1))
        self.font_inc_btn.clicked.connect(lambda: self._emit_relative(1))
        self.font_size_box.lineEdit().editingFinished.connect(self._emit_absolute)
        self.font_size_box.currentTextChanged.connect(lambda _: self._emit_absolute())

        self.dtype_box.currentTextChanged.connect(self.dtype_selected.emit)

    def _current_size(self) -> int:
        try:
            return max(6, min(72, int(self.font_size_box.currentText().strip())))
        except Exception:
            return 12

    def _emit_absolute(self) -> None:
        size = self._current_size()
        self.font_size_box.setCurrentText(str(size))
        self.font_size_changed.emit(size)

    def _emit_relative(self, step: int) -> None:
        size = max(6, min(72, self._current_size() + step))
        self.font_size_box.setCurrentText(str(size))
        self.font_size_changed.emit(size)

    def _current_cell_width(self) -> int:
        """
        get the current width of each cell
        Returns:

        """
        try:
            return max(20, min(400, int(self.cell_width_box.currentText().strip())))
        except Exception:
            return 100

    def _current_cell_height(self) -> int:
        """
        get the current height of each cell
        Returns:

        """
        try:
            return max(16, min(200, int(self.cell_height_box.currentText().strip())))
        except Exception:
            return 30

    def _emit_cell_size(self) -> None:
        """

        Returns:

        """
        w = self._current_cell_width()
        h = self._current_cell_height()
        self.cell_width_box.setCurrentText(str(w))
        self.cell_height_box.setCurrentText(str(h))
        self.cell_size_changed.emit(w, h)

    def set_dtype(self, dtype_name: str) -> None:
        """
        Options to set the dtype.
        Args:
            dtype_name:

        Returns:

        """
        self.dtype_label.setText(dtype_name)

        from PySide6.QtCore import QSignalBlocker
        blocker = QSignalBlocker(self.dtype_box)

        idx = self.dtype_box.findText(dtype_name)
        if idx >= 0:
            self.dtype_box.setCurrentIndex(idx)

    def set_current_file(self, name: str) -> None:
        self.file_label.setText(name)

    def set_cell_size(self, col_width: int, row_height: int) -> None:
        b1 = QSignalBlocker(self.cell_width_box)
        b2 = QSignalBlocker(self.cell_height_box)
        self.cell_width_box.setCurrentText(str(col_width))
        self.cell_height_box.setCurrentText(str(row_height))


class TitleRowModel(QAbstractTableModel):
    """
    The Title that is dependent with main data and only labels each column of data.
    """
    def __init__(self, backend: DataBackend) -> None:
        super().__init__()
        self.backend = backend
        self.titles = [f"标题 {i}" for i in range(backend.shape[1])]
        self.backend.data_changed.connect(self._sync_columns)

    def _sync_columns(self) -> None:
        cols = self.backend.shape[1]
        if len(self.titles) < cols:
            self.titles.extend(f"标题 {i}" for i in range(len(self.titles), cols))
        else:
            self.titles = self.titles[:cols]
        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 1

    def columnCount(self, parent=QModelIndex()) -> int:
        return self.backend.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return self.titles[index.column()]

        if role == Qt.BackgroundRole:
            return QBrush(QColor(245, 245, 245))

        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False
        self.titles[index.column()] = str(value)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        return True

    def set_titles(self, titles: list[str]) -> None:
        cols = self.backend.shape[1]
        titles = [str(x) for x in titles]

        if len(titles) < cols:
            titles = titles + [f"标题 {i}" for i in range(len(titles), cols)]
        else:
            titles = titles[:cols]

        self.titles = titles
        self.beginResetModel()
        self.endResetModel()

    def export_titles(self) -> list[str]:
        return list(self.titles)


class MainWindow(QMainWindow):
    def __init__(self, backend: DataBackend) -> None:
        super().__init__()
        self.backend = backend

        # Current opened file
        self.current_file_path: str | None = None
        self.current_file_name = "Untitled1.csv"
        self.setAcceptDrops(True)

        self.setWindowTitle("PySide + NumPy Table Console")
        self.resize(1100, 760)

        # sizes information
        self._baseline_font_size = 12
        self._font_size_now = 12
        self._cell_col_width = 100
        self._cell_row_height = 30

        # Main Framework
        self.table_model = NumpyTableModel(backend)
        self.table_view = SelectionTableView()
        self.table_view.setFrameShape(QFrame.NoFrame)
        self.table_view.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.table_view.paste_requested.connect(self._paste_into_table)
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table_view.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)

        self.console = ConsolePanel(self.table_model.text_palette)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table_view)
        splitter.addWidget(self.console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        # ribbons
        self.ribbon = RibbonToolbar(list(self.backend.ALLOWED_TYPES.keys()))
        self.ribbon.set_dtype(self.backend.current_dtype_name)
        #self.ribbon.set_cell_size(self.table_view.columnWidth(0), self.table_view.rowHeight(0) if self.table_model.rowCount() > 0 else 30)

        # The part of the title line
        self.title_model = TitleRowModel(backend)
        self.title_view = QTableView()
        self.title_view.setModel(self.title_model)
        self.title_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.title_view.setFocusPolicy(Qt.ClickFocus)
        self.title_view.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.title_view.setFixedHeight(40)
        self.title_view.horizontalHeader().hide()
        self.title_view.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.title_view.verticalHeader().hide()
        self.title_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.title_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.title_view.setFrameShape(QFrame.NoFrame)
        self.title_view.setShowGrid(True)
        self.title_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.title_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Adding menu when click right key
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        #self._last_viewport_size = self.table_view.viewport().size()  # record the size before resizeEvent triggered.

        self.title_corner = QWidget()
        self.title_corner.setFixedWidth(self.table_view.verticalHeader().width())

        # The part of the Toolbar
        self.title_bar = QWidget()
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar_layout.setSpacing(0)
        title_bar_layout.addWidget(self.title_corner)
        title_bar_layout.addWidget(self.title_view, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ribbon)
        layout.addWidget(self.title_bar)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # zoom in/out by keyboard
        self._geometry_scaling = False
        self.table_view.viewport().installEventFilter(self)

        self.zoom_in_shortcut = QShortcut(QKeySequence("Ctrl+="), self)
        self.zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        self.zoom_in_shortcut.activated.connect(lambda: self.zoom_cell_size(1))
        self.zoom_out_shortcut.activated.connect(lambda: self.zoom_cell_size(-1))

        self._setup_connections()
        self._setup_menu()
        self.ribbon.set_current_file(self.current_file_name)
        self._apply_view_metrics()
        self._sync_title_view_geometry()
        self._sync_title_corner_geometry()
        self.update_status()

    # Section: FILE I/O
    def _read_title_comment(self, path: str) -> list[str] | None:
        """
        Reads a title comment from a text file.
        Args:
            path:

        Returns:

        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
        except Exception:
            return None

        if not first.startswith("#"):
            return None

        text = first[1:].strip()
        if not text:
            return None

        parts = [p for p in re.split(r",+", text) if p]
        return parts or None

    def _set_current_file(self, path: str | None) -> None:
        self.current_file_path = path
        if path:
            self.current_file_name = os.path.basename(path)
        else:
            self.current_file_name = "Untitled1.csv"
        self.ribbon.set_current_file(self.current_file_name)

    def _open_file(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "打开文件",
                "",
                "Data Files (*.csv *.txt *.dat);;All Files (*)"
            )
        if not path:
            return

        titles = self._read_title_comment(path)
        self.backend.load_csv(path)

        if titles:
            self.title_model.set_titles(titles)
        else:
            self.title_model._sync_columns()

        self._set_current_file(path)
        self._sync_title_view_geometry()

    def _save_file(self) -> None:
        if not self.current_file_path:
            self._save_file_as()
            return

        self.backend.save_csv(
            self.current_file_path,
            title=self.title_model.export_titles()
        )
        self._set_current_file(self.current_file_path)

    def _save_file_as(self) -> None:
        default_name = self.current_file_name or "Untitled1.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "另存为",
            default_name,
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return

        if not os.path.splitext(path)[1]:
            path += ".csv"

        self.backend.save_csv(
            path,
            title=self.title_model.export_titles()
        )
        self._set_current_file(path)

    # Section: launch setup
    def _setup_connections(self) -> None:
        sel_model = self.table_view.selectionModel()
        sel_model.selectionChanged.connect(self._on_view_selection_changed)

        self.table_view.space_pressed.connect(self._write_selection_to_console)
        self.table_view.copy_requested.connect(self._copy_from_table)
        self.table_view.customContextMenuRequested.connect(self._show_table_context_menu)

        self.table_view.horizontalScrollBar().valueChanged.connect(
            self.title_view.horizontalScrollBar().setValue
        )
        self.table_view.horizontalHeader().sectionResized.connect(
            self._sync_title_section_width
        )
        # sync the title left offset to match this offset of main data
        self.table_view.verticalHeader().geometriesChanged.connect(self._sync_title_corner_geometry)
        # connect the change of backend data
        self.backend.data_changed.connect(self._sync_title_view_geometry)
        self.ribbon.font_size_changed.connect(self.set_table_font_size)
        self.ribbon.dtype_selected.connect(self.backend.change_types)
        self.backend.dtype_changed.connect(self.ribbon.set_dtype)

        self.backend.status_selection_changed.connect(self._on_status_selection_changed)
        self.backend.data_changed.connect(self.update_status)
        self.backend.warning_emitted.connect(self.show_warning)

        self._geometry_scaling = False
        self.table_view.viewport_resized.connect(self._on_table_viewport_resized)
        # set cell size
        self.ribbon.cell_size_changed.connect(self.set_cell_size)


    # menu looks
    def _setup_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")

        open_action = QAction("打开", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)

        save_action = QAction("保存", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_file)

        save_as_action = QAction("另存为", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_file_as)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)

        help_menu = self.menuBar().addMenu("帮助")
        help_action = QAction("说明", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

    def set_cell_size(self, col_width: int, row_height: int) -> None:
        """
        Change cell size by Toolbar set
        Args:
            col_width:
            row_height:

        Returns:

        """
        self._cell_col_width = max(8, min(400, int(col_width)))
        self._cell_row_height = max(8, min(200, int(row_height)))
        self._apply_view_metrics()

    def _selected_rows_cols(self) -> tuple[list[int], list[int]]:
        """
        Get the selected row and column indices.
        Returns:

        """
        indexes = self.table_view.selectionModel().selectedIndexes()
        if indexes:
            rows = sorted({idx.row() for idx in indexes})
            cols = sorted({idx.column() for idx in indexes})
            return rows, cols

        current = self.table_view.currentIndex()
        if current.isValid():
            return [current.row()], [current.column()]

        return [], []

    def _delete_row_exact(self, row: int) -> None:
        """
        Only delete one row from the table indexed by row index.
        Args:
            row:

        Returns:

        """
        rows, _ = self.backend.shape
        if rows <= 1:
            self.show_warning("至少保留一行")
            return

        if row < 0 or row >= rows:
            return

        if row > 0:
            self.backend.delr(row - 1, 1, '>')
        else:
            self.backend.delr(1, 1, '<')

    def _delete_col_exact(self, col: int) -> None:
        """
        Only delete one column from the table indexed by column index.
        Args:
            col:

        Returns:

        """
        _, cols = self.backend.shape
        if cols <= 1:
            self.show_warning("至少保留一列")
            return

        if col < 0 or col >= cols:
            return

        if col > 0:
            self.backend.delc(col - 1, 1, '>')
        else:
            self.backend.delc(1, 1, '<')

    def _insert_row_below_selection(self) -> None:
        rows, _ = self._selected_rows_cols()
        if not rows:
            return
        self.backend.addr(max(rows), 1, '>')

    def _insert_col_right_selection(self) -> None:
        _, cols = self._selected_rows_cols()
        if not cols:
            return
        self.backend.addc(max(cols), 1, '>')

    def _delete_selected_rows(self) -> None:
        rows, _ = self._selected_rows_cols()
        if not rows:
            return

        for r in sorted(rows, reverse=True):
            self._delete_row_exact(r)

    def _delete_selected_cols(self) -> None:
        _, cols = self._selected_rows_cols()
        if not cols:
            return

        for c in sorted(cols, reverse=True):
            self._delete_col_exact(c)

    def zoom_cell_size(self, step: int) -> None:
        """
        Change cell size by Zoom out/in
        Args:
            step:

        Returns:

        """
        factor = 1.1 if step > 0 else (1 / 1.1)
        self._cell_col_width = max(8, min(400, round(self._cell_col_width * factor)))
        self._cell_row_height = max(8, min(200, round(self._cell_row_height * factor)))
        self._apply_view_metrics()

    def _apply_view_metrics(self) -> None:
        font = self.table_view.font()
        font.setPointSize(self._font_size_now)
        self.table_view.setFont(font)
        self.title_view.setFont(font)

        fm = QFontMetrics(font)
        min_row_h = max(8, int(fm.height() * 1.1))
        min_title_h = max(10, int(fm.height() * 1.2))
        min_col_w = max(16, int(fm.horizontalAdvance("0") * 1.6))

        col_width = max(min_col_w, int(self._cell_col_width))
        row_height = max(min_row_h, int(self._cell_row_height))

        for c in range(self.table_model.columnCount()):
            self.table_view.setColumnWidth(c, col_width)
            self.title_view.setColumnWidth(c, col_width)

        for r in range(self.table_model.rowCount()):
            self.table_view.setRowHeight(r, row_height)

        if self.title_model.rowCount() > 0:
            title_h = max(min_title_h, row_height)
            self.title_view.setRowHeight(0, title_h)
            self.title_view.setFixedHeight(title_h + self.title_view.frameWidth() * 2)

        self._sync_title_corner_geometry()
        self.ribbon.set_cell_size(col_width, row_height)

    def dragEnterEvent(self, event) -> None:
        """
        To support dragEnter files.
        Args:
            event:

        Returns:

        """
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls and urls[0].isLocalFile():
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return

        path = urls[0].toLocalFile()
        if path:
            self._open_file(path)
            event.acceptProposedAction()

    def _sync_title_section_width(self, logical_index: int, _old: int, new: int) -> None:
        """
        synchronize title section width to `new`
        Args:
            logical_index:
            _old:
            new:

        Returns:

        """
        self.title_view.setColumnWidth(logical_index, new)

    def _sync_title_view_geometry(self) -> None:
        cols = self.table_model.columnCount()
        for c in range(cols):
            self.title_view.setColumnWidth(c, self.table_view.columnWidth(c))
        self._sync_title_corner_geometry()

    def _sync_title_corner_geometry_(self) -> None:
        self.title_corner.setFixedWidth(self.table_view.verticalHeader().width())
        self.title_corner.setFixedHeight(self.title_view.height())

    def _sync_title_corner_geometry(self) -> None:
        """
        Synchronize the corner of the title row with the data area.
        This method ensures that the left part of the title row
        aligns with the left part of the data area (i.e., the row header).
        """
        self.title_corner.setFixedWidth(self.table_view.verticalHeader().width())
        h = self.title_view.rowHeight(0) + self.title_view.frameWidth() * 2 if self.title_model.rowCount() > 0 else self.title_view.height()
        self.title_corner.setFixedHeight(h)
        self.title_bar.setFixedHeight(self.title_view.height())

    def _on_table_viewport_resized(self, old_size, new_size) -> None:
        """

        Args:
            old_size:
            new_size:

        Returns:

        """
        if self._geometry_scaling:
            return

        if old_size.width() <= 0 or old_size.height() <= 0:
            return

        sx = new_size.width() / old_size.width()
        sy = new_size.height() / old_size.height()

        self._geometry_scaling = True
        try:
            self._scale_table_geometry(sx, sy)
        finally:
            self._geometry_scaling = False

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "使用说明",
            ">> 本程序包含数据表、控制台（console）和输出面板三个部分 <<\n\n"
            "1. 数据列表主要仅用于可视化和选择数据，支持区域复制粘贴和双击修改单个值。\n\n"
            "2. 主要计算和操作部分在控制台中进行，变量`data`指向数据列表中的数据，可作为numpy的数组对象被使用。"
            "控制台支持输入所有 Python 3.12、NumPy 2.4.2 和 Matplotlib 2.10.8 的语法和指令，可自由编程。其中，\n"
            "    * NumPy的指令以 `np.` 开头（如`np.sum(data)`，对整个data的数据求和），\n"
            "    * Matplotlib的指令以 `plt.` 开头（如`plt.matshow(data[:5, :5]); plt.show()`，展示整个data前5行5列数据的热图），\n"
            "    * Python的原生指令无前缀（如`print(len(data))`，打印data的行数）。\n\n"
            "3. 输出、警告和报错信息打印在输出面板。对单行命令输出面板直接显示结果，多行命令则必须手动print相应变量才能显示结果。\n\n"
            "4. 区域选择方法\n"
            "    在表格中选中区域后按空格，可将矩形选区写入控制台中。\n"
            "    data[...] 的字体颜色与表格高亮颜色对应。\n\n"
            "本程序还有以下特有函数：\n"
            "    addr(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None, "
            "向第`i`行（从0开始计数）的上方（direct='<'）/下方（direct='>'，默认值）添加`size`行新行，初始以0填充。\n"
            "    addc(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None, "
            "向第`i`列（从0开始计数）的左侧（direct='<'）/右侧（direct='>'，默认值）添加`size`列新列，初始以0填充。\n"
            "    delr(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None, "
            "向第`i`行（从0开始计数）的上方（direct='<'）/下方（direct='>'）删除`size`行。\n"
            "    delc(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None, `delr`的列版本。"
        )

    def show_warning(self, text: str) -> None:
        self.console.append_output(f"[warning] {text}")

    def _show_table_context_menu(self, pos) -> None:
        """
        Show the menu
        Args:
            pos:

        Returns:

        """
        rows, cols = self._selected_rows_cols()  # get the selected rows/cols

        menu = QMenu(self)

        act_insert_row = menu.addAction("插入行")
        act_insert_col = menu.addAction("插入列")

        menu.addSeparator()

        act_delete_rows = menu.addAction("删除整行")
        act_delete_cols = menu.addAction("删除整列")

        menu.addSeparator()

        # 预留项
        act_reserved_1 = menu.addAction("更多编辑功能（预留）")
        act_reserved_1.setEnabled(False)
        act_reserved_2 = menu.addAction("格式功能（预留）")
        act_reserved_2.setEnabled(False)

        # deactivate options if no valide zone selected.
        has_row = bool(rows)
        has_col = bool(cols)
        act_insert_row.setEnabled(has_row)
        act_insert_col.setEnabled(has_col)
        act_delete_rows.setEnabled(has_row)
        act_delete_cols.setEnabled(has_col)

        action = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        if action is act_insert_row:
            self._insert_row_below_selection()
        elif action is act_insert_col:
            self._insert_col_right_selection()
        elif action is act_delete_rows:
            self._delete_selected_rows()
        elif action is act_delete_cols:
            self._delete_selected_cols()

    def update_status(self) -> None:
        rows, cols = self.backend.shape
        sel = self.backend.current_status_rect()
        self.status.showMessage(f"shape=({rows}, {cols}) | selection={sel}")

    def _parse_clipboard_table(self, text: str) -> np.ndarray:
        """
        Parse clipboard text into a 2D float ndarray.

        Row separator:
            any newline style supported by str.splitlines(), including
            POSIX '\n' and Windows '\r\n'.

        Column separator:
            spaces, tabs, commas, or mixed runs of them.
        """
        if not text or not text.strip():
            raise ValueError("剪贴板为空")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("剪贴板中没有可解析的数据")

        values: list[list[float]] = []
        width: int | None = None

        for line in lines:
            parts = [p for p in re.split(r"[,\t ]+", line.strip()) if p]
            if not parts:
                continue

            row = [float(x) for x in parts]

            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("各行列数不一致，无法解析为规则表格")

            values.append(row)

        if not values:
            raise ValueError("未解析出任何数值")

        return np.asarray(values, dtype=float)

    def _copy_from_table(self) -> None:
        indexes = self.table_view.selectionModel().selectedIndexes()
        if not indexes:
            return

        cells = {(idx.row(), idx.column()) for idx in indexes}
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1

        full = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
        if cells != full:
            self.show_warning("当前选区不是矩形，暂不支持复制为规则表格字符串")
            return

        arr = self.backend.data[r0:r1, c0:c1]

        lines = []
        for row in arr:
            parts = []
            for x in row:
                if isinstance(x, (float, np.floating)):
                    parts.append(f"{x:.8g}")
                else:
                    parts.append(str(x))
            lines.append(" ".join(parts))

        QApplication.clipboard().setText("\n".join(lines))

    def _paste_into_table(self) -> None:
        """
        To trigger the signal that paste the data area.
        Returns:

        """
        text = QApplication.clipboard().text()
        try:
            arr = self._parse_clipboard_table(text)
        except Exception as e:
            self.show_warning(f"剪贴板解析失败: {e}")
            return

        indexes = self.table_view.selectionModel().selectedIndexes()

        if indexes:
            cells = {(idx.row(), idx.column()) for idx in indexes}
            rows_ = [r for r, _ in cells]
            cols_ = [c for _, c in cells]
            r0, r1 = min(rows_), max(rows_) + 1
            c0, c1 = min(cols_), max(cols_) + 1

            full = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}

            # 矩形选区
            if cells == full:
                self.backend.set_block_to_region(r0, r1, c0, c1, arr)
                return

            # 非矩形选区：只允许单值填充
            if arr.shape == (1, 1):
                self.backend.fill_block(cells, arr.item())
                return

            self.show_warning("非矩形选区只能粘贴单个值")
            return

        current = self.table_view.currentIndex()
        if current.isValid():
            start_row = current.row()
            start_col = current.column()
        else:
            start_row = 0
            start_col = 0

        self.backend.set_block_at(start_row, start_col, arr)

    def _on_view_selection_changed(self, *_args) -> None:
        indexes = self.table_view.selectionModel().selectedIndexes()
        cells = {(idx.row(), idx.column()) for idx in indexes}
        self.backend.set_mouse_selection(cells)

    def _write_selection_to_console(self) -> None:
        """
        write the string expression of the mouse selections to the console.
        Returns:

        """
        cells = self.table_model.mouse_cells
        if not cells:
            return

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rs, re = min(rows), max(rows) + 1
        cs, ce = min(cols), max(cols) + 1

        full = {(r, c) for r in range(rs, re) for c in range(cs, ce)}
        if full != cells:
            self.show_warning("当前鼠标选区不是矩形，无法直接写成单个 data[r0:r1, c0:c1] 切片")
            return

        expr = f"data[{rs}:{re}, {cs}:{ce}]"

        self.table_view.clearSelection()
        self.backend.clear_mouse_selection()
        self.console.append_input(expr)

    def _scale_table_geometry(self, sx: float, sy: float) -> None:
        """
        Scale cell sizes (column widths and row heights) based on the given
        width and height scaling factors (sx, sy).
        """
        # Limit the scaling factors to avoid extreme scaling
        #sx = max(0.5, min(2.0, sx))
        #sy = max(0.5, min(2.0, sy))

        # Get the number of columns and rows
        cols = self.table_model.columnCount()
        rows = self.table_model.rowCount()

        # Scale column widths according to the horizontal scaling factor (sx)
        for c in range(cols):
            old_width = self.table_view.columnWidth(c)
            new_width = max(0, round(old_width * sx))  # Minimum column width is 40
            self.table_view.setColumnWidth(c, new_width)
            self.title_view.setColumnWidth(c, new_width)

        # Scale row heights according to the vertical scaling factor (sy)
        for r in range(rows):
            old_height = self.table_view.rowHeight(r)
            new_height = max(0, round(old_height * sy))  # The Minimum row height is 18
            self.table_view.setRowHeight(r, new_height)

        # Scale title row height separately
        if self.title_model.rowCount() > 0:
            old_title_h = self.title_view.rowHeight(0)
            new_title_h = max(20, round(old_title_h * sy))  # minimum title height is 20
            self.title_view.setRowHeight(0, new_title_h)
            self.title_view.setFixedHeight(
                new_title_h + self.title_view.frameWidth() * 2
            )

        # Ensure title row and data area left alignment
        self.title_view.resizeRowsToContents()
        self._sync_title_corner_geometry()
        self.title_bar.updateGeometry()

    def eventFilter(self, obj, event):
        if obj is self.table_view.viewport():
            if event.type() == QEvent.Wheel and (event.modifiers() & Qt.ControlModifier):
                delta = event.angleDelta().y()
                step = 1 if delta > 0 else -1
                self.zoom_cell_size(step)
                return True

            #if event.type() == QEvent.Resize and not self._geometry_scaling:
            #    old_size = event.oldSize()
            #    new_size = event.size()
#
            #    if old_size.width() > 0 and old_size.height() > 0:
            #        sx = new_size.width() / old_size.width()
            #        sy = new_size.height() / old_size.height()
#
            #        self._geometry_scaling = True
            #        try:
            #            self._scale_table_geometry(sx, sy)
            #        finally:
            #            self._geometry_scaling = False

        return super().eventFilter(obj, event)

    def set_table_font_size(self, size: int) -> None:
        self._font_size_now = max(6, min(72, int(size)))
        self._apply_view_metrics()

    def _on_status_selection_changed(self, rs: int, re: int, cs: int, ce: int) -> None:
        self.update_status()