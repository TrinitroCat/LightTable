"""
The GUI visualization part.
"""

from __future__ import annotations

import ast
from typing import Optional
import re
import os

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, QEvent
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

        top = QHBoxLayout()
        self.title = QLabel("Console")
        self.run_button = QPushButton("执行")
        self.clear_after_submit = QCheckBox("提交后清空")
        self.clear_output_button = QPushButton("清空输出")
        # 新增的工具栏区域
        top = QHBoxLayout()
        self.title = QLabel("Console")
        self.run_button = QPushButton("执行")
        self.clear_after_submit = QCheckBox("提交后清空输入")
        self.clear_output_button = QPushButton("清空输出")

        # 添加调整字号的 QComboBox 控件
        self.font_size_combobox = QComboBox()
        self.font_size_combobox.addItems(["8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "32", "36", "40"])
        self.font_size_combobox.setCurrentText("16")  # Default font size: 20

        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.run_button)
        top.addWidget(self.clear_after_submit)
        top.addWidget(self.clear_output_button)
        top.addWidget(QLabel("字号: "))
        top.addWidget(self.font_size_combobox)  # Font config

        self.input_edit = ConsoleInput()
        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.adjust_font_size("16")
        self.input_edit.setPlaceholderText(
            "支持实时预览 data[...] 选区\n"
            "Shift+Enter 提交；表格中选区按 Space 写入 data[...] \n\n"
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

    def clear_output(self) -> None:
        self.output_edit.clear()

    def append_input(self, text: str) -> None:
        self.input_edit.insertPlainText(text)

        #self.input_edit.setPlainText(text)
        #self.input_edit.setFocus()
        #cursor = self.input_edit.textCursor()
        #cursor.movePosition(QTextCursor.End)
        #self.input_edit.setTextCursor(cursor)


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

        # 预留空组
        self.edit_group = RibbonGroup("编辑")
        self.view_group = RibbonGroup("视图")

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
        #self.table_view.horizontalHeader().hide()
        #self.table_view.setCornerButtonEnabled(False)
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
        self.ribbon = RibbonToolbar(list(self.backend.ALLOWED_TYPES.keys()))
        self.ribbon.set_dtype(self.backend.current_dtype_name)

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
        self.zoom_in_shortcut.activated.connect(lambda: self.set_table_font_size(self.table_view.font().pointSize() + 1))
        self.zoom_out_shortcut.activated.connect(lambda: self.set_table_font_size(self.table_view.font().pointSize() - 1))
        self.zoom_in_shortcut2 = QShortcut(QKeySequence.ZoomIn, self)
        self.zoom_out_shortcut2 = QShortcut(QKeySequence.ZoomOut, self)
        self.zoom_in_shortcut2.activated.connect(lambda: self.set_table_font_size(self.table_view.font().pointSize() + 1))
        self.zoom_out_shortcut2.activated.connect(lambda: self.set_table_font_size(self.table_view.font().pointSize() - 1))

        self._setup_connections()
        self._setup_menu()
        self.ribbon.set_current_file(self.current_file_name)
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
            "说明",
            "data[...] 的字体颜色与表格高亮颜色对应。\n"
            "表格中选区按空格，可把矩形选区写入 console。\n"
            "NumPy 现在需要显式写为 np.xxx。\n"
            "支持 delr/delc 作为 addr/addc 的逆操作。"
        )

    def show_warning(self, text: str) -> None:
        self.console.append_output(f"[warning] {text}")

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
        self.console.append_input(expr)

        self.table_view.clearSelection()
        self.backend.clear_mouse_selection()

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
                self.set_table_font_size(self.table_view.font().pointSize() + step)
                return True

            if event.type() == QEvent.Resize and not self._geometry_scaling:
                old_size = event.oldSize()
                new_size = event.size()

                if old_size.width() > 0 and old_size.height() > 0:
                    sx = new_size.width() / old_size.width()
                    sy = new_size.height() / old_size.height()

                    self._geometry_scaling = True
                    try:
                        self._scale_table_geometry(sx, sy)
                    finally:
                        self._geometry_scaling = False

        return super().eventFilter(obj, event)

    #def resizeEvent(self, event) -> None:
    #    """
    #    Event triggered when the window is resized. This will scale the table
    #    and title row proportionally.
    #    """
    #    super().resizeEvent(event)
#
    #    # Get the new size of the table's viewport
    #    new_size = self.table_view.viewport().size()
    #    old_size = self._last_viewport_size
#
    #    if old_size.width() > 0 and old_size.height() > 0:
    #        # Calculate scaling factors for width and height based on previous size
    #        sx = new_size.width() / old_size.width()
    #        sy = new_size.height() / old_size.height()
#
    #        # Apply scaling to table and title row
    #        self._scale_table_geometry(sx, sy)
#
    #    # Update the last viewport size for next resize event
    #    self._last_viewport_size = self.table_view.viewport().size()

    #def wheelEvent(self, event) -> None:
    #    """捕获鼠标滚轮事件，支持 Ctrl + 滚轮来调整表格大小"""
    #    if event.modifiers() == Qt.ControlModifier:
    #        delta = event.angleDelta().y()
    #        if delta > 0:
    #            current = self.table_view.font().pointSize()
    #            self.set_table_font_size(current + 1)  # 放大字体
    #        else:
    #            current = self.table_view.font().pointSize()
    #            self.set_table_font_size(current - 1)  # 缩小字体
#
    #    super().wheelEvent(event)
#
    #def keyPressEvent(self, event: QKeyEvent) -> None:
    #    """监听 Ctrl + + 和 Ctrl + - 来调整字号"""
    #    if event.modifiers() == Qt.ControlModifier:
    #        if event.key() == Qt.Key_Equal:  # Ctrl + "+"
    #            _current_size = self.table_view.font().pointSize()
    #            self.set_table_font_size(_current_size + 1)  # 放大字体
    #        elif event.key() == Qt.Key_Minus:  # Ctrl + "-"
    #            _current_size = self.table_view.font().pointSize()
    #            self.set_table_font_size(_current_size - 1)  # 缩小字体
    #    else:
    #        super().keyPressEvent(event)  # 其他事件继续传递

    def set_table_font_size(self, size: int) -> None:
        """
        Set the font size and the corresponding cell size synchronously.
        Args:
            size:

        Returns:

        """
        size = max(6, min(72, int(size)))

        old_size = max(6, self.table_view.font().pointSize())
        ratio = size / old_size

        font = self.table_view.font()
        font.setPointSize(size)
        self.table_view.setFont(font)
        self.title_view.setFont(font)

        fm = QFontMetrics(font)

        min_row_h = int(fm.height() * 1.75)
        min_title_h = int(fm.height() * 1.9)
        min_col_w = max(
            int(fm.horizontalAdvance("0000.000") + 18),
            int(fm.horizontalAdvance("标题 00") + 24),
        )

        # 数据区行高按比例缩放
        for row in range(self.table_model.rowCount()):
            old_h = self.table_view.rowHeight(row)
            new_h = max(min_row_h, round(old_h * ratio))
            self.table_view.setRowHeight(row, new_h)

        # 标题行高度单独设置
        old_title_h = self.title_view.rowHeight(0) if self.title_model.rowCount() > 0 else min_title_h
        self.title_view.setRowHeight(0, max(min_title_h, round(old_title_h * ratio)))

        # 列宽按比例缩放，并保证最小宽度
        for col in range(self.table_model.columnCount()):
            old_w = self.table_view.columnWidth(col)
            new_w = max(min_col_w, round(old_w * ratio))
            self.table_view.setColumnWidth(col, new_w)
            self.title_view.setColumnWidth(col, new_w)

        self.title_view.setFixedHeight(
            self.title_view.rowHeight(0) + self.title_view.frameWidth() * 2
        )
        self._sync_title_corner_geometry()

    def _on_status_selection_changed(self, rs: int, re: int, cs: int, ce: int) -> None:
        self.update_status()