from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Any

import numpy as np
from PySide6.QtCore import QObject, Signal


FLOAT_ARRAY_FORMAT = {
    'max_line_width': 1024,
    'precision': 8,
    'separator': '  ',
    'sign': ' ',
    'floatmode': 'fixed',
    'threshold': 21_4748_3647,
    'formatter': {'float_kind': lambda x: f'{x:0< 16.8f}', 'int_kind': lambda x: f'{x:> 16d}'}
}

INT_ARRAY_FORMAT = {
    'max_line_width': 1024,
    'separator': '  ',
    'sign': ' ',
    'threshold': 21_4748_3647,
    'formatter': {'int_kind': lambda x: f'{x:> 16d}'}
}

STRING_ARRAY_FORMAT = {
    'max_line_width': 1024,
    'separator': '  ',
    'sign': ' ',
    'threshold': 21_4748_3647,
    'formatter': {'str_kind': lambda x: f'{x}'}
}


@dataclass
class RectSelection:
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    def is_valid(self) -> bool:
        return self.row_start < self.row_end and self.col_start < self.col_end

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.row_start, self.row_end, self.col_start, self.col_end

    def __str__(self) -> str:
        return (
            f"rows[{self.row_start}:{self.row_end}], "
            f"cols[{self.col_start}:{self.col_end}]"
        )


class DataArray(np.ndarray):
    def __new__(cls, shape, dtype=np.float64, buffer=None, offset=0, strides=None, order=None):
        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset, strides, order)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __setitem__tmp(self, key, value):  # Fixme Could not succeed to auto-extend
        # 规范化 key 用于计算形状
        norm_key = self._normalize_key(key)
        value_arr = np.asarray(value)
        # 计算所需最小形状
        required_shape = self._compute_required_shape(norm_key, value_arr)
        # 若当前形状不足，则扩容
        if required_shape != self.shape:
            self[:] = np.ascontiguousarray(self)
            self.resize(required_shape, refcheck=True)
        # 使用原始 key 进行赋值（保留广播等行为）
        super().__setitem__(key, value)

    def _normalize_key(self, key):
        """将索引规范化为长度等于数组维度的元组"""
        ndim = self.ndim
        # 非元组：作为第一维索引，其余维为 :
        if not isinstance(key, tuple):
            return (key,) + (slice(None),) * (ndim - 1)

        # 处理元组中的省略号
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_fill = ndim - (len(key) - 1)
            if n_fill < 0:
                raise IndexError("省略号表示的维度过多")
            parts = list(key)
            parts[idx:idx+1] = [slice(None)] * n_fill
            return tuple(parts)

        # 元组长度不足时，末尾补 :
        if len(key) < ndim:
            return key + (slice(None),) * (ndim - len(key))

        # 长度相等或超出（超出部分通常为 newaxis）
        if len(key) > ndim:
            if any(k is None for k in key):
                raise NotImplementedError("newaxis 自动扩容未实现")
            # 理论上不应出现，截断至 ndim
            return key[:ndim]
        return key

    def _compute_required_shape(self, key, value):
        """根据规范化的 key 和 value 计算所需的最小形状"""
        current_shape = self.shape
        ndim = len(current_shape)
        if len(key) != ndim:
            raise ValueError(f"规范化后索引维度 {len(key)} 与数组维度 {ndim} 不匹配")

        # 将 value 的形状对齐到 ndim（右对齐，左侧补1）
        value_ndim = value.ndim
        if value_ndim < ndim:
            # 左侧补1
            value_shape_pad = (1,) * (ndim - value_ndim) + value.shape
        else:
            # 只考虑最后 ndim 维，前面的必须为1或可广播，这里忽略
            value_shape_pad = value.shape[-ndim:]

        required = list(current_shape)
        for axis, idx in enumerate(key):
            need = self._required_length_for_axis(idx, axis, current_shape[axis], value_shape_pad[axis])
            required[axis] = max(required[axis], need)
        return tuple(required)

    def _required_length_for_axis(self, idx, axis, current_len, value_dim):
        """根据索引和 value 在该维度的尺寸，计算该维度所需的最小数组长度"""
        # 1. 整数
        if isinstance(idx, (int, np.integer)):
            if idx >= 0:
                need = idx + 1
            else:
                need = -idx  # 负索引要求长度至少为其绝对值
            return max(need, current_len)

        # 2. 切片（仅支持正步长）
        elif isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if step is None:
                step = 1
            if step <= 0:
                raise NotImplementedError("负步长切片自动扩容未实现")

            # 处理 start
            need = current_len
            if start is not None:
                if isinstance(start, int):
                    if start >= 0:
                        if start >= current_len:
                            need = max(need, start + 1)
                    else:  # start 为负
                        # 负 start 对应的正索引为 current_len + start，若为负则越界
                        pos_start = current_len + start
                        if pos_start < 0:
                            # 需要扩容使得 current_len >= -start
                            need = max(need, -start)
            # 处理 stop
            if stop is not None:
                if isinstance(stop, int):
                    if stop >= 0:
                        if stop > current_len:
                            need = max(need, stop)
                    else:  # stop 为负，对应的正索引为 current_len + stop，若为负则无元素，不影响
                        pass
            else:
                # stop 为 None：根据 value_dim 动态确定所需长度
                # 切片区域应从 start 开始，取 value_dim 个元素（步长为 step）
                # 实际 start 可能为 None，需转换
                if start is None:
                    start_val = 0
                elif isinstance(start, int):
                    start_val = start if start >= 0 else current_len + start
                else:
                    start_val = 0  # 非整数 start 暂不处理
                # 计算最后一个元素的索引
                last_idx = start_val + (value_dim - 1) * step
                if last_idx >= current_len:
                    need = max(need, last_idx + 1)
            return need

        # 3. 整数数组 / 列表
        elif isinstance(idx, (list, np.ndarray)):
            idx_arr = np.asarray(idx)
            if idx_arr.dtype.kind in 'iu':
                # 正索引部分
                pos = idx_arr[idx_arr >= 0]
                max_pos = np.max(pos) if pos.size > 0 else -1
                # 负索引部分
                neg = idx_arr[idx_arr < 0]
                if neg.size > 0:
                    min_neg = np.min(neg)
                    need_neg = -min_neg
                else:
                    need_neg = 0
                need = max(max_pos + 1, need_neg)
                return max(need, current_len)
            elif idx_arr.dtype.kind == 'b':
                # 布尔数组：所需长度至少为布尔数组的形状对应维度
                if idx_arr.ndim == 1:
                    # 一维布尔掩码：该维度长度至少等于数组长度
                    return max(len(idx_arr), current_len)
                else:
                    # 多维布尔掩码：如果 axis 在布尔数组的维度内，则取该维度长度
                    if axis < idx_arr.ndim:
                        return max(idx_arr.shape[axis], current_len)
                    else:
                        return current_len
            else:
                return current_len

        # 4. 省略号（已展开，不会直接出现）
        elif idx is Ellipsis:
            return current_len

        # 5. None (newaxis) 已在前置检查中抛出异常
        elif idx is None:
            raise NotImplementedError("newaxis 自动扩容未实现")

        # 其他情况（如 slice(None) 已在切片中处理）
        return current_len


class DataBackend(QObject):
    data_changed = Signal()
    dtype_changed = Signal(str)
    mouse_selection_changed = Signal(object)
    preview_selections_changed = Signal(object)
    status_selection_changed = Signal(int, int, int, int)
    warning_emitted = Signal(str)

    def __init__(self, initial_data: Optional[np.ndarray] = None) -> None:
        super().__init__()
        if initial_data is None:
            initial_data = np.zeros((500, 10), dtype=np.float64)

        arr = DataArray(
            initial_data.shape,
            dtype=initial_data.dtype,
            buffer=initial_data,
            offset=0,
            strides=initial_data.strides,
            order='C',
        )
        arr = np.atleast_2d(arr)
        if arr.ndim != 2:
            raise ValueError("当前仅支持二维 ndarray")

        self._data = arr.astype(np.float64, copy=False)
        self._status_rect = RectSelection(0, 1, 0, 1)
        self._mouse_cells: set[tuple[int, int]] = set()
        self._preview_groups: list[set[tuple[int, int]]] = []
        self._data_type_now = 'float64'
        self._py_type = float  # record the type of python obj
        self._np_type = np.float64  # record the type of numpy.ndarray
        # here is the allowed types of the table. in the value tuple, the first is the converter/type of python build-in obj
        # and the second is the dtype of numpy.ndarray class.
        self.ALLOWED_TYPES = {
            'float64': (float, np.float64),
            'float32': (float, np.float32),
            'int8': (int, np.int8),
            'int16': (int, np.int16),
            'int32': (int, np.int32),
            'int64': (int, np.int64),
            'uint8': (int, np.uint8),
            'uint16': (int, np.uint16),
            'uint32': (int, np.uint32),
            'uint64': (int, np.uint64),
            'int': (int, np.int64),
            'float': (float, np.float32),
            'double': (float, np.float64),
            'bool': (bool, np.bool_),
            'object': (lambda x: x, object),
        }

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        arr = np.atleast_2d(arr)
        if arr.ndim != 2:
            raise ValueError("当前仅支持二维 ndarray")
        self._data = arr
        self._sync_dtype_trackers_from_array()
        self.data_changed.emit()

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def current_dtype_name(self) -> str:
        return self._data_type_now

    def current_status_rect(self) -> RectSelection:
        return self._status_rect

    def current_selection_slices(self) -> tuple[slice, slice]:
        s = self._status_rect
        return slice(s.row_start, s.row_end), slice(s.col_start, s.col_end)

    def set_status_rect(self, cells: set[tuple[int, int]]) -> None:
        rect = self._infer_rect_from_cells(cells)
        if rect is not None:
            self._status_rect = rect
            self.status_selection_changed.emit(*rect.as_tuple())

    def set_mouse_selection(self, cells: Iterable[tuple[int, int]]) -> None:
        rows, cols = self.shape
        valid = {
            (r, c) for (r, c) in cells
            if 0 <= r < rows and 0 <= c < cols
        }
        self._mouse_cells = valid
        self.mouse_selection_changed.emit(set(valid))
        self.set_status_rect(valid)

    def clear_mouse_selection(self) -> None:
        self._mouse_cells = set()
        self.mouse_selection_changed.emit(set())

    def set_preview_groups(self, groups: list[set[tuple[int, int]]]) -> None:
        rows, cols = self.shape
        normalized: list[set[tuple[int, int]]] = []
        for group in groups:
            valid = {
                (r, c) for (r, c) in group
                if 0 <= r < rows and 0 <= c < cols
            }
            if valid:
                normalized.append(valid)
        self._preview_groups = normalized
        self.preview_selections_changed.emit([set(g) for g in normalized])

        if normalized:
            self.set_status_rect(normalized[-1])

    def clear_preview_groups(self) -> None:
        self._preview_groups = []
        self.preview_selections_changed.emit([])

    def load_csv(self, path: str) -> None:
        """
        load saved csv file
        Args:
            path:

        Returns:

        """
        with open(path, 'r') as f:
            _raw_data = f.readlines()
        if _raw_data[0].startswith('#'):
            _start_ptr = 1
        else:
            _start_ptr = 0

        _data = [_.replace(',', ' ').strip().split() for _ in _raw_data[_start_ptr:]]
        arr = np.asarray(_data, dtype=self._np_type)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise RuntimeError(f'Failed to read file {path}: Invalid array shape {arr.shape}.')
        self.data = arr

    def save_csv(self, path: str, title: None|list|str = None) -> None:
        """
        save data to a csv file.
        Args:
            path:
            title:

        Returns:

        """
        # parse data
        if self._data_type_now.startswith(('f', 'd'), ):
            formatter = FLOAT_ARRAY_FORMAT
        elif self._data_type_now.startswith(('i', 'u'), ):
            formatter = INT_ARRAY_FORMAT
        else:
            formatter = STRING_ARRAY_FORMAT
        arr_string = np.array2string(self._data, **formatter).replace('[', ' ').replace(']', ' ')
        # parse title
        if title is None:
            with open(path, "w") as f:
                f.write(arr_string)
        elif isinstance(title, str):
            if not title.startswith('#'):
                title = '#' + title
            with open(path, "w") as f:
                f.write(title)
                f.write(arr_string)
        elif isinstance(title, list):
            with open(path, "w") as f:
                f.write("#" + ",".join(title) + '\n')
                f.write(arr_string)
        else:
            raise TypeError(f'Invalid title type: {type(title)}')


    def ensure_shape(self, min_rows: int, min_cols: int) -> None:
        """
        To automatically extend the data shape if given row & column number are larger than them of current shape.
        Args:
            min_rows:
            min_cols:

        Returns:

        """
        rows, cols = self.shape
        new_rows = max(rows, min_rows)
        new_cols = max(cols, min_cols)
        if new_rows == rows and new_cols == cols:
            return

        new_data = np.zeros((new_rows, new_cols), dtype=self._data.dtype)
        new_data[:rows, :cols] = self._data
        self._data = new_data
        self.data_changed.emit()

    def addr(self, i: int, size: int) -> None:
        """
        Add rows starting at row i with size `size`.
        Args:
            i:
            size:

        Returns:

        """
        if size <= 0:
            return
        rows, cols = self.shape
        if i < -1 or i >= rows:
            raise IndexError("行插入位置越界")
        insert_at = i + 1
        extra = np.zeros((size, cols), dtype=self._data.dtype)
        self._data = np.vstack([
            self._data[:insert_at, :],
            extra,
            self._data[insert_at:, :]
        ])
        self.data_changed.emit()

    def addc(self, i: int, size: int) -> None:
        """
        Add columns starting at column i with size `size`.
        Args:
            i:
            size:

        Returns:

        """
        if size <= 0:
            return
        rows, cols = self.shape
        if i < -1 or i >= cols:
            raise IndexError("列插入位置越界")
        insert_at = i + 1
        extra = np.zeros((rows, size), dtype=self._data.dtype)
        self._data = np.hstack([
            self._data[:, :insert_at],
            extra,
            self._data[:, insert_at:]
        ])
        self.data_changed.emit()

    def delr(self, i: int, size: int) -> None:
        """
        Delete rows starting at row i with size `size`.
        Args:
            i:
            size:

        Returns:

        """
        if size <= 0:
            return
        rows, _ = self.shape
        if i < -1 or i >= rows:
            raise IndexError("行删除位置越界")
        start = i + 1
        end = start + size
        if start >= rows:
            return
        end = min(end, rows)
        self._data = np.vstack([
            self._data[:start, :],
            self._data[end:, :]
        ])
        self.data_changed.emit()

    def delc(self, i: int, size: int) -> None:
        """
        Delete columns starting at column i with size `size`.
        Args:
            i:
            size:

        Returns:

        """
        if size <= 0:
            return
        _, cols = self.shape
        if i < -1 or i >= cols:
            raise IndexError("列删除位置越界")
        start = i + 1
        end = start + size
        if start >= cols:
            return
        end = min(end, cols)
        self._data = np.hstack([
            self._data[:, :start],
            self._data[:, end:]
        ])
        self.data_changed.emit()

    def change_types(self, dtype):
        """
        Change data type to the given `dtype` which must be in `self.ALLOWED_TYPES`.
        Args:
            dtype:

        Returns:

        """
        if dtype not in self.ALLOWED_TYPES:
            raise ValueError(f'Invalid data type: {dtype}.')
        self._data_type_now = dtype
        self._py_type, self._np_type = self.ALLOWED_TYPES[dtype]
        self._data = self._data.astype(self._np_type)
        self._sync_dtype_trackers_from_array()
        self.data_changed.emit()

    def set_data_with_auto_expand(self, index: Any, value: Any) -> None:
        target_rows, target_cols = self._infer_required_shape_for_assignment(index)
        if target_rows is not None and target_cols is not None:
            self.ensure_shape(target_rows, target_cols)
        self._data[index] = value
        self.data_changed.emit()

    def set_value(self, row: int, col: int, value: Any) -> None:
        """
        Set a single value of a cell at (row, col).
        Args:
            row:
            col:
            value:

        Returns:

        """
        self.ensure_shape(row + 1, col + 1)
        self._data[row, col] = self._py_type(value)
        self.data_changed.emit()

    def set_block_at(self, row: int, col: int, values) -> None:
        """
        Set blocked values started from (row, col), and automatically overwrite the following zone in the shape of `values`.
        Args:
            row:
            col:
            values:

        Returns:

        """
        arr = np.asarray(values, dtype=self._np_type)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)

        h, w = arr.shape
        self.ensure_shape(row + h, col + w)
        self._data[row:row + h, col:col + w] = arr
        self.data_changed.emit()

    def set_block_to_region(
            self,
            row0: int,
            row1: int,
            col0: int,
            col1: int,
            values
    ) -> None:
        """
        Set the values into the region of (row0:row1, col0:col1), requiring the shape of `values` matching (row1 - row0, col1 - col0).
        Args:
            row0:
            row1:
            col0:
            col1:
            values:

        Returns:

        """
        # data convert and at least 2d
        arr = np.asarray(values, dtype=self._np_type)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)

        target_h = row1 - row0
        target_w = col1 - col0

        if target_h <= 0 or target_w <= 0:
            raise ValueError("目标区域为空")

        # 单值允许填充整个区域
        if arr.shape == (1, 1):
            self.ensure_shape(row1, col1)
            self._data[row0:row1, col0:col1] = arr.item()
            self.data_changed.emit()
            return

        # 非单值必须严格匹配
        if arr.shape != (target_h, target_w):
            raise ValueError(
                f"粘贴数据形状 {arr.shape} 与选区形状 {(target_h, target_w)} 不匹配"
            )

        self.ensure_shape(row1, col1)
        self._data[row0:row1, col0:col1] = arr
        self.data_changed.emit()

    def fill_block(self, cells: set[tuple[int, int]], value: Any) -> None:
        """
        Fill a single `value` to the range specified by `cells`.
        Args:
            cells:
            value:

        Returns:

        """
        for r, c in cells:
            self._data[r, c] = self._py_type(value)
        self.data_changed.emit()

    def _infer_required_shape_for_assignment(
        self, index: Any
    ) -> tuple[int | None, int | None]:
        rows, cols = self.shape

        if isinstance(index, tuple) and len(index) == 2:
            r_need = self._required_axis_size(index[0], rows)
            c_need = self._required_axis_size(index[1], cols)
            return r_need, c_need

        r_need = self._required_axis_size(index, rows)
        return r_need, cols

    def _required_axis_size(self, sel: Any, current_size: int) -> int:
        if isinstance(sel, slice):
            stop = sel.stop
            if stop is None:
                return current_size
            if stop < 0:
                return current_size
            return max(current_size, int(stop))

        if isinstance(sel, (int, np.integer)):
            idx = int(sel)
            if idx < 0:
                return current_size
            return max(current_size, idx + 1)

        arr = np.asarray(sel)
        if arr.dtype == bool:
            return current_size

        if arr.size == 0:
            return current_size

        flat = arr.reshape(-1)
        max_idx = None
        for x in flat:
            xi = int(x)
            if xi >= 0:
                max_idx = xi if max_idx is None else max(max_idx, xi)

        if max_idx is None:
            return current_size
        return max(current_size, max_idx + 1)

    def _infer_rect_from_cells(
        self, cells: set[tuple[int, int]]
    ) -> Optional[RectSelection]:
        if not cells:
            return None
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rs, re = min(rows), max(rows) + 1
        cs, ce = min(cols), max(cols) + 1
        full = {
            (r, c)
            for r in range(rs, re)
            for c in range(cs, ce)
        }
        if full == cells:
            return RectSelection(rs, re, cs, ce)
        return None

    def _sync_dtype_trackers_from_array(self) -> None:
        """
        synchronize if array dtype was changed.
        Returns:

        """
        dt = np.dtype(self._data.dtype)
        key = "object" if dt == np.dtype(object) else dt.name

        if key in self.ALLOWED_TYPES:
            self._data_type_now = key
            self._py_type, self._np_type = self.ALLOWED_TYPES[key]
        else:
            # 允许显示未知 dtype，但不把它加入下拉选项
            self._data_type_now = key
            self._py_type = lambda x: x
            self._np_type = dt.type

        self.dtype_changed.emit(self._data_type_now)
