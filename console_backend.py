from __future__ import annotations

import ast
import builtins
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from data_backend import DataBackend



def safe_import(name, *args, **kwargs):
    # 允许导入 numpy 及其子模块（如 numpy.core._methods）
    # can be extended in future
    if name.startswith('numpy.') or name == 'numpy':
        return builtins.__import__(name, *args, **kwargs)
    # 禁止其他所有导入
    raise ImportError(f"Import of '{name}' is not allowed")

SAFE_BUILTINS = {
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "bool": builtins.bool,
    "dict": builtins.dict,
    "enumerate": builtins.enumerate,
    "filter": builtins.filter,
    "float": builtins.float,
    "int": builtins.int,
    "__import__": safe_import,
    "len": builtins.len,
    "list": builtins.list,
    "map": builtins.map,
    "max": builtins.max,
    "min": builtins.min,
    "object": builtins.object,
    "print": builtins.print,
    "range": builtins.range,
    "repr": builtins.repr,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "set": builtins.set,
    "slice": builtins.slice,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "tuple": builtins.tuple,
    "type": builtins.type,
    "zip": builtins.zip,
}

class _DataIndexCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[ast.Subscript] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "data":
            self.nodes.append(node)
        self.generic_visit(node)


class ConsoleBackend:
    def __init__(self, backend: DataBackend) -> None:
        self.backend = backend
        self.user_vars: dict[str, Any] = {}

    def build_env(self) -> dict[str, Any]:
        """
        Manage the vars containing build-in and user-defined variables.
        Returns:

        """
        rows_slice, cols_slice = self.backend.current_selection_slices()

        env = dict(self.user_vars)
        env["np"] = np
        env["plt"] = plt
        env["data"] = self.backend.data
        env["save_csv"] = self.backend.save_csv
        env["load_csv"] = self.backend.load_csv
        env["sel"] = (rows_slice, cols_slice)
        env["selected"] = self.backend.data[rows_slice, cols_slice]
        env["addr"] = self.backend.addr
        env["addc"] = self.backend.addc
        env["delr"] = self.backend.delr
        env["delc"] = self.backend.delc
        return env

    def execute(self, code: str) -> str:
        """
        Execute the code and return the result
        Args:
            code (str): The code to execute
        """
        code = code.strip()
        if not code:
            return ""

        if code == "clear":
            self.user_vars.clear()
            return "[info] 已清除全部缓存变量"

        if code.startswith("clear "):
            var_name = code[6:].strip()
            if not var_name:
                self.user_vars.clear()
                return "[info] 已清除全部缓存变量"
            if var_name in self.user_vars:
                del self.user_vars[var_name]
                return f"[info] 已清除变量 {var_name}"
            return f"[warning] 变量 {var_name} 不存在"

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        out = [f">>> {code}\n"]

        env = self.build_env()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                result = self._run_code(code, env)

            self._store_user_vars(env)

            std_text = stdout_buffer.getvalue()
            err_text = stderr_buffer.getvalue()

            if std_text:
                out.append(std_text)
            if err_text:
                out.append(err_text)

            if result is not None:
                out.append(self._format_result(result))
                if not out[-1].endswith("\n"):
                    out[-1] += "\n"

            self.preview_selection_from_code(code, emit_warning=True)

        except Exception:
            out.append(traceback.format_exc())

        return "".join(out).rstrip()

    def _run_code(self, code: str, env: dict[str, Any]) -> Any:
        globals_dict = {"__builtins__": SAFE_BUILTINS}
        if "\n" in code:
            compiled = compile(code, "<console>", "exec")
            exec(compiled, globals_dict, env)
            return None
        else:
            try:
                compiled = compile(code, "<console>", "eval")
                return eval(compiled, globals_dict, env)
            except SyntaxError:  # Catch the assignment situations
                compiled = compile(code, "<console>", "exec")
                exec(compiled, globals_dict, env)
                self.backend.data = env["data"]
                return None

    def _store_user_vars(self, env: dict[str, Any]) -> None:
        """
        A container to store user-defined variables.
        Args:
            env:

        Returns:

        """
        reserved = {
            "np", "plt", "data", "sel", "selected",
            "addr", "addc", "delr", "delc",
        }
        new_vars = {}
        for k, v in env.items():
            if k in reserved or k.startswith("_"):
                continue
            new_vars[k] = v
        self.user_vars = new_vars

    def _format_result(self, result: Any) -> str:
        if isinstance(result, np.ndarray):
            return np.array2string(result, precision=4, suppress_small=True)
        return repr(result)

    def preview_selection_from_code(self, code: str, emit_warning: bool = False) -> None:
        code = code.strip()
        if not code:
            self.backend.clear_preview_groups()
            return

        try:
            subscript_nodes = self.extract_data_subscript_nodes(code)
        except SyntaxError:
            self.backend.clear_preview_groups()
            return

        if not subscript_nodes:
            self.backend.clear_preview_groups()
            return

        groups: list[set[tuple[int, int]]] = []
        warnings: list[str] = []

        for node in subscript_nodes:
            try:
                cells = self.resolve_cells_from_subscript_node(node)
                if cells:
                    groups.append(cells)
                elif emit_warning:
                    warnings.append(f"{ast.unparse(node)} 未映射到可显示单元格")
            except Exception as e:
                if emit_warning:
                    warnings.append(f"{ast.unparse(node)} 无法解析索引: {e}")

        self.backend.set_preview_groups(groups)

        if emit_warning:
            for msg in warnings:
                self.backend.warning_emitted.emit(msg)

    def extract_data_subscript_nodes(self, code: str) -> list[ast.Subscript]:
        tree = ast.parse(code, mode="exec")
        collector = _DataIndexCollector()
        collector.visit(tree)
        return collector.nodes

    def resolve_cells_from_subscript_node(self, node: ast.Subscript) -> set[tuple[int, int]]:
        rows, cols = self.backend.shape
        slice_node = node.slice

        if isinstance(slice_node, ast.Tuple):
            if len(slice_node.elts) != 2:
                raise ValueError("当前仅支持二维索引")
            row_indices = self._normalize_axis_selector(slice_node.elts[0], rows)
            col_indices = self._normalize_axis_selector(slice_node.elts[1], cols)

            row_adv = self._is_advanced_index(slice_node.elts[0])
            col_adv = self._is_advanced_index(slice_node.elts[1])

            if row_adv and col_adv and len(row_indices) == len(col_indices):
                return {(r, c) for r, c in zip(row_indices, col_indices)}

            return {(r, c) for r in row_indices for c in col_indices}

        row_indices = self._normalize_axis_selector(slice_node, rows)
        return {(r, c) for r in row_indices for c in range(cols)}

    def _is_advanced_index(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Slice):
            return False
        value = self._eval_ast_node(node)
        if isinstance(value, (int, np.integer, slice)):
            return False
        arr = np.asarray(value)
        return True

    def _normalize_axis_selector(self, node: ast.AST, axis_size: int) -> list[int]:
        if isinstance(node, ast.Slice):
            s = slice(
                self._literal_or_eval(node.lower),
                self._literal_or_eval(node.upper),
                self._literal_or_eval(node.step),
            )
            return list(range(axis_size))[s]

        value = self._eval_ast_node(node)

        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if idx < 0:
                idx += axis_size
            if not (0 <= idx < axis_size):
                raise IndexError(f"索引越界: {idx}")
            return [idx]

        if isinstance(value, slice):
            return list(range(axis_size))[value]

        arr = np.asarray(value)

        if arr.dtype == bool:
            if arr.ndim != 1 or arr.shape[0] != axis_size:
                raise ValueError("bool mask 的长度与轴长度不匹配")
            return list(np.flatnonzero(arr))

        flat = arr.reshape(-1)
        indices = []
        for x in flat:
            idx = int(x)
            if idx < 0:
                idx += axis_size
            if not (0 <= idx < axis_size):
                raise IndexError(f"索引越界: {idx}")
            indices.append(idx)
        return indices

    def _eval_ast_node(self, node: ast.AST) -> Any:
        env = self.build_env()
        code = compile(ast.Expression(node), "<index>", "eval")
        return eval(code, {"__builtins__": SAFE_BUILTINS}, env)

    def _literal_or_eval(self, node: ast.AST | None) -> Any:
        if node is None:
            return None
        try:
            return ast.literal_eval(node)
        except Exception:
            return self._eval_ast_node(node)

