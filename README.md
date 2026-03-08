# LightTable
A light table program that deals with list data by numpy, matplotlib, and python build-in operations.

使用说明：
>> 本程序包含数据表、控制台（console）和输出面板三个部分
1. 数据列表主要仅用于可视化和选择数据，支持区域复制粘贴和双击修改单个值。


2. 主要计算和操作部分在控制台中进行，变量`data`指向数据列表中的数据，可作为numpy的数组对象被使用。
控制台支持输入所有 Python 3.12、NumPy 2.4.2 和 Matplotlib 2.10.8 的语法和指令，可自由编程。
其中，
   * NumPy的指令以 `np.` 开头（如`np.sum(data)`，对整个data的数据求和）
   * Matplotlib的指令以 `plt.` 开头（如`plt.matshow(data[:5, :5]); plt.show()`，展示整个data前5行5列数据的热图）
   * Python的原生指令无前缀（如`print(len(data))`，打印data的行数）。


3. 输出、警告和报错信息打印在输出面板。对单行命令输出面板直接显示结果，多行命令则必须手动`print`相应变量才能显示结果。


4. 区域选择方法：
  在表格中选中区域后按**空格**，可将矩形选区写入控制台中。`data[...]` 的字体颜色与表格高亮颜色对应。


5. 本程序还有以下特有函数：
   * `addr(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None`, "向第`i`行（从0开始计数）的上方（direct='<'）/下方（direct='>'，默认值）添加`size`行新行，初始以0填充。
   * `addc(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None`, "向第`i`列（从0开始计数）的左侧（direct='<'）/右侧（direct='>'，默认值）添加`size`列新列，初始以0填充。
   * `delr(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None`, "向第`i`行（从0开始计数）的上方（direct='<'）/下方（direct='>'）删除`size`行。
   * `delc(self, i: int, size: int, direct: Literal['>', '<'] = '>') -> None`, `delr`的列版本。
