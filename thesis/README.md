# 学位论文（江西师范大学硕士学位论文 LaTeX 模板，2023 版）

```
main.tex            主文件：封面信息（姓名、学号、导师等请自行填写）、宏包与命令
hnuthesis.cls       模板类文件（对字体加载做了两处兼容性修改，见文件内“本地修改”注释）
hnunumerical.bst    模板的参考文献样式
chapters/           摘要、各章、结语、附录、致谢、发表情况
figures/            手工绘制的图（TikZ）与校徽
gen/                由脚本生成的表格、图与数字宏，不要手工修改
refs.bib            参考文献
```

## 编译前的准备

模板自带的两个中文字体文件与校徽图片不随本仓库分发，请从学校下发的模板压缩包中复制到本目录：

```
simsun.ttc  simhei.ttf        ->  thesis/
figures/Jxnu-logo.png         ->  thesis/figures/
```

## 编译

```bash
python scripts/make_assets.py --out thesis/gen     # 第 3 章的表、图与数字（来自 results/analysis.json）
python scripts/ob_assets.py                        # 第 4 章的表、图与数字（来自 results/ob/）
make -C thesis                                     # xelatex -> bibtex -> xelatex x2
```

西文字体：模板指定 Times New Roman / Arial / Courier New。系统装有这些字体时（Windows）直接使用；
没有时（Linux）类文件自动退回到 TeX 发行版自带的 TeX Gyre Termes / Heros / Cursor，版式基本一致，
但提交学校前请在装有指定字体的机器上重新编译一次。

正文中的实验数字都写成 `\ptwo{路径}` 这样的宏，由 `gen/numbers.tex` 与 `gen/ob_numbers.tex` 提供；
路径不存在时 PDF 中会出现醒目的 `??路径??`，便于发现。
