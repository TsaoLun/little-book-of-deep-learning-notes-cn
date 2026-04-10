# Little Book of Deep Learning 中文笔记项目总结

## 项目概述

本项目旨在将《Little Book of Deep Learning》一书整理为中文笔记，便于中文读者学习和参考。项目采用分章节整理的方式，提供结构清晰、内容完整的中文Markdown文件。

**目标**：
- 将PDF书籍内容提取为文本
- 清理文本格式问题（断字、行中断、页码等）

## 使用的工具和技术

### 1. 文本提取
- **pdftotext**（来自 poppler-utils）：将PDF转换为纯文本
- 命令：`pdftotext -layout lbdl.pdf lbdl_full.txt`
- 输出文件：`lbdl_full.txt`（172,912字节）

### 2. 文本清理脚本
项目开发了多个Python脚本处理文本格式问题：

#### `extract_deepmodels.py`
- 功能：基于行号提取Chapter 4和Chapter 5内容
- 关键参数：
  - Chapter 4起始行：1554（"Chapter 4"）
  - Chapter 5起始行：2841（"Chapter 5"）
  - Part III起始行：3434（"Part III"）
- 输出：`4_deepmodels.md`、`5_deepmodels.md`（原始英文，脚本在scripts目录）

#### `clean_extract.py`
- 功能：修复跨行连字符、断字问题
- 主要方法：
  - `fix_hyphenation_and_line_breaks()`：处理以连字符结尾的行
  - `fix_broken_words()`：合并被拆分的单词（如"van\nishing"）
  - `clean_text()`：移除换页符(ASCII 12)和页码行
- 输出：`4_deepmodels_clean.md`、`5_deepmodels_clean.md`

#### `clean_hyphenation.py`
- 功能：专门修复连字符和断字问题
- 特点：使用启发式方法判断是否合并单词，避免错误合并完整单词

## 提取和清理过程详解

### 1. 章节边界确定
通过grep命令确定章节起始行号：
```bash
grep -n "Chapter 4" lbdl_full.txt
grep -n "Chapter 5" lbdl_full.txt
grep -n "Part III" lbdl_full.txt
```

结果：
- `Chapter 4`：第1554行
- `Chapter 5`：第2841行
- `Part III`：第3434行

### 2. 常见文本问题及处理
1. **跨行连字符**：`backprop-\nagation` → `backpropagation`
2. **断字问题**：`van\nishing gradient` → `vanishing gradient`
3. **页码行**：单独的数字行（如"12"）被移除
4. **换页符**：ASCII 12字符被移除
5. **重复标题**：移除文本中重复的"Chapter X"标题行

### 3. 清理算法要点
- **单词合并判断**：检查单词片段长度、常见后缀（ing, ed, ly等）
- **大小写判断**：下一行以大写开头可能是新句子，不合并
- **合并后长度检查**：合并后单词长度在3-15字符范围内才合并

## 翻译和整理方法

### 1. 翻译原则
- **准确性优先**：技术术语翻译准确，必要时保留英文原文
- **可读性**：中文表达自然流畅，避免生硬直译
- **一致性**：相同术语在整个项目中保持一致翻译

### 2. 技术术语处理
- **层**（layers）
- **卷积层**（convolutional layers）
- **注意力机制**（attention mechanism）
- **残差连接**（residual connections）
- **批归一化**（batch normalization）

### 3. 公式和数学表达式
- 保留LaTeX格式的数学表达式
- 确保下标、上标等特殊符号正确显示
- 复杂公式添加简要中文说明

## 实用命令参考

```bash
# 提取PDF文本
pdftotext -layout lbdl.pdf lbdl_full.txt

# 查找章节边界
grep -n "Chapter 4" lbdl_full.txt
grep -n "Part III" lbdl_full.txt

# 运行提取脚本
python3 extract_deepmodels.py

# 运行清理脚本
python3 clean_extract.py
python3 clean_hyphenation.py

# 检查文件大小
wc -l *.md
```

## 联系与贡献

本项目为开源学习项目，欢迎提出改进建议：
- 术语翻译修正
- 格式改进建议
- 内容补充建议

**项目维护**：通过Claude Code工具持续整理和完善。

---

*最后更新：2026年4月10日*  
*项目状态：已完成（8/8章完成）*