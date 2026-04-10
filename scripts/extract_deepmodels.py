#!/usr/bin/env python3
import re

def clean_text(text):
    """清理文本：移除换页符、页码行，修复公式格式"""
    # 移除换页符 (ASCII 12)
    text = text.replace('\x0c', '')

    # 移除单独的页码行（仅包含数字的行）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过仅包含数字的行（页码）
        if stripped.isdigit():
            continue
        # 跳过仅包含空格和数字的行
        if re.match(r'^\s*\d+\s*$', line):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 修复公式：将单独的下标数字连接到前一个单词
    # 例如：f(x; w)ᵧ 可能被拆分成 f(x; w)ᵧ
    # 这里暂时保留原样，后续可能需要专门处理

    return text

def extract_chapter(text, start_line, end_line, chapter_title):
    """提取指定行范围的章节内容"""
    lines = text.split('\n')
    chapter_lines = lines[start_line-1:end_line]  # 行号从1开始
    chapter_text = '\n'.join(chapter_lines)

    # 清理文本
    chapter_text = clean_text(chapter_text)

    # 添加Markdown标题
    output = f"# {chapter_title}\n\n{chapter_text}"

    return output

def main():
    # 读取完整的文本文件
    with open('../lbdl_full.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    # 定义章节边界（行号从1开始）
    # Chapter 4: Model Components (行号基于grep结果)
    chapter4_start = 1554  # "Chapter 4"所在行
    chapter5_start = 2841  # "Chapter 5"所在行
    part3_start = 3434     # "Part III"所在行

    # 提取Chapter 4: Model Components
    print("提取Chapter 4: Model Components...")
    chapter4_text = extract_chapter(full_text, chapter4_start, chapter5_start-1, "Chapter 4: Model Components")

    # 提取Chapter 5
    print("提取Chapter 5...")
    # 需要确定Chapter 5的标题
    # 从Chapter 5开始的行中查找标题
    lines = full_text.split('\n')
    chapter5_title_line = chapter5_start
    # 跳过空行直到找到非空行
    while chapter5_title_line < len(lines) and not lines[chapter5_title_line-1].strip():
        chapter5_title_line += 1

    if chapter5_title_line < len(lines):
        chapter5_title = lines[chapter5_title_line-1].strip()
        # 如果标题是"Chapter 5"，则取下一行作为实际标题
        if chapter5_title == "Chapter 5":
            next_line_idx = chapter5_title_line
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            if next_line_idx < len(lines):
                chapter5_title = f"Chapter 5: {lines[next_line_idx].strip()}"
            else:
                chapter5_title = "Chapter 5"
    else:
        chapter5_title = "Chapter 5"

    chapter5_text = extract_chapter(full_text, chapter5_start, part3_start-1, chapter5_title)

    # 保存Chapter 4
    with open('../4_deepmodels.md', 'w', encoding='utf-8') as f:
        f.write(chapter4_text)
    print("已保存: 4_deepmodels.md")

    # 保存Chapter 5
    with open('../5_deepmodels.md', 'w', encoding='utf-8') as f:
        f.write(chapter5_text)
    print("已保存: 5_deepmodels.md")

    print("\n提取完成！")

if __name__ == '__main__':
    main()