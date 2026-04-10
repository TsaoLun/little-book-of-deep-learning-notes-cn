#!/usr/bin/env python3
import re
import sys

def fix_hyphenation(text):
    """修复连字符断字"""
    # 修复跨行连字符：word-\nword -> wordword
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text

def fix_broken_words(text):
    """修复被拆分的单词"""
    lines = text.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 如果当前行以单词片段结束，检查下一行是否以单词片段开始
        if i + 1 < len(lines):
            # 获取当前行末尾的单词片段
            match_end = re.search(r'(\w+)\s*$', line)
            # 获取下一行开头的单词片段
            match_start = re.search(r'^\s*(\w+)', lines[i + 1])

            if match_end and match_start:
                end_part = match_end.group(1)
                start_part = match_start.group(1)

                # 检查合并后是否形成合理的英语单词
                # 简单启发式：如果end_part以ing、ed、ly等结尾，可能是完整的单词
                common_suffixes = ['ing', 'ed', 'ly', 'tion', 'ment', 'ness', 'able', 'ible', 'ful', 'less', 'est', 'er', 'al', 'ize', 'ise']
                has_common_suffix = any(end_part.endswith(suffix) for suffix in common_suffixes)

                # 如果end_part看起来像完整单词，不要合并
                if has_common_suffix and len(end_part) > 3:
                    fixed_lines.append(line)
                    i += 1
                    continue

                # 检查start_part是否可能是新单词的开始
                # 如果start_part以大写字母开头，可能是新句子
                if start_part and start_part[0].isupper():
                    fixed_lines.append(line)
                    i += 1
                    continue

                # 检查合并后的单词长度是否合理
                merged = end_part + start_part
                if 3 <= len(merged) <= 15:  # 合理的英语单词长度
                    # 修复当前行：替换末尾的单词部分
                    line_fixed = re.sub(r'(\w+)\s*$', merged, line)
                    # 修复下一行：移除开头的单词部分
                    next_line_fixed = re.sub(r'^\s*\w+', '', lines[i + 1]).lstrip()

                    if next_line_fixed:
                        fixed_lines.append(line_fixed)
                        lines[i + 1] = next_line_fixed
                        i += 1  # 继续处理当前行（已修改）
                        continue
                    else:
                        fixed_lines.append(line_fixed)
                        i += 2  # 跳过下一行（已完全移除）
                        continue

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)

def fix_line_breaks(text):
    """修复不必要的换行，合并短行"""
    lines = text.split('\n')
    merged_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # 如果行非常短（小于40字符），且不是标题或列表项，尝试与下一行合并
        if (len(line) < 40 and i + 1 < len(lines) and
            not re.match(r'^\d+\.\d+\s', line) and  # 不是小节标题
            not re.match(r'^[•\-*]\s', line) and     # 不是列表项
            not re.match(r'^Figure \d+\.\d+:', line) and  # 不是图标题
            not re.match(r'^Table \d+\.\d+:', line) and   # 不是表标题
            line and lines[i + 1].strip()):  # 当前行和下一行都不为空

            next_line = lines[i + 1].lstrip()
            # 检查下一行是否以小写字母开头（可能是同一句子）
            if next_line and next_line[0].islower():
                merged = line + ' ' + next_line
                merged_lines.append(merged)
                i += 2
                continue

        merged_lines.append(line)
        i += 1

    return '\n'.join(merged_lines)

def clean_text(text):
    """清理文本"""
    # 移除换页符
    text = text.replace('\x0c', '')

    # 移除单独的页码行
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.isdigit():
            continue
        if re.match(r'^\s*\d+\s*$', line):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 修复断字
    text = fix_hyphenation(text)

    # 修复被拆分的单词
    text = fix_broken_words(text)

    # 修复不必要的换行
    text = fix_line_breaks(text)

    # 移除多余空行
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text

def extract_and_clean():
    """提取并清理Chapter 6, 7, 8"""
    # 读取原始文本文件
    with open('../lbdl_full.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    # 章节边界（基于之前的grep结果）
    part3_start = 3434      # "Part III"
    chapter6_start = 3442   # "Chapter 6"
    chapter7_start = 3961   # "Chapter 7"
    chapter8_start = 4177   # "Chapter 8"
    bibliography_start = 4696  # "Bibliography"

    # 总行数
    lines = full_text.split('\n')
    total_lines = len(lines)

    print(f"总行数: {total_lines}")
    print(f"Part III开始行: {part3_start}")
    print(f"Chapter 6开始行: {chapter6_start}")
    print(f"Chapter 7开始行: {chapter7_start}")
    print(f"Chapter 8开始行: {chapter8_start}")
    print(f"Bibliography开始行: {bibliography_start}")

    # 提取Chapter 6: Prediction
    chapter6_lines = lines[chapter6_start-1:chapter7_start-1]
    chapter6_text = '\n'.join(chapter6_lines)

    # 提取Chapter 7: Synthesis
    chapter7_lines = lines[chapter7_start-1:chapter8_start-1]
    chapter7_text = '\n'.join(chapter7_lines)

    # 提取Chapter 8: The Compute Schism
    chapter8_lines = lines[chapter8_start-1:bibliography_start-1]
    chapter8_text = '\n'.join(chapter8_lines)

    # 清理
    print("清理Chapter 6...")
    chapter6_cleaned = clean_text(chapter6_text)
    print("清理Chapter 7...")
    chapter7_cleaned = clean_text(chapter7_text)
    print("清理Chapter 8...")
    chapter8_cleaned = clean_text(chapter8_text)

    # 移除重复的标题行
    chapter6_cleaned = re.sub(r'^Chapter 6\s*$', '', chapter6_cleaned, flags=re.MULTILINE)
    chapter6_cleaned = re.sub(r'^Prediction\s*$', '', chapter6_cleaned, flags=re.MULTILINE)

    chapter7_cleaned = re.sub(r'^Chapter 7\s*$', '', chapter7_cleaned, flags=re.MULTILINE)
    chapter7_cleaned = re.sub(r'^Synthesis\s*$', '', chapter7_cleaned, flags=re.MULTILINE)

    chapter8_cleaned = re.sub(r'^Chapter 8\s*$', '', chapter8_cleaned, flags=re.MULTILINE)
    chapter8_cleaned = re.sub(r'^The Compute Schism\s*$', '', chapter8_cleaned, flags=re.MULTILINE)

    # 添加Markdown标题
    chapter6_final = f"# Chapter 6: Prediction\n\n{chapter6_cleaned}"
    chapter7_final = f"# Chapter 7: Synthesis\n\n{chapter7_cleaned}"
    chapter8_final = f"# Chapter 8: The Compute Schism\n\n{chapter8_cleaned}"

    # 保存英文清理版（供翻译参考）
    with open('../6_applications_en.md', 'w', encoding='utf-8') as f:
        f.write(chapter6_final)

    with open('../7_applications_en.md', 'w', encoding='utf-8') as f:
        f.write(chapter7_final)

    with open('../8_applications_en.md', 'w', encoding='utf-8') as f:
        f.write(chapter8_final)

    print("已保存清理后的英文文件:")
    print("  - 6_applications_en.md")
    print("  - 7_applications_en.md")
    print("  - 8_applications_en.md")

    # 也保存一个合并的文本文件用于翻译
    combined = f"# Chapter 6: Prediction\n\n{chapter6_cleaned}\n\n\n# Chapter 7: Synthesis\n\n{chapter7_cleaned}\n\n\n# Chapter 8: The Compute Schism\n\n{chapter8_cleaned}"
    with open('../applications_combined_en.md', 'w', encoding='utf-8') as f:
        f.write(combined)
    print("  - applications_combined_en.md (合并版本)")

if __name__ == '__main__':
    extract_and_clean()