#!/usr/bin/env python3
import re

def fix_hyphenation_and_line_breaks(text):
    """修复断字和行中断问题"""
    lines = text.split('\n')
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 如果行以连字符结尾，与下一行合并
        if line.rstrip().endswith('-') and i + 1 < len(lines):
            # 移除连字符，连接下一行
            next_line = lines[i + 1].lstrip()
            merged_line = line.rstrip()[:-1] + next_line
            cleaned_lines.append(merged_line)
            i += 2
            continue

        # 如果行以空格结尾或行很短，可能与下一行是同一个单词的一部分
        # 检查行尾是否是一个不完整的单词（没有标点或空格分隔）
        if i + 1 < len(lines):
            current_line_stripped = line.rstrip()
            next_line_stripped = lines[i + 1].lstrip()

            # 如果当前行以单词片段结束，且下一行以单词片段开始，合并
            # 简单启发式：如果当前行以字母结尾，下一行以字母开头
            if (current_line_stripped and current_line_stripped[-1].isalpha() and
                next_line_stripped and next_line_stripped[0].isalpha()):
                # 检查是否可能是同一个单词（例如 "van" 和 "ishing"）
                merged_line = current_line_stripped + next_line_stripped
                cleaned_lines.append(merged_line)
                i += 2
                continue

        cleaned_lines.append(line)
        i += 1

    return '\n'.join(cleaned_lines)

def fix_broken_words(text):
    """修复被拆分的单词，例如 'gra-\ndient' 或 'van\nishing'"""
    # 首先处理跨行连字符
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # 处理被拆分的单词（无连字符）
    # 模式：单词部分在行尾，下一行以单词部分开始
    lines = text.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            # 检查当前行是否以单词片段结束，下一行是否以单词片段开始
            match1 = re.search(r'(\w+)\s*$', lines[i])
            match2 = re.search(r'^\s*(\w+)', lines[i + 1])
            if match1 and match2:
                # 合并单词
                word_part1 = match1.group(1)
                word_part2 = match2.group(1)
                # 检查合并后是否形成合理单词（简单检查）
                merged_word = word_part1 + word_part2
                # 如果合并后的单词在英语中常见（简单启发式）
                if len(word_part1) > 2 and len(word_part2) > 2:
                    # 替换当前行尾的单词部分
                    line1_fixed = re.sub(r'(\w+)\s*$', merged_word, lines[i])
                    # 移除下一行开头的单词部分
                    line2_fixed = re.sub(r'^\s*\w+', '', lines[i + 1]).strip()
                    if line2_fixed:
                        fixed_lines.append(line1_fixed)
                        fixed_lines.append(line2_fixed)
                    else:
                        fixed_lines.append(line1_fixed)
                    i += 2
                    continue

        fixed_lines.append(lines[i])
        i += 1

    return '\n'.join(fixed_lines)

def clean_text(text):
    """清理文本：移除换页符、页码行，修复格式"""
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

    # 修复断字和行中断
    text = fix_broken_words(text)

    # 修复常见的断字模式
    text = fix_hyphenation_and_line_breaks(text)

    # 移除多余的空行（保留段落分隔）
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text

def extract_chapters():
    """提取并清理Chapter 4和Chapter 5"""
    with open('../lbdl_full.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    # 定义章节边界
    chapter4_start = 1554  # "Chapter 4"
    chapter5_start = 2841  # "Chapter 5"
    part3_start = 3434     # "Part III"

    # 分割文本为行
    lines = full_text.split('\n')

    # 提取Chapter 4
    chapter4_lines = lines[chapter4_start-1:chapter5_start-1]
    chapter4_text = '\n'.join(chapter4_lines)

    # 提取Chapter 5
    chapter5_lines = lines[chapter5_start-1:part3_start-1]
    chapter5_text = '\n'.join(chapter5_lines)

    # 清理文本
    chapter4_cleaned = clean_text(chapter4_text)
    chapter5_cleaned = clean_text(chapter5_text)

    # 确定Chapter 5标题
    chapter5_title = "Chapter 5: Architectures"
    # 从文本中提取实际标题
    title_match = re.search(r'Chapter 5\s*\n\s*(.+)', chapter5_text[:200])
    if title_match:
        title = title_match.group(1).strip()
        if title and not title.startswith('Chapter'):
            chapter5_title = f"Chapter 5: {title}"

    # 添加Markdown标题并移除重复的标题行
    # Chapter 4
    chapter4_final = f"# Chapter 4: Model Components\n\n{chapter4_cleaned}"
    # 移除文本中可能重复的"Chapter 4"和"Model Components"行
    chapter4_final = re.sub(r'^Chapter 4\s*$', '', chapter4_final, flags=re.MULTILINE)
    chapter4_final = re.sub(r'^Model Components\s*$', '', chapter4_final, flags=re.MULTILINE)

    # Chapter 5
    chapter5_final = f"# {chapter5_title}\n\n{chapter5_cleaned}"
    chapter5_final = re.sub(r'^Chapter 5\s*$', '', chapter5_final, flags=re.MULTILINE)
    # 移除可能的标题行
    title_line_match = re.search(r'^Chapter 5\s*\n\s*(.+)$', chapter5_text[:200], re.MULTILINE)
    if title_line_match:
        title_text = title_line_match.group(1).strip()
        chapter5_final = re.sub(f'^{re.escape(title_text)}\\s*$', '', chapter5_final, flags=re.MULTILINE)

    # 移除多余空行
    chapter4_final = re.sub(r'\n\s*\n\s*\n+', '\n\n', chapter4_final)
    chapter5_final = re.sub(r'\n\s*\n\s*\n+', '\n\n', chapter5_final)

    # 保存文件
    with open('../4_deepmodels_clean.md', 'w', encoding='utf-8') as f:
        f.write(chapter4_final)

    with open('../5_deepmodels_clean.md', 'w', encoding='utf-8') as f:
        f.write(chapter5_final)

    print("清理后的文件已保存:")
    print("  - 4_deepmodels_clean.md")
    print("  - 5_deepmodels_clean.md")

if __name__ == '__main__':
    extract_chapters()