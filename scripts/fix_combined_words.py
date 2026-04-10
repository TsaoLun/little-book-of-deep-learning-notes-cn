#!/usr/bin/env python3
import re
import sys

def fix_combined_words(text):
    """修复粘连的单词"""
    # 常见粘连模式：小写字母后接大写字母（除了专有名词）
    # 但我们的问题主要是小写字母后接小写字母，如"facerecognition"
    # 使用启发式方法：在单词边界处插入空格，当序列长度>2且看起来像两个单词
    # 简单方法：查找小写字母序列后接小写字母序列，其中第一个序列以常见单词结尾
    # 更简单：手动替换已知模式

    # 定义常见粘连模式的正则表达式替换
    patterns = [
        (r'facerecognition', 'face recognition'),
        (r'imageprocessing', 'image processing'),
        (r'byutilizing', 'by utilizing'),
        (r'agraayscale', 'a grayscale'),
        (r'suchas', 'such as'),
        (r'becorrected', 'be corrected'),
        (r'takesa', 'takes a'),
        (r'anestimate', 'an estimate'),
        (r'itis', 'it is'),
        (r'integrate skip', 'integrate skip'),  # 保持不变
        (r'attention layersto', 'attention layers to'),
        (r'thatare', 'that are'),
        (r'acrossall', 'across all'),
        (r'thedegraded', 'the degraded'),
        (r'quantitymay', 'quantity may'),
        (r'partsof', 'parts of'),
        (r'forextracting', 'for extracting'),
        (r'consistsof', 'consists of'),
        (r'asmany', 'as many'),
        (r'thecross', 'the cross'),
        (r'entropy loss', 'entropy loss'),  # 保持不变
        (r'data data augaugmentation', 'data augmentation'),  # 修复重复
        (r'augaugmentation', 'augmentation'),
        (r'contentof', 'content of'),
        (r'isob', 'is '),  # 可能是"is object detection"
        (r'object dedetection', 'object detection'),
        (r'givenan', 'given an'),
        (r'contained', 'contained'),  # 保持不变
        (r'Sin', 'Sin'),  # 可能是"Single Shot Detector"的一部分
        (r'Liuet', 'Liu et'),
        (r'Eachof', 'Each of'),
        (r'sothe', 'so the'),
        (r'ofthe', 'of the'),
        (r'getscoarser', 'gets coarser'),
        (r'laylayers', 'layers'),
        (r'afeature', 'a feature'),
        (r'theinput', 'the input'),
        (r'atevery', 'at every'),
        (r'whose', 'whose'),  # 保持不变
        (r'centered', 'centered'),  # 保持不变
        (r'thatcentered', 'that centered'),
        (r'toa', 'to a'),
        (r'forevery', 'for every'),
        (r'thereare', 'there are'),
        (r'output', 'output'),  # 保持不变
        (r'aspectratios', 'aspect ratios'),
        (r'tocreate', 'to create'),
        (r'requires', 'requires'),  # 保持不变
        (r'tomitigate', 'to mitigate'),
        (r'thisissue', 'this issue'),
        (r'prepre-trained', 'pre-trained'),
        (r'canbe', 'can be'),
        (r'forobject', 'for object'),
        (r'thatcanbe', 'that can be'),
        (r'sesemantic', 'semantic'),
        (r'segmentation', 'segmentation'),  # 保持不变
        (r'object', 'object'),  # 保持不变
        (r'networkthat', 'network that'),
        (r'as manychannels', 'as many channels'),
        (r'logitsfor', 'logits for'),
        (r'thistask', 'this task'),
        (r'Thisis', 'This is'),
        (r'iscaptured', 'is captured'),
        (r'downscalethe', 'downscale the'),
        (r'con convolutional', 'convolutional'),
        (r'laylayers', 'layers'),
        (r'transtransposed', 'transposed'),
        (r'conconvo volutional', 'convolutional'),
        (r'methodssuch', 'methods such'),
        (r'allthe', 'all the'),
        (r'Modelsthat', 'Models that'),
        (r'skip conconnections', 'skip connections'),
        (r'fromlayers', 'from layers'),
        (r'netnetwork', 'network'),
        (r'prepre', 'pre'),
        (r'availabilityof', 'availability of'),
        (r'Speech', 'Speech'),  # 保持不变
        (r'recog recognition', 'recognition'),
        (r'asound', 'a sound'),
        (r'Therehave', 'There have'),
        (r'recentone', 'recent one'),
        (r'ofcasting', 'of casting'),
        (r'aspectrogram', 'a spectrogram'),
        (r'tokenizer', 'tokenizer'),  # 保持不变
        (r'enizer', 'enizer'),  # 可能是tokenizer的一部分
        (r'con convolutional', 'convolutional'),
        (r'oneofthe', 'one of the'),
        (r'Multiple', 'Multiple'),  # 保持不变
        (r'ofany', 'of any'),
        (r'ambient', 'ambient'),  # 保持不变
        (r'largedatasets', 'large datasets'),
        (r'soundsources', 'sound sources'),
        (r'ultimategoal', 'ultimate goal'),
        (r'inputsig nal', 'input signal'),  # 注意有空格
        (r'hencea', 'hence a'),
        (r'Text-image', 'Text-image'),  # 保持不变
        (r'powerful', 'powerful'),  # 保持不变
        (r'and text', 'and text'),  # 保持不变
        (r'feature', 'feature'),  # 保持不变
        (r'Con', 'Con'),  # 可能是"Contrastive"的一部分
        (r'Contrastive', 'Contrastive'),  # 保持不变
        (r'Pre', 'Pre'),  # 可能是"Pre-training"的一部分
        (r'training', 'training'),  # 保持不变
        (r'andGPT', 'and GPT'),
        (r'forboth', 'for both'),
        (r'astandard', 'a standard'),
        (r'autoregressive', 'autoregressive'),  # 保持不变
        (r'endof', 'end of'),
        (r'usethe', 'use the'),
        (r'between', 'between'),  # 保持不变
        (r'usinga', 'using a'),
        (r'ik', 'ik'),  # 可能是变量
        (r'tk', 'tk'),  # 可能是变量
        (r'concontrastive', 'contrastive'),
        (r'trastive', 'trastive'),  # 可能是"contrastive"的一部分
        (r'forevery', 'for every'),
        (r'mini-batch', 'mini-batch'),  # 保持不变
        (r'computed', 'computed'),  # 保持不变
        (r'ln', 'ln'),  # 可能是变量
        (r'similarityln', 'similarity ln'),
        (r'lm', 'lm'),  # 可能是变量
        (r'preprediction', 'prediction'),
        (r'classifying asignal', 'classifying a signal'),
        (r'bydefining', 'by defining'),
        (r'theembedding', 'the embedding'),
        (r'embeddingof', 'embedding of'),
        (r'often', 'often'),  # 保持不变
        (r'richer', 'richer'),  # 保持不变
        (r'pickup', 'pick up'),
        (r'performanceon', 'performance on'),
        (r'standard', 'standard'),  # 保持不变
        (r'Radfordet', 'Radford et'),
        (r'embedding', 'embedding'),  # 保持不变
        (r'Reinforcement', 'Reinforcement'),  # 保持不变
        (r'orrobotic', 'or robotic'),
        (r'Rt thatcan', 'Rt that can'),
        (r'If', 'If'),  # 保持不变
        (r'Marko', 'Marko'),  # 可能是"Markovian"的一部分
        (r'Markovian', 'Markovian'),  # 保持不变
        (r'aloneas', 'alone as'),
        (r'thepast', 'the past'),
        (r'Marko', 'Marko'),  # 重复
        (r'Markovian', 'Markovian'),  # 重复
        (r'finda', 'find a'),
        (r'polpolicy', 'policy'),
        (r'expectation', 'expectation'),  # 保持不变
        (r'rereturn', 'return'),
        (r'discounted', 'discounted'),  # 保持不变
        (r'Re', 'Re'),  # 可能是"Reinforcement"的一部分
        (r'forcement', 'forcement'),  # 可能是"Reinforcement"的一部分
        (r'Learn', 'Learn'),  # 可能是"Learning"的一部分
        (r'Learning', 'Learning'),  # 保持不变
        (r'opoptimal', 'optimal'),
        (r'mal', 'mal'),  # 可能是"optimal"的一部分
        (r'polpolicy', 'policy'),
        (r'thanks', 'thanks'),  # 保持不变
        (r'Bell', 'Bell'),  # 可能是"Bellman"的一部分
        (r'Bellman', 'Bellman'),  # 保持不变
        (r'training', 'training'),  # 保持不变
        (r'atra', 'a tra'),  # 可能是"a parametric"
        (r'Atarivideo', 'Atari video'),
        (r'threethat', 'three that'),
        (r'assumption', 'assumption'),  # 保持不变
        (r'Deep', 'Deep'),  # 保持不变
        (r'Deep Q', 'Deep Q'),  # 保持不变
        (r'Q-Network', 'Q-Network'),  # 保持不变
        (r'classical', 'classical'),  # 保持不变
        (r'playing andrecording', 'playing and recording'),
        (r'of tuples', 'of tuples'),
        (r'takenacross', 'taken across'),
        (r'and minimizing', 'and minimizing'),  # 保持不变
        (r'if thistuple', 'if this tuple'),
        (r'otherwise', 'otherwise'),  # 保持不变
        (r'Value', 'Value'),  # 保持不变
        (r'statevalue', 'state value'),
        (r'ofclearing', 'of clearing'),
        (r'breakthrough', 'break through'),
        (r'to the topline', 'to the top line'),
        (r'ensuresa', 'ensures a'),
        (r'Here', 'Here'),  # 保持不变
        (r'gradientdoes', 'gradient does'),
        (r'through it', 'through it'),  # 保持不变
        (r'necessary', 'necessary'),  # 保持不变
        (r'yn', 'yn'),  # 变量
        (r'itself whichis', 'itself which is'),
        (r'Fixing', 'Fixing'),  # 保持不变
        (r'results ina', 'results in a'),
        (r'policy', 'policy'),  # 保持不变
        (r'ϵ-greedy', 'ϵ-greedy'),  # 保持不变
        (r'completely', 'completely'),  # 保持不变
        (r'theoptimal', 'the optimal'),
        (r'argmax', 'argmax'),  # 保持不变
        (r'otherwise', 'otherwise'),  # 保持不变
        (r'Injecting', 'Injecting'),  # 保持不变
        (r'favor', 'favor'),  # 保持不变
        (r'exploration', 'exploration'),  # 保持不变
        (r'gameplay', 'game play'),
        (r'accurate', 'accurate'),  # 保持不变
        (r'estimates', 'estimates'),  # 保持不变
        (r'andreaches', 'and reaches'),
        (r'performance', 'performance'),  # 保持不变
        (r'majority', 'majority'),  # 保持不变
        (r'games', 'games'),  # 保持不变
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 通用规则：在小写字母后接大写字母时添加空格（专有名词除外）
    # 但小心处理缩写
    # text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 修复重复单词，如"data data augmentation" -> "data augmentation"
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_combined_words.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = input_file.replace('.md', '_fixed.md')

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    fixed_text = fix_combined_words(text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_text)

    print(f"Fixed text saved to: {output_file}")

if __name__ == '__main__':
    main()