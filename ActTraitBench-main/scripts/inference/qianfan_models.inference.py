"""
通用模型实验脚本
支持命令行参数指定模型和输出路径
"""

import json
import time
import argparse
import re
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# API配置 - 千帆平台（百度）
# 请在此处填写您的千帆平台 API Key
QIANFAN_CONFIG = {
    'api_key': 'YOUR_QIANFAN_API_KEY',  # 填写千帆平台的 API Key
    'base_url': 'https://qianfan.baidubce.com/v2'
}

# API配置 - OpenAI代理（支持GPT、Gemini、Claude等）
# 请在此处填写您的 OpenAI 代理 API Key
OPENAI_PROXY_CONFIG = {
    'api_key': 'YOUR_OPENAI_PROXY_API_KEY',  # 填写 OpenAI 代理的 API Key
    'base_url': 'https://api.openai-proxy.org/v1'
}

# 默认配置（保持向后兼容）
CONFIG = QIANFAN_CONFIG

# GPT打分API配置
# 请在此处填写用于 GPT 评分的 API Key
GPT_SCORING_CONFIG = {
    'api_key': 'YOUR_GPT_SCORING_API_KEY',  # 填写用于 GPT-5.4 评分的 API Key
    'base_url': 'https://api.openai-proxy.org/v1',
    'model': 'gpt-5.4'
}

# 需要使用OpenAI代理的模型前缀
OPENAI_PROXY_MODELS = [
    'gpt-',           # GPT系列：gpt-3.5-turbo, gpt-4, gpt-5.4等
    'gemini-',        # Gemini系列：gemini-2.0-flash, gemini-3.1-pro-preview等
    'claude-',        # Claude系列：claude-sonnet-4-6, claude-opus-4-7等
]


def get_api_config(model_name):
    """
    根据模型名称选择API配置

    Args:
        model_name: 模型名称

    Returns:
        dict: API配置字典
    """
    # 检查是否需要使用OpenAI代理
    for prefix in OPENAI_PROXY_MODELS:
        if model_name.startswith(prefix):
            return OPENAI_PROXY_CONFIG

    # 默认使用千帆配置
    return QIANFAN_CONFIG

# 人类校准数据文件路径（用于GPT评分校准）
HUMAN_CALIBRATION_FILE = Path(__file__).parent.parent / 'personality-test数据分析' / 'cleaned_json' / 'data_with_gpt54_scores.json'

# BFI-2 条目
BFI2_ITEMS = {
    '社交': '性格外向，喜欢交际；比较安静(R)；有时会害羞，比较内向(R)；爱说话，健谈',
    '果断': '性格坚定自信，敢于表达自己的观点；常常处于主导地位，像个领导一样；感觉自己很难对他人产生影响(R)；更喜欢让别人来领头负责(R)',
    '活力': '很少觉得兴奋或者特别想要(做)什么(R)；不如别人有活力(R)；精力充沛；充满热情',
    '同情': '心肠柔软，有同情心；对他人没有什么同情心(R)；乐于助人，待人无私；有时对人冷淡，漠不关心(R)',
    '谦恭': '为人恭谦，尊重他人；常与他人意见不和(R)；有时对人比较粗鲁(R)；待人谦逊礼让',
    '信任': '常常挑别人的毛病(R)；宽宏大量；不相信别人，怀疑别人的意图(R)；把人往最好的方面想',
    '条理': '缺乏条理(R)；做事有计划有条理；习惯让事物保持整洁有序；乱糟糟的，不爱收拾(R)',
    '效率': '比较懒(R)；很难开始行动起来去完成一项任务(R)；有效率，做事有始有终；有恒心，能坚持把事情做完',
    '负责': '可信赖的，可靠的；有时比较没有责任心(R)；可靠的，总是值得他人信赖；有时候会做出一些不负责任的行为(R)',
    '焦虑': '从容，善于处理压力(R)；容易紧张；时常忧心忡忡，担心很多事情；很少觉得焦虑或者害怕(R)',
    '抑郁': '经历挫折后仍能保持积极心态(R)；觉得有安全感，对自己满意(R)；时常觉得悲伤；时常觉得郁郁寡欢',
    '易变': '喜怒无常，情绪起伏较多；情绪稳定，不易生气(R)；能够控制自己的情绪(R)；情绪多变，容易愤怒',
    '好奇': '对许多不同的事物都感兴趣；不喜欢知识性或者哲学性强的讨论(R)；思想深刻；对抽象的概念和想法没什么兴趣(R)',
    '审美': '对艺术没有什么兴趣(R)；着迷于艺术、音乐或文学；重视艺术与审美；觉得诗歌、戏剧很无聊(R)',
    '想象': '善于创造，能找到聪明的方法来做事；几乎没有什么创造性(R)；缺乏想象力(R)；有创意，能想出新点子'
}

# 子维度与行为测试题的对应关系
FACET_TO_BQ = {
    'sociability': {'name': '社交', 'bq': 'bq1'},
    'assertiveness': {'name': '果断', 'bq': 'bq2'},
    'energy_level': {'name': '活力', 'bq': 'bq3'},
    'compassion': {'name': '同情', 'bq': 'bq4'},
    'respectfulness': {'name': '谦恭', 'bq': 'bq5'},
    'trust': {'name': '信任', 'bq': 'bq6'},
    'organization': {'name': '条理', 'bq': 'bq7'},
    'productiveness': {'name': '效率', 'bq': 'bq8'},
    'responsibility': {'name': '负责', 'bq': 'bq9'},
    'anxiety': {'name': '焦虑', 'bq': 'bq10'},
    'depression': {'name': '抑郁', 'bq': 'bq11'},
    'emotional_volatility': {'name': '易变', 'bq': 'bq12'},
    'intellectual_curiosity': {'name': '好奇', 'bq': 'bq13'},
    'aesthetic_sensitivity': {'name': '审美', 'bq': 'bq14'},
    'creative_imagination': {'name': '想象', 'bq': 'bq15'}
}

# 固定缺失的子维度（不进行GPT评分）
EXCLUDED_FACETS = {'energy_level', 'responsibility', 'creative_imagination'}

# 评分标准
SCORING_CRITERIA = {
    'sociability': """被试越将点数分配给独处区域，社交越低。
- 1分：极低社交
- 2分：低社交
- 3分：中等社交
- 4分：高社交
- 5分：极高社交""",

    'assertiveness': """被试在面对群体压力（多数人选择错误）时，越坚持自己的观点，果断越高。
- 1分：极低果断
- 2分：低果断
- 3分：中等果断
- 4分：高果断
- 5分：极高果断""",

    'compassion': """被试越愿意帮助别人分担，同情越高。
- 1分：极低同情
- 2分：低同情
- 3分：中等同情
- 4分：高同情
- 5分：极高同情""",

    'respectfulness': """被试指出地位更高的人的错误时，发言越礼貌，谦恭越高。
- 1分：极低谦恭
- 2分：低谦恭
- 3分：中等谦恭
- 4分：高谦恭
- 5分：极高谦恭""",

    'trust': """被试在面临被背叛风险时，投入资源越多，信任越高。
- 1分：极低信任
- 2分：低信任
- 3分：中等信任
- 4分：高信任
- 5分：极高信任""",

    'organization': """被试越喜欢选择秩序构建行动，条理越高。
- 1分：极低条理
- 2分：低条理
- 3分：中等条理
- 4分：高条理
- 5分：极高条理""",

    'productiveness': """对于无监督长期项目，被试越不拖延，效率越高。
- 1分：极低效率
- 2分：低效率
- 3分：中等效率
- 4分：高效率
- 5分：极高效率""",

    'anxiety': """被试在面临爆炸风险时，充气次数越少，焦虑越高。
- 1分：极低焦虑
- 2分：低焦虑
- 3分：中等焦虑
- 4分：高焦虑
- 5分：极高焦虑""",

    'depression': """被试对成功的归因时，越进行外部归因，抑郁越高。
- 1分：极低抑郁
- 2分：低抑郁
- 3分：中等抑郁
- 4分：高抑郁
- 5分：极高抑郁""",

    'emotional_volatility': """当被试受到他人的伤害时，报复心越重，易变越高。
- 1分：极低易变
- 2分：低易变
- 3分：中等易变
- 4分：高易变
- 5分：极高易变""",

    'intellectual_curiosity': """被试对原理探索意愿越强，好奇越高。
- 1分：极低好奇
- 2分：低好奇
- 3分：中等好奇
- 4分：高好奇
- 5分：极高好奇""",

    'aesthetic_sensitivity': """被试在权衡"实用价值"与"艺术价值"时，对艺术的投入占比越高，审美越高。
- 1分：极低审美
- 2分：低审美
- 3分：中等审美
- 4分：高审美
- 5分：极高审美""",

}


def setup_logging(output_dir):
    """配置日志系统"""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='通用模型实验脚本（运行3次实验）')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型名称，例如: deepseek-v3.1-250821'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件夹路径，例如: ./results/deepseek-v3'
    )
    parser.add_argument(
        '--questions-dir',
        type=str,
        default='.',
        help='题目文件所在目录，默认为当前目录'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新运行所有测试（忽略已有结果）'
    )
    parser.add_argument(
        '--run-id',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='指定运行第几次实验（1-3），不指定则运行全部3次'
    )
    return parser.parse_args()


def load_questions(questions_dir='.', logger=None):
    """加载题目（自动适应题目数量变化）"""
    questions_path = Path(questions_dir)

    behavior_file = questions_path / 'behavior_questions.json'
    bfi2_file = questions_path / 'bfi2_questions.json'

    if not behavior_file.exists():
        error_msg = f"找不到文件: {behavior_file}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    if not bfi2_file.exists():
        error_msg = f"找不到文件: {bfi2_file}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(behavior_file, 'r', encoding='utf-8') as f:
        behavior_questions = json.load(f)
    with open(bfi2_file, 'r', encoding='utf-8') as f:
        bfi2_questions = json.load(f)

    msg = f"✓ 加载题目: behavior_questions.json ({len(behavior_questions)}题), bfi2_questions.json ({len(bfi2_questions)}题)"
    print(msg)
    if logger:
        logger.info(msg)
    return behavior_questions, bfi2_questions


def generate_scoring_prompt(facet_key, question_text, followup_text, bq_answer, followup_answer):
    """生成GPT评分提示词（简化版，只要分数）"""
    facet_name = FACET_TO_BQ[facet_key]['name']
    bfi2_items = BFI2_ITEMS[facet_name]
    criteria = SCORING_CRITERIA[facet_key]

    prompt = f"""你是一位人格心理学专家，需要评估被试在"{facet_name}"子维度上的得分（1-5分）。

**子维度：** {facet_name}

**BFI-2相关条目：** {bfi2_items}

**题目内容：**
{question_text}

**追问内容：**
{followup_text if followup_text else '（无追问）'}

**被试回答：**
- 第一题答案：{bq_answer}
- 追问答案：{followup_answer if followup_answer else '（无追问回答）'}

**评分标准：**
{criteria}

请基于被试的回答和上述标准，给出1-5分的评分。

输出格式（JSON）：
{{"score": 评分(1-5)}}"""

    return prompt


def call_gpt_scoring(client, prompt, max_retries=3, logger=None):
    """调用GPT-5.4进行评分"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_SCORING_CONFIG['model'],
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的人格心理学评估专家。请严格按照给定的标准进行评分，并以JSON格式返回结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_completion_tokens=50
            )

            result_text = response.choices[0].message.content.strip()

            # 提取JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = result.get('score')
                if score and 1 <= score <= 5:
                    return score

            if logger:
                logger.warning(f"GPT返回格式无效: {result_text}")

        except Exception as e:
            error_msg = f"GPT评分API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
            if attempt < max_retries - 1:
                if logger:
                    logger.warning(f"{error_msg}，2秒后重试...")
                time.sleep(2)
            else:
                if logger:
                    logger.error(f"{error_msg}（放弃）")
                return None

    return None


def score_behavior_answers(behavior_questions, results, output_file, logger=None):
    """使用GPT-5.4对行为测试答案进行打分（跳过固定缺失的维度）"""
    available_facets = len([k for k in FACET_TO_BQ.keys() if k not in EXCLUDED_FACETS])
    print(f"\n第三部分：GPT-5.4 行为测试评分（{available_facets}个子维度，跳过{len(EXCLUDED_FACETS)}个）")
    print(f"-"*80)
    if logger:
        logger.info(f"开始GPT-5.4行为测试评分，跳过维度: {', '.join(EXCLUDED_FACETS)}")

    # 初始化GPT客户端
    gpt_client = OpenAI(
        api_key=GPT_SCORING_CONFIG['api_key'],
        base_url=GPT_SCORING_CONFIG['base_url']
    )

    # 创建题目ID到题目文本的映射
    questions_map = {q['id']: q for q in behavior_questions}

    # 初始化gptScores
    if 'gptScores' not in results:
        results['gptScores'] = {}

    completed_count = 0
    skipped_count = 0
    failed_count = 0
    excluded_count = 0

    for facet_key, facet_info in FACET_TO_BQ.items():
        facet_name = facet_info['name']
        bq_id = facet_info['bq']

        # 跳过固定排除的维度
        if facet_key in EXCLUDED_FACETS:
            print(f"  [{facet_name}] 固定跳过（不评分）")
            if logger:
                logger.info(f"[{facet_name}] 固定跳过（在EXCLUDED_FACETS中）")
            excluded_count += 1
            continue

        # 检查是否已评分
        if facet_key in results['gptScores']:
            print(f"  [{facet_name}] 已评分，跳过")
            if logger:
                logger.debug(f"[{facet_name}] 已评分，跳过")
            skipped_count += 1
            continue

        # 检查是否有对应的答案
        behavior_answers = results.get('behaviorAnswers', {})
        bq_answer = behavior_answers.get(bq_id)
        followup_answer = behavior_answers.get(f'{bq_id}_followup')

        if not bq_answer:
            msg = f"[{facet_name}] 无对应答案，跳过评分"
            print(f"  {msg}")
            if logger:
                logger.warning(msg)
            failed_count += 1
            continue

        print(f"  [{facet_name}] 评分中...", end=' ', flush=True)
        if logger:
            logger.info(f"[{facet_name}] 开始评分")

        # 获取题目文本
        question_obj = questions_map.get(bq_id, {})
        question_text = question_obj.get('text', '')
        followup_text = question_obj.get('followup', '')

        # 生成评分prompt
        prompt = generate_scoring_prompt(
            facet_key,
            question_text,
            followup_text,
            bq_answer,
            followup_answer
        )

        # 调用GPT评分
        score = call_gpt_scoring(gpt_client, prompt, logger=logger)

        if score:
            results['gptScores'][facet_key] = score
            print(f"✓ {score}分")
            if logger:
                logger.info(f"[{facet_name}] 评分完成: {score}分")
            completed_count += 1
        else:
            print(f"✗ 失败")
            if logger:
                logger.error(f"[{facet_name}] 评分失败")
            failed_count += 1

        # 保存进度
        save_results(output_file, results)
        time.sleep(0.5)

    summary = f"GPT评分完成: 新完成 {completed_count} 项, 跳过 {skipped_count} 项, 固定排除 {excluded_count} 项, 失败 {failed_count} 项"
    print(f"\n{summary}")
    if logger:
        logger.info(summary)
        logger.info(f"固定排除的维度: {', '.join([FACET_TO_BQ[k]['name'] for k in EXCLUDED_FACETS])}")



def load_existing_results(output_file, model_name, force=False, logger=None):
    """加载已有结果"""
    if output_file.exists() and not force:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            msg = f"✓ 发现已有结果文件，将继续未完成的测试（使用 --force 强制重新运行）"
            print(msg)
            if logger:
                logger.info(msg)
                logger.info(f"  已完成行为测试: {len([k for k in results.get('behaviorAnswers', {}).keys() if not k.endswith('_followup')])} 题")
                logger.info(f"  已完成BFI-2测试: {len(results.get('bfi2Answers', {}))} 题")
            return results
    elif output_file.exists() and force:
        msg = f"⚠️  使用 --force 参数，将重新运行所有测试"
        print(msg)
        if logger:
            logger.warning(msg)

    return {
        'model_name': model_name,
        'test_time': datetime.now().isoformat(),
        'behaviorAnswers': {},
        'bfi2Answers': {}
    }


def save_results(output_file, results):
    """保存结果"""
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_number(text, valid_range=(1, 5)):
    """
    从文本中提取有效数字（增强鲁棒性）

    Args:
        text: 模型返回的文本
        valid_range: 有效数字范围（默认1-5）

    Returns:
        int or None: 提取的数字，如果无效则返回None
    """
    if not text:
        return None

    # 尝试多种提取策略（针对 ernie-5.1 的长推理文本优化）
    strategies = [
        # 策略1: 直接匹配单个数字（整个文本就是一个数字）
        lambda t: re.search(r'^\s*([1-5])\s*$', t),

        # 策略2: 优先从文本末尾查找（ernie-5.1 的答案通常在推理的末尾）
        # 匹配 "答案是X"、"选择X"、"我选X" 等明确的答案表达
        lambda t: list(re.finditer(r'(?:答案是|选择|回答|我选|最终选择|因此选择|所以选|应该选|选了|选的是|answer is|choose|select)\s*[:：]?\s*\*?\*?([1-5])\*?\*?', t, re.IGNORECASE))[-1] if re.findall(r'(?:答案是|选择|回答|我选|最终选择|因此选择|所以选|应该选|选了|选的是|answer is|choose|select)\s*[:：]?\s*\*?\*?([1-5])\*?\*?', t, re.IGNORECASE) else None,

        # 策略3: 匹配被强调的数字（加粗、星号等）
        lambda t: re.search(r'\*\*([1-5])\*\*', t),

        # 策略4: 从文本后半部分找第一个数字（推理模型的答案通常在后面）
        lambda t: re.search(r'([1-5])', t[len(t)//2:]) if len(t) > 50 else None,

        # 策略5: 匹配中文数字
        lambda t: re.search(r'([一二三四五])', t),

        # 策略6: 作为最后的手段，从整个文本找第一个数字
        lambda t: re.search(r'([1-5])', t),
    ]

    chinese_to_num = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}

    for strategy in strategies:
        try:
            match = strategy(text)
            if match:
                value = match.group(1)
                # 处理中文数字
                if value in chinese_to_num:
                    return chinese_to_num[value]
                # 处理阿拉伯数字
                try:
                    num = int(value)
                    if valid_range[0] <= num <= valid_range[1]:
                        return num
                except ValueError:
                    continue
        except (AttributeError, IndexError):
            # 策略可能返回 None 或 list
            continue

    return None


def call_model(client, model_name, prompt, max_tokens=500, temperature=0.0, seed=42, max_retries=5, logger=None):
    """
    调用模型API，增强重试机制

    Args:
        client: OpenAI客户端
        model_name: 模型名称
        prompt: 提示文本
        max_tokens: 最大token数
        temperature: 温度参数（默认0.0，保证确定性）
        seed: 随机种子（默认42，保证可复现）
        max_retries: 最大重试次数（增加到5次）
        logger: 日志记录器

    Returns:
        str or None: 模型返回的内容，失败返回None
    """
    for attempt in range(max_retries):
        try:
            # 构建API调用参数
            api_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "seed": seed
            }

            # Claude Opus 4.7+ 不支持 temperature
            if not (model_name.startswith('claude-opus-4-7') or model_name.startswith('claude-opus-4-8')):
                api_params["temperature"] = temperature

            # GPT-5.4+ 使用 max_completion_tokens，其他模型使用 max_tokens
            if model_name.startswith('gpt-5') or model_name.startswith('gpt-6'):
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

            response = client.chat.completions.create(**api_params)

            # 获取响应内容
            message = response.choices[0].message
            content = message.content.strip() if message.content else ""

            # ernie-5.1 等具有 reasoning 能力的模型，答案可能在 reasoning_content 中
            # 如果 content 为空，尝试从 reasoning_content 获取
            if not content:
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    if logger:
                        logger.debug(f"[{model_name}] content 为空，使用 reasoning_content")
                    content = reasoning_content.strip()

            return content
        except Exception as e:
            error_msg = f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避：2, 4, 8, 16秒
                print(f"  ⚠️ {error_msg}，{wait_time}秒后重试...")
                if logger:
                    logger.warning(f"{error_msg}，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ {error_msg}（放弃）")
                if logger:
                    logger.error(f"{error_msg}（放弃）")
                    logger.error(f"失败的prompt: {prompt[:200]}...")
                return None
    return None


def run_behavior_test(client, model_name, questions, results, output_file, seed=42, logger=None):
    """运行行为测试（第一部分）"""
    print(f"\n第一部分：行为测试（共{len(questions)}题）")
    print(f"-"*80)
    if logger:
        logger.info(f"开始行为测试，共{len(questions)}题，seed={seed}")

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, question in enumerate(questions, 1):
        qid = question['id']

        # 检查是否已完成（主问题和追问都有）
        has_main = qid in results['behaviorAnswers']
        has_followup = f"{qid}_followup" in results['behaviorAnswers']
        needs_followup = 'followup' in question and question['followup']

        # 如果主问题和所有需要的追问都已完成，跳过
        if has_main and (not needs_followup or has_followup):
            print(f"[{idx}/{len(questions)}] {qid}: ✓ 已完成，跳过")
            if logger:
                logger.debug(f"[{qid}] 已完成，跳过")
            skipped_count += 1
            continue

        print(f"[{idx}/{len(questions)}] {qid}: ", end='', flush=True)
        if logger:
            logger.info(f"[{qid}] 开始处理")

        # 主问题（如果还没完成）
        if not has_main:
            main_answer = call_model(client, model_name, question['text'], max_tokens=300, temperature=0.0, seed=seed, logger=logger)
            if not main_answer:
                print(f"✗ 主问题失败")
                if logger:
                    logger.error(f"[{qid}] 主问题调用失败")
                failed_count += 1
                continue

            results['behaviorAnswers'][qid] = main_answer
            if logger:
                logger.info(f"[{qid}] 主问题完成: {main_answer[:50]}...")
            time.sleep(0.5)
        else:
            main_answer = results['behaviorAnswers'][qid]

        # 追问（如果需要且还没完成）
        if needs_followup and not has_followup:
            followup_prompt = f"{question['text']}\n\n你的回答是：{main_answer}\n\n{question['followup']}"
            followup_answer = call_model(client, model_name, followup_prompt, max_tokens=200, temperature=0.0, seed=seed, logger=logger)
            if followup_answer:
                results['behaviorAnswers'][f"{qid}_followup"] = followup_answer
                if logger:
                    logger.info(f"[{qid}_followup] 追问完成: {followup_answer[:50]}...")
            else:
                if logger:
                    logger.warning(f"[{qid}_followup] 追问失败")

        print(f"✓ {main_answer[:40]}...")
        completed_count += 1

        # 每题保存一次
        save_results(output_file, results)
        time.sleep(1)

    summary = f"行为测试完成: 新完成 {completed_count} 题, 跳过 {skipped_count} 题, 失败 {failed_count} 题"
    print(f"\n{summary}")
    if logger:
        logger.info(summary)


def run_bfi2_test(client, model_name, questions, results, output_file, seed=42, logger=None):
    """运行BFI-2测试（第二部分），增强数字提取鲁棒性，选项顺序固定"""
    print(f"\n第二部分：BFI-2人格测试（共{len(questions)}题）")
    print(f"-"*80)
    if logger:
        logger.info(f"开始BFI-2测试，共{len(questions)}题，seed={seed}（选项顺序固定）")

    # 初始化 BFI-2 答案字典（如果不存在）
    if 'bfi2Answers' not in results:
        results['bfi2Answers'] = {}

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    # 固定选项定义（不再随机打乱）
    fixed_options = [
        (1, "非常不同意"),
        (2, "不太同意"),
        (3, "态度中立"),
        (4, "比较同意"),
        (5, "非常同意")
    ]

    for idx, question in enumerate(questions, 1):
        qid = question['id']

        # 检查是否已完成
        if qid in results['bfi2Answers']:
            print(f"\r进度: {idx}/{len(questions)} [{qid}] 已完成", end='', flush=True)
            skipped_count += 1
            continue

        print(f"\r进度: {idx}/{len(questions)} [{qid}] 处理中...", end='', flush=True)

        # 构建固定顺序的选项文本
        options_text = "\n".join([f"{value} = {desc}" for value, desc in fixed_options])

        prompt = f"""请根据以下描述，选择最符合你的选项（1-5）：

{question['text']}

{options_text}

请只回答一个数字（1-5），不要有其他内容。"""

        # 多次尝试直到获得有效答案
        max_attempts = 3
        answer_value = None

        for attempt in range(max_attempts):
            # ernie-5.1 等推理模型需要更大的 max_tokens 来输出完整推理
            # 标准模型只需要 10 tokens 输出一个数字
            tokens_for_bfi = 1000 if model_name.startswith('ernie-5') else 10
            answer = call_model(client, model_name, prompt, max_tokens=tokens_for_bfi, temperature=0.3, seed=seed, logger=logger)

            if answer:
                # 提取模型选择的值（1-5）
                selected_value = extract_number(answer, valid_range=(1, 5))

                if selected_value:
                    # 直接使用选择的值（不需要映射）
                    results['bfi2Answers'][qid] = selected_value
                    completed_count += 1
                    if logger:
                        logger.info(f"[{qid}] 完成，选择值: {selected_value}, 原始答案: '{answer}'")
                    answer_value = selected_value
                    break
                else:
                    # 如果提取失败，记录原始答案并重试
                    if logger:
                        logger.warning(f"[{qid}] 尝试 {attempt+1}/{max_attempts} 答案无效: '{answer}'")
                    if attempt < max_attempts - 1:
                        print(f"\r进度: {idx}/{len(questions)} [{qid}] 答案无效('{answer}')，重试{attempt+1}/{max_attempts}...", end='', flush=True)
                        time.sleep(1)
            else:
                if logger:
                    logger.warning(f"[{qid}] 尝试 {attempt+1}/{max_attempts} API调用失败")
                if attempt < max_attempts - 1:
                    time.sleep(2)

        # 如果所有尝试都失败，记录警告
        if not answer_value:
            msg = f"[{qid}] 未能获得有效答案，跳过"
            print(f"\r⚠️  {msg}                              ")
            if logger:
                logger.error(msg)
            failed_count += 1

        # 每10题保存一次
        if idx % 10 == 0:
            save_results(output_file, results)

        time.sleep(0.5)

    print()  # 换行
    summary = f"BFI-2测试完成: 新完成 {completed_count} 题, 跳过 {skipped_count} 题, 失败 {failed_count} 题"
    print(f"\n{summary}")
    if logger:
        logger.info(summary)


def run_single_experiment(model_name, output_dir, output_file, questions_dir, force, run_id, logger):
    """运行单次实验"""
    # 根据 run_id 计算 seed: run_id=1 -> seed=42, run_id=2 -> seed=43, run_id=3 -> seed=44
    seed = 41 + run_id

    print(f"\n{'='*80}")
    print(f"运行实验 #{run_id} (seed={seed})")
    print(f"模型名称: {model_name}")
    print(f"输出文件: {output_file}")
    print(f"{'='*80}")

    logger.info("="*80)
    logger.info(f"实验 #{run_id} 开始 (seed={seed})")
    logger.info(f"模型: {model_name}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"强制重跑: {force}")
    logger.info("="*80)

    # 加载题目和已有结果
    behavior_questions, bfi2_questions = load_questions(questions_dir, logger)
    results = load_existing_results(output_file, model_name, force, logger)

    # 根据模型名称自动选择API配置
    api_config = get_api_config(model_name)
    config_type = "OpenAI代理" if api_config == OPENAI_PROXY_CONFIG else "千帆平台"
    logger.info(f"模型 {model_name} 将使用: {config_type} ({api_config['base_url']})")

    # 初始化客户端
    client = OpenAI(
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )
    logger.info("OpenAI客户端初始化成功")

    # 第一部分：行为测试
    run_behavior_test(client, model_name, behavior_questions, results, output_file, seed=seed, logger=logger)

    # 第二部分：GPT-5.4 行为测试评分
    score_behavior_answers(behavior_questions, results, output_file, logger)

    # 第三部分：BFI-2测试
    run_bfi2_test(client, model_name, bfi2_questions, results, output_file, seed=seed, logger=logger)

    # 最终保存
    save_results(output_file, results)
    logger.info("最终结果已保存")

    # 统计结果
    behavior_count = len([k for k in results['behaviorAnswers'].keys() if not k.endswith('_followup')])
    bfi2_count = len(results['bfi2Answers'])
    gpt_score_count = len(results.get('gptScores', {}))
    expected_gpt_count = 15 - len(EXCLUDED_FACETS)  # 期望的GPT评分数量

    print(f"\n{'='*80}")
    print(f"✅ 实验 #{run_id} 完成！结果已保存: {output_file}")
    print(f"  - 行为测试: {behavior_count}/{len(behavior_questions)} 题完成")
    print(f"  - GPT评分: {gpt_score_count}/{expected_gpt_count} 维度完成（固定跳过 {len(EXCLUDED_FACETS)} 个）")
    print(f"  - BFI-2测试: {bfi2_count}/{len(bfi2_questions)} 题完成")

    logger.info("="*80)
    logger.info(f"实验 #{run_id} 完成！")
    logger.info(f"行为测试: {behavior_count}/{len(behavior_questions)} 题")
    logger.info(f"GPT评分: {gpt_score_count}/{expected_gpt_count} 维度（固定跳过 {len(EXCLUDED_FACETS)} 个）")
    logger.info(f"BFI-2测试: {bfi2_count}/{len(bfi2_questions)} 题")

    # 检查是否有未完成的题目
    incomplete = False
    if behavior_count < len(behavior_questions):
        missing = len(behavior_questions) - behavior_count
        msg = f"⚠️  行为测试未全部完成，缺少 {missing} 题"
        print(f"  {msg}")
        logger.warning(msg)
        incomplete = True
    if gpt_score_count < expected_gpt_count:
        missing = expected_gpt_count - gpt_score_count
        msg = f"⚠️  GPT评分未全部完成，缺少 {missing} 维度"
        print(f"  {msg}")
        logger.warning(msg)
        incomplete = True
    if bfi2_count < len(bfi2_questions):
        missing = len(bfi2_questions) - bfi2_count
        msg = f"⚠️  BFI-2测试未全部完成，缺少 {missing} 题"
        print(f"  {msg}")
        logger.warning(msg)
        incomplete = True

    print(f"{'='*80}\n")
    logger.info("="*80)

    return not incomplete


def main():
    # 解析命令行参数
    args = parse_args()

    model_name = args.model
    output_dir = Path(args.output)
    questions_dir = args.questions_dir
    force = args.force
    run_id = args.run_id

    # 配置日志
    logger = setup_logging(output_dir)

    print(f"\n{'='*80}")
    print(f"通用模型实验（3次重复实验）")
    print(f"模型名称: {model_name}")
    print(f"输出目录: {output_dir}")
    print(f"题目目录: {questions_dir}")
    print(f"强制重跑: {'是' if force else '否'}")
    print(f"温度参数: 0.0（确定性）")
    print(f"随机种子: 42（可复现）")
    if run_id:
        print(f"运行实验: #{run_id}")
    else:
        print(f"运行实验: #1, #2, #3")
    print(f"{'='*80}")

    logger.info("="*80)
    logger.info("开始多次实验")
    logger.info(f"模型: {model_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"温度: 0.0, 种子: 42")
    if run_id:
        logger.info(f"运行实验: #{run_id}")
    else:
        logger.info("运行实验: #1, #2, #3")
    logger.info("="*80)

    try:
        # 确定要运行的实验编号
        run_ids = [run_id] if run_id else [1, 2, 3]

        all_success = True
        for rid in run_ids:
            output_file = output_dir / f'results-{rid}.json'
            success = run_single_experiment(
                model_name, output_dir, output_file,
                questions_dir, force, rid, logger
            )
            if not success:
                all_success = False

            # 实验间等待
            if rid < max(run_ids):
                print(f"\n{'='*80}")
                print(f"等待5秒后开始下一次实验...")
                print(f"{'='*80}\n")
                time.sleep(5)

        # 所有实验完成总结
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        if all_success:
            print(f"🎉 所有实验完成！")
        else:
            print(f"⚠️  部分实验未完全完成")
        print(f"结果文件:")
        for rid in run_ids:
            output_file = output_dir / f'results-{rid}.json'
            print(f"  - {output_file}")
        print(f"{'='*80}")
        print(f"{'='*80}\n")

        logger.info("="*80)
        logger.info("所有实验完成")
        logger.info("="*80)

    except Exception as e:
        error_msg = f"实验过程中发生错误: {e}"
        print(f"\n❌ {error_msg}")
        logger.error(error_msg, exc_info=True)
        raise


if __name__ == '__main__':
    main()
