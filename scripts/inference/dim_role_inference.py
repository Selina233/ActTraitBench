"""
维度角色注入实验脚本
对每道行为测试题/BFI-2题目，按其对应维度注入高/低特质角色
每个条件重复3次实验（temperature=0, seed=42）
"""

import json
import time
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# API配置 - 千帆平台（百度）
QIANFAN_CONFIG = {
    'api_key': 'YOUR_QIANFAN_API_KEY',  # 填写千帆平台的 API Key,
    'base_url': 'https://qianfan.baidubce.com/v2'
}

# API配置 - OpenAI代理（支持GPT、Gemini、Claude等）
OPENAI_PROXY_CONFIG = {
    'api_key': 'YOUR_OPENAI_PROXY_API_KEY',  # 填写 OpenAI 代理的 API Key,
    'base_url': 'https://api.openai-proxy.org/v1'
}

# 默认配置（保持向后兼容）
CONFIG = QIANFAN_CONFIG

# GPT打分API配置
GPT_SCORING_CONFIG = {
    'api_key': 'YOUR_OPENAI_PROXY_API_KEY',  # 填写 OpenAI 代理的 API Key,
    'base_url': 'https://api.openai-proxy.org/v1',
    'model': 'gpt-5.4'
}

# 需要使用OpenAI代理的模型前缀
OPENAI_PROXY_MODELS = [
    'gpt-',           # GPT系列：gpt-3.5-turbo, gpt-4, gpt-4o等
    'gemini-',        # Gemini系列：gemini-2.0-flash, gemini-3.1-pro-preview等
    'claude-',        # Claude系列：claude-sonnet-4-6等
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

# 子维度到大五维度的映射
FACET_TO_BIG5 = {
    # Extraversion
    'sociability': 'Extraversion',
    'assertiveness': 'Extraversion',
    'energy_level': 'Extraversion',
    # Agreeableness
    'compassion': 'Agreeableness',
    'respectfulness': 'Agreeableness',
    'trust': 'Agreeableness',
    # Conscientiousness
    'organization': 'Conscientiousness',
    'productiveness': 'Conscientiousness',
    'responsibility': 'Conscientiousness',
    # Neuroticism (标准Big Five名称)
    'anxiety': 'Neuroticism',
    'depression': 'Neuroticism',
    'emotional_volatility': 'Neuroticism',
    # Openness (标准Big Five名称)
    'intellectual_curiosity': 'Openness',
    'aesthetic_sensitivity': 'Openness',
    'creative_imagination': 'Openness'
}

# 行为测试题到子维度的映射
BQ_TO_FACET = {
    'bq1': 'sociability',
    'bq2': 'assertiveness',
    'bq3': 'energy_level',
    'bq4': 'compassion',
    'bq5': 'respectfulness',
    'bq6': 'trust',
    'bq7': 'organization',
    'bq8': 'productiveness',
    'bq9': 'responsibility',
    'bq10': 'anxiety',
    'bq11': 'depression',
    'bq12': 'emotional_volatility',
    'bq13': 'intellectual_curiosity',
    'bq14': 'aesthetic_sensitivity',
    'bq15': 'creative_imagination'
}

# BFI-2题目到子维度的映射
BFI2_TO_FACET = {
    'bfi1': 'sociability', 'bfi16': 'sociability', 'bfi31': 'sociability', 'bfi46': 'sociability',
    'bfi6': 'assertiveness', 'bfi21': 'assertiveness', 'bfi36': 'assertiveness', 'bfi51': 'assertiveness',
    'bfi11': 'energy_level', 'bfi26': 'energy_level', 'bfi41': 'energy_level', 'bfi56': 'energy_level',
    'bfi2': 'compassion', 'bfi17': 'compassion', 'bfi32': 'compassion', 'bfi47': 'compassion',
    'bfi7': 'respectfulness', 'bfi22': 'respectfulness', 'bfi37': 'respectfulness', 'bfi52': 'respectfulness',
    'bfi12': 'trust', 'bfi27': 'trust', 'bfi42': 'trust', 'bfi57': 'trust',
    'bfi3': 'organization', 'bfi18': 'organization', 'bfi33': 'organization', 'bfi48': 'organization',
    'bfi8': 'productiveness', 'bfi23': 'productiveness', 'bfi38': 'productiveness', 'bfi53': 'productiveness',
    'bfi13': 'responsibility', 'bfi28': 'responsibility', 'bfi43': 'responsibility', 'bfi58': 'responsibility',
    'bfi4': 'anxiety', 'bfi19': 'anxiety', 'bfi34': 'anxiety', 'bfi49': 'anxiety',
    'bfi9': 'depression', 'bfi24': 'depression', 'bfi39': 'depression', 'bfi54': 'depression',
    'bfi14': 'emotional_volatility', 'bfi29': 'emotional_volatility', 'bfi44': 'emotional_volatility', 'bfi59': 'emotional_volatility',
    'bfi10': 'intellectual_curiosity', 'bfi25': 'intellectual_curiosity', 'bfi40': 'intellectual_curiosity', 'bfi55': 'intellectual_curiosity',
    'bfi5': 'aesthetic_sensitivity', 'bfi20': 'aesthetic_sensitivity', 'bfi35': 'aesthetic_sensitivity', 'bfi50': 'aesthetic_sensitivity',
    'bfi15': 'creative_imagination', 'bfi30': 'creative_imagination', 'bfi45': 'creative_imagination', 'bfi60': 'creative_imagination'
}


def setup_logging(output_dir):
    """配置日志系统"""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'dim_role_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger



def load_role_prompts(prompt_file):
    """加载维度角色prompt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['BFI_Standard_Prompts']


def load_questions(questions_dir):
    """加载题目"""
    behavior_file = Path(questions_dir) / 'behavior_questions.json'
    bfi2_file = Path(questions_dir) / 'bfi2_questions.json'

    with open(behavior_file, 'r', encoding='utf-8') as f:
        behavior_questions = json.load(f)
    with open(bfi2_file, 'r', encoding='utf-8') as f:
        bfi2_questions = json.load(f)

    return behavior_questions, bfi2_questions


def call_model_with_role(client, model_name, role_prompt, question_text, max_tokens=500, temperature=0.0, seed=42, logger=None):
    """
    调用模型API，注入角色prompt

    Args:
        client: OpenAI客户端
        model_name: 模型名称
        role_prompt: 角色设定prompt
        question_text: 题目内容
        max_tokens: 最大token数
        temperature: 温度参数（默认0.0）
        seed: 随机种子（默认42）
        logger: 日志记录器
    """
    for attempt in range(5):
        try:
            # 构建API调用参数
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": question_text}
                ],
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

            # 调试：检查响应结构
            if not response.choices or len(response.choices) == 0:
                raise ValueError(f"API返回无choices: {response}")

            choice = response.choices[0]
            message = choice.message
            content = message.content.strip() if message.content else ""

            # ernie-5.1 等具有 reasoning 能力的模型，答案可能在 reasoning_content 中
            # 如果 content 为空，尝试从 reasoning_content 获取
            if not content:
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    if logger:
                        logger.debug(f"[{model_name}] content 为空，使用 reasoning_content")
                    content = reasoning_content.strip()
                else:
                    # 如果 reasoning_content 也没有，则报错
                    finish_reason = choice.finish_reason
                    if logger:
                        logger.warning(f"API返回content和reasoning_content都为空, finish_reason={finish_reason}")
                    raise ValueError(f"API返回内容为空 (finish_reason={finish_reason})")

            return content
        except Exception as e:
            error_msg = f"API调用失败 (尝试 {attempt + 1}/5): {e}"
            if attempt < 4:
                wait_time = 2 ** attempt
                if logger:
                    logger.warning(f"{error_msg}，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                if logger:
                    logger.error(f"{error_msg}（放弃）")
                return None
    return None


def generate_scoring_prompt(facet_key, question_text, followup_text, bq_answer, followup_answer):
    """生成GPT评分提示词"""
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

            # 调试：检查响应结构
            if not response.choices or len(response.choices) == 0:
                raise ValueError(f"GPT评分API返回无choices: {response}")

            choice = response.choices[0]
            content = choice.message.content

            if content is None:
                finish_reason = choice.finish_reason
                if logger:
                    logger.warning(f"GPT评分API返回content为None, finish_reason={finish_reason}")
                raise ValueError(f"GPT评分API返回内容为空 (finish_reason={finish_reason})")

            result_text = content.strip()

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


def score_behavior_answers_for_role(behavior_questions, results, output_dir, model_name, logger=None):
    """对行为测试答案进行GPT评分（维度角色实验版本）"""
    logger.info(f"\n{'='*80}")
    logger.info(f"第三部分：GPT-5.4 行为测试评分")
    logger.info(f"{'='*80}\n")

    # 初始化GPT客户端
    gpt_client = OpenAI(
        api_key=GPT_SCORING_CONFIG['api_key'],
        base_url=GPT_SCORING_CONFIG['base_url']
    )

    # 创建题目ID到题目文本的映射
    questions_map = {q['id']: q for q in behavior_questions}

    scored_count = 0
    skipped_count = 0

    for qid, data in results['behavior_results'].items():
        facet = data['facet']

        # 跳过排除的子维度
        if facet in EXCLUDED_FACETS:
            print(f"  [{qid}] 跳过评分（排除子维度：{facet}）")
            logger.info(f"[{qid}] 跳过评分（排除子维度：{facet}）")
            skipped_count += 1
            continue

        question_text = data['question']
        followup_text = data.get('followup_question', '')

        logger.info(f"\n[{qid}] 开始GPT评分（子维度：{facet}）")

        # 评分高特质角色的3次回答
        if 'high_trait_gpt_scores' not in data:
            data['high_trait_gpt_scores'] = []

        for run_id, run_data in enumerate(data['high_trait_runs'], 1):
            main_answer = run_data.get('main_answer')
            followup_answer = run_data.get('followup_answer')

            if not main_answer:
                continue

            print(f"    高特质 #{run_id}/3 评分中...", end=' ', flush=True)

            prompt = generate_scoring_prompt(facet, question_text, followup_text, main_answer, followup_answer)
            score = call_gpt_scoring(gpt_client, prompt, logger=logger)

            data['high_trait_gpt_scores'].append({
                'run_id': run_id,
                'gpt_score': score
            })

            if score:
                print(f"✓ {score}分")
                logger.info(f"  高特质 #{run_id}: {score}分")
            else:
                print(f"✗ 失败")
                logger.warning(f"  高特质 #{run_id}: 评分失败")

            time.sleep(0.5)

        # 评分低特质角色的3次回答
        if 'low_trait_gpt_scores' not in data:
            data['low_trait_gpt_scores'] = []

        for run_id, run_data in enumerate(data['low_trait_runs'], 1):
            main_answer = run_data.get('main_answer')
            followup_answer = run_data.get('followup_answer')

            if not main_answer:
                continue

            print(f"    低特质 #{run_id}/3 评分中...", end=' ', flush=True)

            prompt = generate_scoring_prompt(facet, question_text, followup_text, main_answer, followup_answer)
            score = call_gpt_scoring(gpt_client, prompt, logger=logger)

            data['low_trait_gpt_scores'].append({
                'run_id': run_id,
                'gpt_score': score
            })

            if score:
                print(f"✓ {score}分")
                logger.info(f"  低特质 #{run_id}: {score}分")
            else:
                print(f"✗ 失败")
                logger.warning(f"  低特质 #{run_id}: 评分失败")

            time.sleep(0.5)

        scored_count += 1

        # 每题保存一次
        save_results(output_dir, model_name, results)

    logger.info(f"\nGPT评分完成: 评分 {scored_count} 题, 跳过 {skipped_count} 题")
    return results


def run_dimension_role_experiment(model_name, output_dir, questions_dir, role_prompts_file, logger):
    """
    运行维度角色注入实验

    对每道题：
    - 确定对应的大五维度
    - 注入高特质角色，重复3次
    - 注入低特质角色，重复3次
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载角色prompts
    logger.info(f"加载角色prompts: {role_prompts_file}")
    role_prompts = load_role_prompts(role_prompts_file)

    # 加载题目
    logger.info(f"加载题目: {questions_dir}")
    behavior_questions, bfi2_questions = load_questions(questions_dir)

    # 根据模型名称自动选择API配置
    api_config = get_api_config(model_name)
    config_type = "OpenAI代理" if api_config == OPENAI_PROXY_CONFIG else "千帆平台"
    logger.info(f"模型 {model_name} 将使用: {config_type} ({api_config['base_url']})")

    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )

    # 检查是否已有结果文件
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'{model_name}_dim_role_results.json'

    existing_results = None
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            behavior_count = len(existing_results.get('behavior_results', {}))
            bfi2_count = len(existing_results.get('bfi2_results', {}))

            # 检查完整性：12道行为题 + 60道BFI-2题
            if behavior_count >= 12 and bfi2_count >= 60:
                logger.info(f"\n{'='*80}")
                logger.info(f"✅ 检测到完整结果文件，跳过实验")
                logger.info(f"  文件: {output_file}")
                logger.info(f"  行为题: {behavior_count}/12")
                logger.info(f"  BFI-2题: {bfi2_count}/60")
                logger.info(f"{'='*80}\n")
                return existing_results
            else:
                logger.info(f"\n{'='*80}")
                logger.info(f"⚠️  检测到不完整结果文件，将继续补充")
                logger.info(f"  文件: {output_file}")
                logger.info(f"  行为题: {behavior_count}/12")
                logger.info(f"  BFI-2题: {bfi2_count}/60")
                logger.info(f"{'='*80}\n")
        except Exception as e:
            logger.warning(f"读取已有结果文件失败: {e}，将重新开始实验")
            existing_results = None

    # 结果存储（如果有已有结果则继续，否则新建）
    if existing_results:
        results = existing_results
        logger.info(f"继续基于已有结果补充实验数据")
    else:
        results = {
            'model_name': model_name,
            'experiment_time': datetime.now().isoformat(),
            'behavior_results': {},
            'bfi2_results': {}
        }

    logger.info(f"\n{'='*80}")
    logger.info(f"开始维度角色注入实验")
    logger.info(f"模型: {model_name}")
    logger.info(f"{'='*80}\n")

    # ===== 第一部分：行为测试题 =====
    logger.info(f"\n第一部分：行为测试题（共{len(behavior_questions)}题）")
    logger.info(f"-"*80)

    for idx, question in enumerate(behavior_questions, 1):
        qid = question['id']
        question_text = question['text']

        # 确定对应的维度
        facet = BQ_TO_FACET.get(qid)
        if not facet:
            logger.warning(f"[{qid}] 未找到对应的子维度，跳过")
            continue

        big5_dim = FACET_TO_BIG5.get(facet)
        if not big5_dim or big5_dim not in role_prompts:
            logger.warning(f"[{qid}] 未找到对应的大五维度 ({big5_dim})，跳过")
            continue

        logger.info(f"\n[{idx}/{len(behavior_questions)}] {qid} - 维度: {big5_dim}")

        # 获取追问题目
        followup_text = question.get('followup', '')

        # 检查该题是否已完成
        if qid in results['behavior_results']:
            existing_data = results['behavior_results'][qid]
            high_runs = len(existing_data.get('high_trait_runs', []))
            low_runs = len(existing_data.get('low_trait_runs', []))
            if high_runs >= 3 and low_runs >= 3:
                logger.info(f"  ⏭️  已完成 (高特质: {high_runs}/3, 低特质: {low_runs}/3)，跳过")
                continue

        results['behavior_results'][qid] = {
            'question': question_text,
            'followup_question': followup_text,
            'facet': facet,
            'big5_dimension': big5_dim,
            'high_trait_runs': results['behavior_results'].get(qid, {}).get('high_trait_runs', []),
            'low_trait_runs': results['behavior_results'].get(qid, {}).get('low_trait_runs', [])
        }

        # 高特质角色 - 3次重复
        high_role = role_prompts[big5_dim]['High']
        logger.info(f"  高特质角色: {high_role}")

        for run_id in range(1, 4):
            # 每次运行使用不同的seed
            run_seed = 41 + run_id  # run_id=1->42, run_id=2->43, run_id=3->44

            print(f"    主问题 #{run_id}/3 (seed={run_seed})...", end=' ', flush=True)
            main_answer = call_model_with_role(client, model_name, high_role, question_text, max_tokens=300, temperature=0.0, seed=run_seed, logger=logger)

            # 如果有追问，基于主回答进行追问
            followup_answer = None
            if followup_text and main_answer:
                time.sleep(0.3)
                print(f"追问...", end=' ', flush=True)
                # 构建追问prompt，包含主回答
                followup_prompt = f"{question_text}\n\n你的回答是：{main_answer}\n\n{followup_text}"
                followup_answer = call_model_with_role(client, model_name, high_role, followup_prompt, max_tokens=200, temperature=0.0, seed=run_seed, logger=logger)

            if main_answer:
                results['behavior_results'][qid]['high_trait_runs'].append({
                    'run_id': run_id,
                    'seed': run_seed,
                    'main_answer': main_answer,
                    'followup_answer': followup_answer
                })
                print(f"✓")
            else:
                print(f"✗ 失败")
            time.sleep(0.5)

        # 低特质角色 - 3次重复
        low_role = role_prompts[big5_dim]['Low']
        logger.info(f"  低特质角色: {low_role}")

        for run_id in range(1, 4):
            # 每次运行使用不同的seed
            run_seed = 41 + run_id  # run_id=1->42, run_id=2->43, run_id=3->44

            print(f"    主问题 #{run_id}/3 (seed={run_seed})...", end=' ', flush=True)
            main_answer = call_model_with_role(client, model_name, low_role, question_text, max_tokens=300, temperature=0.0, seed=run_seed, logger=logger)

            # 如果有追问，基于主回答进行追问
            followup_answer = None
            if followup_text and main_answer:
                time.sleep(0.3)
                print(f"追问...", end=' ', flush=True)
                # 构建追问prompt，包含主回答
                followup_prompt = f"{question_text}\n\n你的回答是：{main_answer}\n\n{followup_text}"
                followup_answer = call_model_with_role(client, model_name, low_role, followup_prompt, max_tokens=200, temperature=0.0, seed=run_seed, logger=logger)

            if main_answer:
                results['behavior_results'][qid]['low_trait_runs'].append({
                    'run_id': run_id,
                    'seed': run_seed,
                    'main_answer': main_answer,
                    'followup_answer': followup_answer
                })
                print(f"✓")
            else:
                print(f"✗ 失败")
            time.sleep(0.5)

        # 保存中间结果
        save_results(output_dir, model_name, results)

    # ===== 第二部分：BFI-2题目 =====
    logger.info(f"\n第二部分：BFI-2人格测试（共{len(bfi2_questions)}题）")
    logger.info(f"-"*80)
    logger.info("使用选项顺序随机化（每次运行使用不同seed: 42/43/44）")

    # 初始化 bfi2_results（如果不存在）
    if 'bfi2_results' not in results:
        results['bfi2_results'] = {}

    # 原始选项定义
    original_options = [
        (1, "非常不同意"),
        (2, "不太同意"),
        (3, "态度中立"),
        (4, "比较同意"),
        (5, "非常同意")
    ]

    for idx, question in enumerate(bfi2_questions, 1):
        qid = question['id']
        question_text = question['text']

        # 确定对应的维度
        facet = BFI2_TO_FACET.get(qid)
        if not facet:
            logger.warning(f"[{qid}] 未找到对应的子维度，跳过")
            continue

        big5_dim = FACET_TO_BIG5.get(facet)
        if not big5_dim or big5_dim not in role_prompts:
            logger.warning(f"[{qid}] 未找到对应的大五维度 ({big5_dim})，跳过")
            continue

        print(f"\r进度: {idx}/{len(bfi2_questions)} [{qid}] - 维度: {big5_dim}", end='', flush=True)

        # 检查该题是否已完成
        if qid in results['bfi2_results']:
            existing_data = results['bfi2_results'][qid]
            high_runs = len(existing_data.get('high_trait_runs', []))
            low_runs = len(existing_data.get('low_trait_runs', []))
            if high_runs >= 3 and low_runs >= 3:
                continue

        # 保留已有数据
        if qid not in results['bfi2_results']:
            results['bfi2_results'][qid] = {
                'question': question_text,
                'facet': facet,
                'big5_dimension': big5_dim,
                'high_trait_runs': [],
                'low_trait_runs': [],
                'option_orders': []  # 记录每次运行的选项顺序
            }
        else:
            # 更新question信息但保留已有的runs
            results['bfi2_results'][qid].update({
                'question': question_text,
                'facet': facet,
                'big5_dimension': big5_dim
            })

        # 高特质角色 - 3次重复（每次使用不同的seed: 42, 43, 44）
        high_role = role_prompts[big5_dim]['High']
        for run_id in range(1, 4):
            # 每次运行使用不同的seed
            run_seed = 41 + run_id  # run_id=1->42, run_id=2->43, run_id=3->44

            # 使用固定的选项顺序（不再打乱）
            fixed_options = original_options  # 直接使用原始顺序

            # 构建固定顺序的选项文本
            options_text = "\n".join([f"{value} = {desc}" for value, desc in fixed_options])

            bfi2_prompt = f"""请根据以下描述，选择最符合你的选项（1-5）：

{question_text}

{options_text}

请只回答一个数字（1-5），不要有其他内容。"""

            # ernie-5.1 等推理模型需要更大的 max_tokens 来输出完整推理
            tokens_for_bfi = 1000 if model_name.startswith('ernie-5') else 100
            answer = call_model_with_role(client, model_name, high_role, bfi2_prompt, max_tokens=tokens_for_bfi, temperature=0.0, seed=run_seed, logger=logger)
            if answer:
                # 提取模型选择的值（1-5）
                answer_match = re.search(r'[1-5]', answer)
                if answer_match:
                    selected_value = int(answer_match.group())

                    results['bfi2_results'][qid]['high_trait_runs'].append({
                        'run_id': run_id,
                        'seed': run_seed,
                        'answer': answer,
                        'original_value': selected_value
                    })

                    logger.debug(f"[{qid}] 高特质 run{run_id} (seed={run_seed}): 分值{selected_value}")
            time.sleep(0.3)

        # 低特质角色 - 3次重复（每次使用不同的seed: 42, 43, 44）
        low_role = role_prompts[big5_dim]['Low']
        for run_id in range(1, 4):
            # 每次运行使用不同的seed
            run_seed = 41 + run_id  # run_id=1->42, run_id=2->43, run_id=3->44

            # 使用固定的选项顺序（不再打乱）
            fixed_options = original_options  # 直接使用原始顺序

            # 构建固定顺序的选项文本
            options_text = "\n".join([f"{value} = {desc}" for value, desc in fixed_options])

            bfi2_prompt = f"""请根据以下描述，选择最符合你的选项（1-5）：

{question_text}

{options_text}

请只回答一个数字（1-5），不要有其他内容。"""

            # ernie-5.1 等推理模型需要更大的 max_tokens 来输出完整推理
            tokens_for_bfi = 1000 if model_name.startswith('ernie-5') else 100
            answer = call_model_with_role(client, model_name, low_role, bfi2_prompt, max_tokens=tokens_for_bfi, temperature=0.0, seed=run_seed, logger=logger)
            if answer:
                # 提取模型选择的值（1-5）
                answer_match = re.search(r'[1-5]', answer)
                if answer_match:
                    selected_value = int(answer_match.group())

                    results['bfi2_results'][qid]['low_trait_runs'].append({
                        'run_id': run_id,
                        'seed': run_seed,
                        'answer': answer,
                        'original_value': selected_value
                    })

                    logger.debug(f"[{qid}] 低特质 run{run_id} (seed={run_seed}): 分值{selected_value}")
            time.sleep(0.3)

        # 每10题保存一次
        if idx % 10 == 0:
            save_results(output_dir, model_name, results)

    print()  # 换行

    # 最终保存
    save_results(output_dir, model_name, results)

    # ===== 第三部分：GPT-5.4评分 =====
    score_behavior_answers_for_role(behavior_questions, results, output_dir, model_name, logger)

    # 最终保存（包含GPT分数）
    save_results(output_dir, model_name, results)

    # 统计
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 实验完成！")
    logger.info(f"  行为测试: {len(results['behavior_results'])} 题")
    logger.info(f"  BFI-2测试: {len(results['bfi2_results'])} 题")
    logger.info(f"  每题条件: 高特质×3次 + 低特质×3次 = 6次回答")
    logger.info(f"{'='*80}\n")


def save_results(output_dir, model_name, results):
    """保存结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{model_name}_dim_role_results.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='维度角色注入实验')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型名称'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./dim_role_result',
        help='输出目录（默认: ./dim_role_result）'
    )
    parser.add_argument(
        '--questions-dir',
        type=str,
        default='.',
        help='题目文件所在目录（默认: 当前目录）'
    )
    parser.add_argument(
        '--role-prompts',
        type=str,
        default='./dim_role_prompt.json',
        help='角色prompt文件（默认: ./dim_role_prompt.json）'
    )

    args = parser.parse_args()

    # 配置日志
    logger = setup_logging(args.output)

    try:
        run_dimension_role_experiment(
            model_name=args.model,
            output_dir=args.output,
            questions_dir=args.questions_dir,
            role_prompts_file=args.role_prompts,
            logger=logger
        )
    except Exception as e:
        logger.error(f"实验过程中发生错误: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
