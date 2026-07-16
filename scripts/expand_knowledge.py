# -*- coding: utf-8 -*-
"""
expand_knowledge.py - LLM batch expand fitness knowledge base (18 -> 30)

Usage:
    python scripts/expand_knowledge.py
    python scripts/expand_knowledge.py --dry-run
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.provider import LLMProvider

NEW_TOPICS = []

def _add(filename, title, topic, context):
    NEW_TOPICS.append({
        "filename": filename,
        "title": title,
        "topic": topic,
        "context": context,
    })

_add(
    "19-jianshen-wuqu.md",
    "健身常见误区与科学辟谣",
    "健身常见误区与科学辟谣",
    "面向健身新手，纠正最常见的 8-10 个健身误区。"
    "涵盖：练哪瘦哪（局部减脂不存在）、女生练壮（睾酮水平差异）、"
    "空腹有氧更减肥（代谢对比）、蛋白粉伤肾（RDA vs 健身需求）、"
    "出汗多=减脂多（体温调节 vs 脂肪氧化）、"
    "力量训练让人变笨重（肌肥大 vs 神经适应）等。"
    "每个误区先写常见错误观念，再用运动科学原理解释。"
)

_add(
    "20-nvxing-jianshen.md",
    "女性健身完全指南",
    "女性健身训练与营养指南",
    "专为女性训练者撰写。涵盖：女性与男性的生理差异（激素水平、肌纤维类型比例、"
    "月经周期对训练表现的影响）、女性常见训练误区（怕练壮、忽视上肢力量）、"
    "女性增肌减脂的特殊考量（热量缺口不宜过大以免影响月经周期）、"
    "孕期和产后训练注意事项、女性骨密度与力量训练的关系。实用导向，有可操作建议。"
)

_add(
    "21-zhonglaonian-xunlian.md",
    "中老年人安全训练指南",
    "中老年人群安全训练原则",
    "面向 50 岁以上人群的训练指南。涵盖：衰老带来的肌肉流失（少肌症）及力量训练的必要性、"
    "骨密度与负重训练的关系、中老年训练的安全红线（避免大重量低次数、避免高冲击动作、"
    "控制血压波动）、推荐训练模式（中重量中次数、多关节稳定性练习、平衡训练）、"
    "热身和冷身的重要性加倍、常见慢性病（高血压/糖尿病/关节炎）患者的训练注意事项。"
)

_add(
    "22-qingshaonian-tineng.md",
    "青少年体能训练科学指南",
    "青少年（13-18岁）体能训练",
    "面向青少年及家长。涵盖：青少年力量训练的安全性（骨骺板未闭合的风险与误区澄清"
    "——适当监督下的力量训练反而有助于骨骼发育）、推荐训练模式（自重为主+轻重量高次数、"
    "重点发展神经肌肉协调而非绝对力量）、应避免的动作（极限重量深蹲/硬拉/卧推、"
    "高冲击增强式训练）、营养需求（青少年增肌期的蛋白质和热量需求）、训练频率建议。"
)

_add(
    "23-beisai-yinshi.md",
    "健身比赛备赛饮食与训练策略",
    "健美/健体比赛备赛策略",
    "面向有意参加健身比赛（健美、健体、比基尼等）的训练者。涵盖：备赛阶段划分"
    "（增肌期→维持期→减脂期→高峰周）、各阶段的饮食策略（碳水循环、钠控制、"
    "水分管理）、高峰周（Peak Week）的操作细节与常见错误、备赛期有氧安排、"
    "体脂率目标与比赛状态的判断标准、心理准备（备赛期的社交隔离和饮食疲劳）。"
)

_add(
    "24-zengji-pingtaiji.md",
    "增肌平台期：原因诊断与突破策略",
    "增肌平台期的系统性突破方法",
    "几乎每个训练者都会遇到平台期。本文系统性地拆解平台期可能的 8 个原因："
    "训练量不足→增加组数/频率、训练量过多→deload、渐进超负荷失效→换动作/换次数区间、"
    "饮食热量不足→追踪 macros、蛋白质摄入不足→目标体重x1.6-2.2g/kg、"
    "睡眠质量差→影响睾酮/GH 分泌、神经疲劳→减量周、心理倦怠→换训练模式。"
    "提供平台期诊断自检清单——读者可以逐条排查。"
)

_add(
    "25-jianzhi-pingtaiji.md",
    "减脂平台期：突破热量缺口困境",
    "减脂平台期的科学突破",
    "专注于减脂停滞的解决方案。涵盖：代谢适应（Metabolic Adaptation）的生理机制"
    "——长期热量缺口后基础代谢率下降是正常现象不是新陈代谢坏了、"
    "突破策略的优先级：重新精确追踪热量摄入（很多人低估了20-30%）→"
    "增加 NEAT（非运动活动热量消耗——多走路多站立比多做有氧更可持续）→"
    "饮食休息期（Diet Break）的研究证据→反向饮食（Reverse Diet）的正确做法→"
    "有氧策略调整（用 HIIT 替代部分稳态有氧）。用数据说话，不贩卖焦虑。"
)

_add(
    "26-shendun-faq.md",
    "深蹲常见问题 FAQ：从入门到精通",
    "深蹲动作技术常见问题汇编",
    "以 FAQ 格式组织，覆盖深蹲最常见的 15-20 个问题。"
    "涵盖：脚跟抬起怎么办（踝关节活动度/举重鞋）、膝盖要不要过脚尖（个体差异/股骨长度vs躯干长度）、"
    "全蹲还是半蹲（训练目标决定）、高低杠位怎么选（高杠股四主导/低杠后链主导）、"
    "深蹲时下背酸正常吗（Butt Wink的解释和解决方案）、"
    "深蹲替代动作（膝盖/下背有问题时用什么替代）、"
    "深蹲重量的渐进策略、呼吸与腹内压（Valsalva）。回答简洁，每个80-120字，结论先行。"
)

_add(
    "27-yingla-faq.md",
    "硬拉常见问题 FAQ：安全与技术要求",
    "硬拉动作技术常见问题汇编",
    "以 FAQ 格式组织，覆盖硬拉最常见的 15-20 个问题。"
    "涵盖：传统硬拉 vs 相扑硬拉 vs 罗马尼亚硬拉怎么选（身体比例决定）、"
    "硬拉下背该不该圆（中立位 vs 轻微屈曲的争议与研究）、"
    "硬拉要不要腰带（什么时候开始用、腰带不保护你的下背）、"
    "硬拉和深蹲谁更伤腰（研究对比）、硬拉前热身流程、"
    "硬拉的上背部/握力短板怎么补、硬拉的替代动作（Trap Bar 硬拉/六角杠铃）、"
    "硬拉频率每周一次够不够、起始臀位如何确定。不重复过时的恐吓。"
)

_add(
    "28-wotui-faq.md",
    "卧推常见问题 FAQ：从杠铃到哑铃",
    "卧推动作技术常见问题汇编",
    "以 FAQ 格式组织，覆盖卧推最常见的 15-20 个问题。"
    "涵盖：握距怎么确定（肩宽x1.5-2）、要不要起桥（力量举vs健美式起桥的区别与目的）、"
    "杠铃触胸还是悬停（行程完整度vs肩关节安全）、"
    "肩痛怎么破（肩胛骨收紧+避免肘部外展超过45度+面拉预防）、"
    "卧推不涨怎么办（锁定弱点：底部→暂停卧推/中部→木板卧推/顶部→弹力带卧推）、"
    "哑铃和杠铃哪个好（互补关系不是二选一）、"
    "没有保护者时怎么安全卧推（安全杠/不要上卡扣/滚杠自救法）。回答有实操性。"
)

_add(
    "29-jianshenfang-liyi.md",
    "健身房礼仪与安全守则",
    "健身房行为规范与安全知识",
    "每个健身新人进健身房前都应该知道的事。涵盖：器械用完要归位（杠铃片卸下来"
    "——不仅是礼貌下一个人不知道重量可能受伤）、不要站在镜子前挡住别人的视线、"
    "不要在别人做组时说话（等组间休息）、出汗了擦凳子、不要霸占器械玩手机、"
    "不要给陌生人乱提建议（你不知道他的训练目标和伤病史）、"
    "安全知识：杠铃片一定要上卡扣、每次试举前检查器械是否稳固、"
    "大重量时一定要找保护者或设安全杠、学会失败救援（卧推/深蹲怎么安全失败）。"
)

_add(
    "30-jujia-xunlian.md",
    "居家训练完全指南：零器械到全身塑形",
    "居家训练的系统化方法（零器械到家用器械）",
    "为不能或不愿去健身房的人提供系统化的居家训练方案。涵盖："
    "零器械入门（俯卧撑家族：标准/宽距/钻石/上斜/下斜 + 深蹲家族：自重深蹲/分腿蹲/单腿蹲"
    "+ 核心：平板支撑变式 + 引体替代：门框划船/桌椅划船）、"
    "弹力带进阶（弹力带推胸/划船/侧平举/臀推——50块钱买全身训练）、"
    "可调节哑铃方案（最值得投资的家用器械搭配可调节凳覆盖全身）、"
    "TRX/悬挂带方案（便携+多角度+核心介入）、"
    "居家训练容易忽视的点（有氧替代：跳绳/原地跑/上下楼梯+腿部训练容易不足因为没有深蹲架）、"
    "每周居家训练模板（三分化/全身x3两种模式）、"
    "居家训练的安全注意事项（地面防滑、镜子检查姿势、别用不稳的椅子替代凳子做卧推）。"
)

CONTENT_REQUIREMENT = """请撰写一篇中文健身科普文章。

要求：
1. 使用 Markdown 格式：## 标题 + 正文段落 + 要点列表
2. 正文 400-600 字，口语化科普风格（不是论文摘要）
3. 末尾附 "### 核心要点" 部分，3-5 条要点总结
4. 每条建议有具体可操作的指导（不是"注意安全"这种废话）
5. 如果涉及科学数据，给出具体数字（如"每公斤体重 1.6-2.2g 蛋白质"）
6. 用中文写，不要翻译腔"""


def validate_doc(content: str) -> list:
    issues = []
    if len(content) < 300:
        issues.append("字数不足(%d字<300字)" % len(content))
    if not content.startswith("##"):
        issues.append("应以 ## 标题开头")
    if "### 核心要点" not in content:
        issues.append("缺少 ### 核心要点 部分")
    if "- " not in content and "* " not in content:
        issues.append("缺少要点列表(- 或 *)")
    return issues


def main():
    import argparse
    p = argparse.ArgumentParser(description="LLM batch expand fitness knowledge base")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", default="data/knowledge")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir = Path("..") / args.output_dir
    output_dir = output_dir.resolve()

    print("Existing: 18 self-written + 62 PubMed translated")
    print("Will generate %d new documents\n" % len(NEW_TOPICS))

    llm = LLMProvider()
    generated = []

    for topic in NEW_TOPICS:
        filename = topic["filename"]
        print("[Generate] %s - %s" % (filename, topic["title"]))

        full_prompt = (
            "Topic: %s\n\nContext & Key Points: %s\n\n" % (topic["topic"], topic["context"])
            + CONTENT_REQUIREMENT
        )

        doc_content = None
        for attempt in range(3):
            try:
                resp = llm.chat(
                    [{"role": "user", "content": full_prompt}],
                    temperature=0.6,
                )
                doc_content = resp.content.strip()

                if doc_content.startswith("```"):
                    lines = doc_content.split("\n")
                    doc_content = "\n".join(lines[1:-1])

                issues = validate_doc(doc_content)
                if issues:
                    print("  [Validate] %s" % issues)
                    if attempt < 2:
                        full_prompt += (
                            "\n\nLast output had issues: %s. Please fix and re-output."
                            % "; ".join(issues)
                        )
                        continue
                    else:
                        print("  [Warning] Retries exhausted, accepting as-is")
                break
            except Exception as e:
                print("  [Error] LLM call failed: %s" % e)
                if attempt == 2:
                    print("  [Skip] Cannot generate %s" % filename)
                    break

        if doc_content:
            generated.append((filename, doc_content))
            print("  [Done] %d characters" % len(doc_content))

    if not generated:
        print("\n[Failed] No documents generated.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Generated %d new documents" % len(generated))
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry Run] Preview first doc preview:")
        print(generated[0][1][:300])
        return

    for filename, content in generated:
        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        print("  Written: %s" % filepath)

    print("\nDocuments saved to: %s" % output_dir)
    print("Next: python -m src.main --ingest-knowledge")


if __name__ == "__main__":
    main()
