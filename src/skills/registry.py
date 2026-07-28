"""
================================================================================
文件角色：技能注册中心（SkillRegistry）v2
================================================================================
v2 升级:
  - Skill 从扁平的"名字+触发词+模板标签"升级为自包含的"能力包"
  - 每个 Skill 包含: 触发词/知识引用/工具函数/计划模板/参数校验/组合规则
  - 内置工具: 1RM计算/BMI/热量估算/训练量计算/渐进超负荷检查
================================================================================
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional


# ============================================================================
# 内置工具函数 — 每个 Skill 可附带的可调用的计算/校验逻辑
# ============================================================================

def _calc_1rm(weight: float, reps: int) -> float:
    """Epley 公式估算 1RM（单次最大重量）。"""
    if reps <= 0 or weight <= 0:
        return 0.0
    return round(weight * (1 + reps / 30), 1)


def _calc_bmi(height_cm: float, weight_kg: float) -> dict:
    """计算 BMI 并返回分类。"""
    h = height_cm / 100
    bmi = round(weight_kg / (h * h), 1)
    if bmi < 18.5:
        cat = "偏瘦"
    elif bmi < 24:
        cat = "正常"
    elif bmi < 28:
        cat = "偏重"
    else:
        cat = "肥胖"
    return {"bmi": bmi, "category": cat}


def _calc_calorie_target(weight_kg: float, goal: str) -> dict:
    """估算每日热量目标（Mifflin-St Jeor 简化版）。"""
    base = weight_kg * 22  # 简化基础代谢估算
    if goal == "muscle_building":
        return {"daily_kcal": round(base * 1.15), "protein_g": round(weight_kg * 1.8),
                "note": "增肌期热量盈余 ~15%，蛋白质 1.8g/kg"}
    elif goal == "fat_loss":
        return {"daily_kcal": round(base * 0.8), "protein_g": round(weight_kg * 2.0),
                "note": "减脂期热量缺口 ~20%，蛋白质 2.0g/kg 防肌肉流失"}
    return {"daily_kcal": round(base), "protein_g": round(weight_kg * 1.6),
            "note": "维持期"}


def _check_volume(exercises_per_week: int, sets_per_exercise: int, goal: str) -> dict:
    """检查每周训练量是否在合理范围。"""
    total_sets = exercises_per_week * sets_per_exercise
    if goal == "muscle_building":
        ok = 10 <= total_sets <= 25
        advise = "增肌每肌群每周 10-25 组为最佳区间（Schoenfeld 2016）"
    elif goal == "fat_loss":
        ok = 12 <= total_sets <= 30
        advise = "减脂期可稍高容量，但不超过 30 组/周以免恢复不足"
    else:
        ok = 8 <= total_sets <= 20
        advise = "维持/入门建议每周 8-20 组"
    return {"total_sets": total_sets, "ok": ok, "advice": advise}


def _check_progressive_overload(current_plan: dict, history: list = None) -> dict:
    """检查当前计划是否体现了渐进超负荷原则。"""
    # 简化检查：是否有重量/次数/组数的增长空间
    issues = []
    for day in current_plan.get("days", []):
        for ex in day.get("exercises", []):
            reps = ex.get("reps", "")
            if isinstance(reps, str) and "-" not in reps:
                issues.append(f"{ex.get('name','?')}: 建议用范围次数(如 8-12)而非固定次数")
    return {"ok": len(issues) == 0, "issues": issues, "principle": "渐进超负荷需要可追踪的进步路径"}


# ============================================================================
# Skill 数据结构 v2
# ============================================================================

@dataclass
class Skill:
    """技能定义 —— 自包含的能力包。

    v2 新增字段:
        knowledge_refs:    关联的知识文档 ID 列表，检索时可以限定在这些文档内搜索
        tools:             该技能可用的工具函数 {函数名: Callable}
        plan_template_raw: 实际训练计划模板内容（而非只是一个标签字符串）
        param_schema:      参数约束 schema（如 rep 范围/频率范围/禁忌）
        composes_with:     可组合的 Skill 名称列表（如增肌+伤病适配）
    """
    name: str
    description: str
    triggers: list
    retrieval_filters: dict = field(default_factory=dict)
    plan_template: str = ""                               # [旧字段保留兼容] 模板标签
    # --- v2 新增 ---
    knowledge_refs: list = field(default_factory=list)    # 关联文档 ID
    tools: dict[str, Callable] = field(default_factory=dict)
    plan_template_raw: str = ""                           # 实际模板 markdown
    param_schema: dict = field(default_factory=dict)
    composes_with: list = field(default_factory=list)

    def run_tool(self, tool_name: str, **kwargs) -> Any:
        """调用技能的某个工具函数。"""
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Skill '{self.name}' has no tool '{tool_name}'. "
                           f"Available: {list(self.tools.keys())}")
        return tool(**kwargs)

    def validate_params(self, params: dict) -> list[str]:
        """根据 param_schema 校验参数，返回错误列表。"""
        errors = []
        for key, rules in self.param_schema.items():
            val = params.get(key)
            if val is None:
                continue
            if "min" in rules and val < rules["min"]:
                errors.append(f"{key}={val} 低于最低值 {rules['min']}")
            if "max" in rules and val > rules["max"]:
                errors.append(f"{key}={val} 超过最高值 {rules['max']}")
            if "options" in rules and val not in rules["options"]:
                errors.append(f"{key}='{val}' 不在允许值 {rules['options']} 中")
            if "forbidden" in rules and val in rules["forbidden"]:
                errors.append(f"{key}='{val}' 为禁忌值: {rules['forbidden']}")
        return errors


# ============================================================================
# SkillRegistry v2
# ============================================================================

class SkillRegistry:
    """技能注册中心 v2。"""

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._load_builtin()

    # ------------------------------------------------------------------
    # 内置技能加载
    # ------------------------------------------------------------------

    def _load_builtin(self):
        """加载三个内置技能（v2 增强版）。"""
        # 注册顺序决定优先级（先注册先匹配），保持不变

        # --- exercise_analysis ---
        self.register(Skill(
            name="exercise_analysis",
            description="动作质量分析与安全诊断",
            triggers=[
                "疼", "痛", "不舒服", "咔咔响", "弹响", "撕裂感",
                "姿势", "纠正", "借力", "错误", "不对",
                "找不到", "没感觉", "泵感", "发力感",
                "损伤", "间盘", "腰突", "半月板", "髌骨", "脱臼",
                "腱鞘炎", "网球肘", "肩峰撞击", "跟腱炎",
                "手术", "恢复期", "炎症",
                "是不是", "怎么办", "哪个更", "哪个好", "哪个", "区别",
                "能不能", "会不会加重", "怎么纠正", "怎么改进",
                "怎么判断", "怎么安全",
                "做不了", "算不算",
                "术后", "重建",
                "动作",
            ],
            retrieval_filters={},
            plan_template="分析报告",
            # v2: 关联知识文档（动作标准规范+伤病预防文档）
            knowledge_refs=[
                "10-changjian-shangbing",
                "11-xiabeitong-guanli",
                "12-jianxiu-baohu",
                "13-xiguanjie-jiankang",
                "14-shendun-jishu",
                "15-yingla-jishu",
                "16-wotui-jishu",
                "17-tuiju-jishu",
                "26-shendun-faq",
                "27-yingla-faq",
                "28-wotui-faq",
                "54-fanfuxing-laosun",
            ],
            tools={
                "check_progressive_overload": _check_progressive_overload,
            },
            plan_template_raw="""## 动作分析报告模板
### 用户描述
{user_description}
### 标准动作规范
{standard_reference}
### 偏差分析
{deviation_analysis}
### 可能原因
{possible_causes}
### 改进建议
{suggestions}
### 严重程度
{severity} (1-5, 5=需立即就医)
### 替代动作推荐
{alternatives}""",
            param_schema={},
            composes_with=["muscle_building", "fat_loss"],
        ))

        # --- fat_loss ---
        self.register(Skill(
            name="fat_loss",
            description="减脂训练计划生成",
            triggers=["减脂", "减重", "瘦", "刷脂", "塑形", "体脂"],
            retrieval_filters={"rep_range": "12-15", "rest": "30-60s"},
            plan_template="上下肢分化/全身",
            knowledge_refs=[
                "02-jianzhi-yuanli",
                "25-jianzhi-pingtaiji",
                "46-jianzhi-xinli",
                "51-ketogenic-jianshen",
            ],
            tools={
                "calc_bmi": _calc_bmi,
                "calc_calorie_target": lambda **kw: _calc_calorie_target(**kw, goal="fat_loss"),
                "check_volume": _check_volume,
            },
            plan_template_raw="""## 减脂训练计划
### 每周安排
- 训练频率: {days_per_week} 天/周
- 训练模式: {split}（推荐上下肢分化/全身训练，提高训练频率）
### 训练参数
- 次数范围: 12-15 次/组（较高次数增加卡路里消耗）
- 组间休息: 30-60 秒（维持心率）
### 每日计划
{daily_plan}
### 有氧安排
推荐每周 2-3 次中等强度有氧（30-45分钟），在力量训练后或不同天进行
### 饮食配合
每日热量目标约 {daily_kcal} kcal，蛋白质 {protein_g}g/天""",
            param_schema={
                "days_per_week": {"min": 3, "max": 6},
                "rep_range": {"options": ["12-15", "10-12", "15-20"]},
            },
            composes_with=["exercise_analysis"],
        ))

        # --- muscle_building ---
        self.register(Skill(
            name="muscle_building",
            description="增肌训练计划生成",
            triggers=["增肌", "增重", "变大", "维度", "增肌塑形"],
            retrieval_filters={"rep_range": "6-12", "rest": "60-90s"},
            plan_template="四分化/五分化",
            knowledge_refs=[
                "01-zengji-yuanli",
                "05-xinren-rumen",
                "06-fenhua-xunlian",
                "07-danbaizhi-yinshi",
                "18-zhouqihua-xunlian",
                "24-zengji-pingtaiji",
                "35-xunlian-zhouqihua-shenru",
                "41-jinzhan-tupo",
            ],
            tools={
                "calc_1rm": _calc_1rm,
                "calc_bmi": _calc_bmi,
                "calc_calorie_target": lambda **kw: _calc_calorie_target(**kw, goal="muscle_building"),
                "check_volume": _check_volume,
                "check_progressive_overload": _check_progressive_overload,
            },
            plan_template_raw="""## 增肌训练计划
### 周期策略
{periodization}（推荐线性周期化入门 → 波动周期化进阶）
### 每周安排
- 训练频率: {days_per_week} 天/周
- 训练模式: {split}（推荐四分化/五分化或推拉腿）
### 训练参数
- 次数范围: 6-12 次/组（增肌黄金区间）
- 组间休息: 60-90 秒
- 每肌群每周: 10-20 组
### 每日计划
{daily_plan}
### 渐进策略
{progression_plan}
### 饮食配合
每日热量盈余约 300-500 kcal，蛋白质 {protein_g}g/天，碳水占剩余热量的 50-60%""",
            param_schema={
                "days_per_week": {"min": 3, "max": 6},
                "rep_range": {"options": ["6-12", "5-8", "8-15"]},
                "split": {"options": ["四分化", "五分化", "推拉腿", "上下肢分化", "全身"]},
            },
            composes_with=["exercise_analysis"],
        ))

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def register(self, skill: Skill):
        """注册（或覆盖）一个技能。"""
        self.skills[skill.name] = skill

    def match(self, user_input: str) -> str | None:
        """根据用户输入匹配最佳技能（五层关键词匹配体系，与 v1 一致）。"""
        # 第 0 层：高危医疗关键词
        MEDICAL_EMERGENCY = [
            "间盘", "腰突", "半月板", "髌骨", "脱臼",
            "腱鞘炎", "网球肘", "肩峰撞击", "跟腱炎", "TFCC",
            "手术", "术后", "重建", "炎症", "撕裂感", "弹响", "咔咔响",
            "损伤", "恢复期", "骨折",
        ]
        for kw in MEDICAL_EMERGENCY:
            if kw in user_input:
                return "exercise_analysis"

        # 第 1 层：疼痛/伤病安全信号
        if any(t in user_input for t in ["疼", "痛", "不舒服", "伤到"]):
            return "exercise_analysis"

        # 第 1.5 层：功能障碍信号
        has_dysfunction = any(
            t in user_input for t in ["做不了", "没感觉", "发力感", "找不到", "力竭"]
        )

        # 第 2 层：训练目标
        has_fat = any(t in user_input for t in ["减脂", "减重", "刷脂", "体脂", "减肥"])
        has_muscle = any(t in user_input for t in ["增肌", "增重", "变大", "维度", "增肌塑形"])

        if has_muscle:
            return "muscle_building"
        if has_fat:
            return "fat_loss"
        if "瘦" in user_input:
            return "fat_loss"

        if has_dysfunction:
            return "exercise_analysis"

        # 第 3 层：口语化补充
        if any(t in user_input for t in ["练粗", "练大", "练背", "练胸", "胸肌", "背肌", "胳膊粗", "倒三角"]):
            return "muscle_building"
        if any(t in user_input for t in ["变细", "燃脂", "有氧减", "减减"]):
            return "fat_loss"
        if any(t in user_input for t in ["体态", "驼背", "圆肩", "矫正", "骨盆", "前倾", "后倾"]):
            return "exercise_analysis"

        # 第 4 层：通用触发词
        for t in ["姿势", "纠正", "借力", "错误", "不对", "泵感", "怎么练",
                   "是不是", "怎么办", "哪个更", "哪个好", "哪个", "区别",
                   "能不能", "会不会加重", "怎么纠正", "怎么改进",
                   "怎么判断", "怎么安全", "算不算", "动作"]:
            if t in user_input:
                return "exercise_analysis"

        return "muscle_building"  # 兜底

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def describe_all(self) -> str:
        """生成 LLM 可用的技能描述文本（v2 增强版：含工具列表+知识范围）。"""
        descriptions = {
            "muscle_building": (
                "增肌训练计划。用户想增肌/变大/增重/增加维度时选择。"
                "检索过滤 rep_range=6-12, rest=60-90s，计划模板=四分化/五分化。"
                "可用工具: calc_1rm(最大重量估算), calc_bmi(BMI计算), "
                "calc_calorie_target(热量目标), check_volume(训练量检查), "
                "check_progressive_overload(渐进超负荷检查)。"
                "关联知识: 增肌原理/分化训练/蛋白质/周期化/平台期突破。"
            ),
            "fat_loss": (
                "减脂训练计划。用户想减脂/减重/瘦身/刷脂/塑形时选择。"
                "检索过滤 rep_range=12-15, rest=30-60s，计划模板=上下肢分化/全身。"
                "可用工具: calc_bmi, calc_calorie_target, check_volume。"
                "关联知识: 减脂原理/平台期/心理策略/饮食。"
            ),
            "exercise_analysis": (
                "动作分析与安全诊断。用户描述动作问题/疼痛/伤病/姿势纠正/体态矫正时选择。"
                "无检索过滤，计划模板=分析报告。"
                "可用工具: check_progressive_overload。"
                "关联知识: 常见伤病/下背痛/肩袖保护/膝盖健康/深蹲硬拉卧推FAQ/反复性劳损。"
                "安全优先规则：涉及伤病/疼痛/功能障碍时必须选此技能。"
            ),
        }
        lines = []
        for name in self.skills:
            if name in descriptions:
                lines.append(f"- {name}: {descriptions[name]}")
        return "\n".join(lines)

    def get_knowledge_refs(self, skill_name: str) -> list[str]:
        """获取 Skill 关联的知识文档 ID 列表，用于限定检索范围。"""
        skill = self.skills.get(skill_name)
        if skill is None:
            return []
        return skill.knowledge_refs

    def get_tools_for_skill(self, skill_name: str) -> dict[str, Callable]:
        """获取 Skill 的可用工具集。"""
        skill = self.skills.get(skill_name)
        if skill is None:
            return {}
        return skill.tools

    def validate_skill_params(self, skill_name: str, params: dict) -> list[str]:
        """根据 Skill 的 param_schema 校验参数。"""
        skill = self.skills.get(skill_name)
        if skill is None:
            return [f"Unknown skill: {skill_name}"]
        return skill.validate_params(params)

    def can_compose(self, skill_a: str, skill_b: str) -> bool:
        """检查两个 Skill 是否可以组合（如增肌+伤病适配）。"""
        a = self.skills.get(skill_a)
        if a is None:
            return False
        return skill_b in a.composes_with
