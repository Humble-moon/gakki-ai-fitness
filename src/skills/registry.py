"""
================================================================================
文件角色：SkillLoader — 从 skills/ 目录自动发现和加载业务 Skill
================================================================================
v3 升级 (2026-07-29):
  从 Python 硬编码 → 插件化目录结构。新增 Skill = 新建 skills/xxx/ + SKILL.md。

Skill 目录结构 (对标 Claude Code Skill):
  skills/<name>/
    SKILL.md          — YAML frontmatter: name, description, triggers,
                        params, plan_template, composes_with
    references.md     — 关联知识文档 ID 列表
    scripts/tools.py  — 工具函数 (calc_1rm, calc_bmi 等)

对外 API 与 v2 完全兼容: match(), get(), describe_all(), can_compose()
================================================================================
"""

import importlib.util
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import yaml

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Skill:
    """技能定义 —— 自包含的能力包。"""
    name: str
    description: str
    triggers: list
    retrieval_filters: dict = field(default_factory=dict)
    plan_template: str = ""
    knowledge_refs: list = field(default_factory=list)
    tools: dict[str, Callable] = field(default_factory=dict)
    plan_template_raw: str = ""
    param_schema: dict = field(default_factory=dict)
    composes_with: list = field(default_factory=list)

    def run_tool(self, tool_name: str, **kwargs) -> Any:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Skill '{self.name}' has no tool '{tool_name}'.")
        return tool(**kwargs)

    def validate_params(self, params: dict) -> list[str]:
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
        return errors


# ============================================================================
# SKILL.md 解析
# ============================================================================

def _parse_skill_md(filepath: Path) -> dict | None:
    """解析 SKILL.md 的 YAML frontmatter（PyYAML）。"""
    text = filepath.read_text(encoding="utf-8")
    parts = text.split("---")
    if len(parts) < 2 or not parts[1].strip():
        logger.warning(f"No YAML frontmatter in {filepath}")
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        logger.warning(f"YAML parse error in {filepath}: {e}")
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("name"):
        data["name"] = filepath.parent.name
    return data


def _load_references(skill_dir: Path) -> list[str]:
    """从 references.md 加载关联文档 ID 列表。"""
    ref_file = skill_dir / "references.md"
    if not ref_file.exists():
        return []
    refs = []
    for line in ref_file.read_text(encoding="utf-8").split("\n"):
        m = re.match(r'^-\s+([\w-]+)', line.strip())
        if m:
            refs.append(m.group(1))
    return refs


def _load_tools(skill_dir: Path) -> dict[str, Callable]:
    """从 scripts/tools.py 动态加载工具函数。"""
    tools_file = skill_dir / "scripts" / "tools.py"
    if not tools_file.exists():
        return {}
    try:
        module_name = f"skills.{skill_dir.name}.scripts.tools"
        spec = importlib.util.spec_from_file_location(module_name, tools_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            name: obj
            for name, obj in vars(module).items()
            if callable(obj) and not name.startswith("_")
        }
    except Exception as e:
        logger.warning(f"Failed to load tools from {tools_file}: {e}")
        return {}


# ============================================================================
# SkillLoader — 扫描 skills/ 目录，自动发现和加载 Skill
# ============================================================================

class SkillLoader:
    """Skill 加载器 — 遍历 skills/ 目录，解析 SKILL.md，组装 Skill 对象。"""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills: dict[str, Skill] = {}
        self._load_all(skills_dir)

    def _load_all(self, skills_dir: Path):
        if not skills_dir.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return
        # exercise_analysis 先注册（安全优先）
        priority = ["exercise_analysis", "fat_loss", "muscle_building"]
        loaded = set()
        for name in priority:
            d = skills_dir / name
            if d.is_dir():
                self._load_one(d)
                loaded.add(name)
        for d in sorted(skills_dir.iterdir()):
            if not d.is_dir() or d.name in loaded:
                continue
            if (d / "SKILL.md").exists():
                self._load_one(d)
        logger.info(f"Loaded {len(self.skills)} skills: {list(self.skills.keys())}")

    def _load_one(self, skill_dir: Path):
        md = skill_dir / "SKILL.md"
        if not md.exists():
            return
        data = _parse_skill_md(md)
        if not data:
            return
        skill = Skill(
            name=data.get("name", skill_dir.name),
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            retrieval_filters=data.get("retrieval_filters", {}),
            plan_template=data.get("plan_template", ""),
            knowledge_refs=_load_references(skill_dir),
            tools=_load_tools(skill_dir),
            plan_template_raw=data.get("plan_template", ""),
            param_schema=data.get("params", {}),
            composes_with=data.get("composes_with", []),
        )
        self.register(skill)

    # ---- 公共 API ----

    def register(self, skill: Skill):
        self.skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def match(self, user_input: str) -> str | None:
        # 第 0 层：高危医疗
        for kw in ["间盘", "腰突", "半月板", "髌骨", "脱臼", "腱鞘炎", "网球肘",
                    "肩峰撞击", "跟腱炎", "TFCC", "手术", "术后", "重建", "炎症",
                    "撕裂感", "弹响", "咔咔响", "损伤", "恢复期", "骨折"]:
            if kw in user_input:
                return "exercise_analysis"
        # 第 1 层：疼痛/伤病
        if any(t in user_input for t in ["疼", "痛", "不舒服", "伤到"]):
            return "exercise_analysis"
        # 第 2 层：训练目标
        if any(t in user_input for t in ["增肌", "增重", "变大", "维度", "增肌塑形"]):
            return "muscle_building"
        if any(t in user_input for t in ["减脂", "减重", "刷脂", "体脂", "减肥"]):
            return "fat_loss"
        if "瘦" in user_input:
            return "fat_loss"
        # 第 3 层：口语化
        if any(t in user_input for t in ["练粗", "练大", "练背", "练胸", "胸肌",
                                           "背肌", "胳膊粗", "倒三角"]):
            return "muscle_building"
        if any(t in user_input for t in ["变细", "燃脂", "有氧减", "减减"]):
            return "fat_loss"
        if any(t in user_input for t in ["体态", "驼背", "圆肩", "矫正", "骨盆", "前倾", "后倾"]):
            return "exercise_analysis"
        # 第 4 层：通用触发
        for t in ["姿势", "纠正", "借力", "错误", "不对", "泵感", "怎么练",
                   "是不是", "怎么办", "哪个更", "哪个好", "哪个", "区别",
                   "能不能", "会不会加重", "怎么纠正", "怎么改进",
                   "怎么判断", "怎么安全", "算不算", "动作"]:
            if t in user_input:
                return "exercise_analysis"
        return "muscle_building"

    def describe_all(self) -> str:
        descriptions = {
            "muscle_building": (
                "增肌训练计划。用户想增肌/变大/增重/增加维度时选择。"
                "检索过滤 rep_range=6-12, rest=60-90s，计划模板=四分化/五分化。"
                "可用工具: calc_1rm, calc_bmi, calc_calorie_target, check_volume, "
                "check_progressive_overload。"
            ),
            "fat_loss": (
                "减脂训练计划。用户想减脂/减重/瘦身/刷脂/塑形时选择。"
                "检索过滤 rep_range=12-15, rest=30-60s。"
                "可用工具: calc_bmi, calc_calorie_target, check_volume。"
            ),
            "exercise_analysis": (
                "动作分析与安全诊断。用户描述动作问题/疼痛/伤病/姿势纠正/体态矫正时选择。"
                "安全优先规则：涉及伤病/疼痛/功能障碍时必须选此技能。"
            ),
        }
        lines = []
        for name in self.skills:
            if name in descriptions:
                lines.append(f"- {name}: {descriptions[name]}")
        return "\n".join(lines)

    def get_knowledge_refs(self, skill_name: str) -> list[str]:
        s = self.skills.get(skill_name)
        return s.knowledge_refs if s else []

    def get_tools_for_skill(self, skill_name: str) -> dict[str, Callable]:
        s = self.skills.get(skill_name)
        return s.tools if s else {}

    def validate_skill_params(self, skill_name: str, params: dict) -> list[str]:
        s = self.skills.get(skill_name)
        return s.validate_params(params) if s else [f"Unknown skill: {skill_name}"]

    def can_compose(self, skill_a: str, skill_b: str) -> bool:
        a = self.skills.get(skill_a)
        return a is not None and skill_b in a.composes_with


SkillRegistry = SkillLoader  # 向后兼容别名
