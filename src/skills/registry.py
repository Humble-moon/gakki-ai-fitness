"""
================================================================================
文件角色：技能注册中心（SkillRegistry）
================================================================================
- 被调用者：编排引擎在接收到用户输入后，首先调用 SkillRegistry.match(user_input)
  来确定应该启动哪个技能（增肌计划/减脂计划/动作分析），然后根据匹配到的技能
  加载对应的 retrieval_filters 和 plan_template 配置。
- 调用者：本模块不调用其他模块，仅维护内置技能定义。
- 项目角色：意图识别 + 技能路由——将用户的自然语言输入映射到系统支持的具体
  技能，并为该技能提供默认的检索过滤参数和计划模板。
================================================================================
"""

from dataclasses import dataclass, field


@dataclass
class Skill:
    """
    技能定义数据结构。

    职责：封装一个技能的所有配置信息，包括识别触发词、检索过滤参数和计划模板。

    字段说明：
        name: str              - 技能唯一标识，如 "muscle_building"、"fat_loss"
        description: str       - 技能描述，供调试和日志使用
        triggers: list[str]    - 触发词列表，用户输入中包含任一触发词即匹配该技能。
                                 支持模糊匹配（如"增肌""变大""维度"都触发增肌技能）
        retrieval_filters: dict - 检索时应用的默认过滤条件。
                                 如增肌技能的 rep_range="6-12", rest="60-90s"
                                 这些参数会传给 Retriever 或 Writer 使用
        plan_template: str     - 计划模板类型标签。
                                 如 "四分化/五分化" 表示增肌场景下默认使用
                                 四分化或五分化训练计划模板
    """
    name: str
    description: str
    triggers: list
    retrieval_filters: dict = field(default_factory=dict)  # 默认空字典，避免可变默认参数陷阱
    plan_template: str = ""                                 # 默认空字符串


class SkillRegistry:
    """
    技能注册中心，管理可用技能定义并支持关键词匹配路由。

    职责：
    - 在初始化时加载内置技能（增肌、减脂、动作分析）。
    - 提供 register() 方法注册新技能（可扩展）。
    - 提供 match() 方法根据用户输入文本匹配最佳技能。
    - 提供 get() 方法按名称获取技能完整配置。

    设计思路——为什么用触发词匹配而不是 LLM 分类：
    1. 低成本：关键词匹配不需要一次 LLM API 调用。
    2. 低延迟：字符串匹配在毫秒级完成，LLM 分类需要秒级。
    3. 足够可靠：健身领域的意图分类场景有限（增肌/减脂/分析），触发词覆盖面
       足够广，误匹配概率低。
    4. 兜底策略：match() 在无匹配时默认返回 "muscle_building"（最常用场景），
       避免因用户输入措辞不规范而导致流程中断。
    """

    def __init__(self):
        """
        初始化技能注册中心。

        核心逻辑：
        1. 初始化空的 skills 字典。
        2. 调用 _load_builtin() 加载默认内置技能。
        3. 内置技能包括：增肌训练(muscle_building)、减脂训练(fat_loss)、
           动作分析(exercise_analysis)。
        """
        self.skills: dict[str, Skill] = {}
        self._load_builtin()

    def _load_builtin(self):
        """
        加载系统内置的三个默认技能。

        每个技能的配置逻辑：
        1. muscle_building（增肌）：
           - 触发词覆盖了"增肌""增重""变大""维度""增肌塑形"等多种口语表达
           - 检索过滤：rep_range 6-12（增肌最佳次数区间），rest 60-90s（组间休息）
           - 模板：四分化/五分化（适合中高级训练者，单次训练专注 1-2 个部位）

        2. fat_loss（减脂）：
           - 触发词覆盖"减脂""减重""瘦""刷脂""塑形"
           - 检索过滤：rep_range 12-15（减脂用较高次数），rest 30-60s（短休息维持心率）
           - 模板：上下肢分化/全身（提高训练频率和卡路里消耗）

        3. exercise_analysis（动作分析）：
           - 触发词覆盖"动作""姿势""感觉""疼""不舒服""是不是"
           - 注意："疼"和"不舒服"是关键触发词，表示用户可能正在经历伤病
           - 该技能无检索过滤参数，因为分析场景需要全量动作信息
           - 模板为"分析报告"而不是训练计划
        """
        # NOTE: exercise_analysis 必须最先注册，确保其触发词优先级最高。
        # 因为"增肌期腰痛"类查询同时含"增肌"和"痛"，应路由到动作分析而非增肌。
        self.register(Skill(
            name="exercise_analysis",
            description="动作质量分析",
            triggers=[
                # 疼痛/不适（"痛"是"疼"的同义词，之前漏了）
                "疼", "痛", "不舒服", "咔咔响", "弹响", "撕裂感",
                # 姿势/动作纠正
                "姿势", "纠正", "借力", "错误", "不对",
                # 发力感/肌肉感知问题
                "找不到", "没感觉", "泵感", "发力感",
                # 伤病/诊断关键词（精确医学术语，误匹配概率极低）
                "损伤", "间盘", "腰突", "半月板", "髌骨", "脱臼",
                "腱鞘炎", "网球肘", "肩峰撞击", "跟腱炎",
                "手术", "恢复期", "炎症",
                # 动作分析疑问句式
                "是不是", "怎么办", "哪个更", "哪个好", "哪个", "区别",
                "能不能", "会不会加重", "怎么纠正", "怎么改进",
                "怎么判断", "怎么安全",
                # 能力/限制表达
                "做不了", "算不算",
                # 伤后康复
                "术后", "重建",
                # 通用动作词
                "动作",
            ],
            retrieval_filters={},
            plan_template="分析报告"
        ))
        # fat_loss 必须在 muscle_building 之前注册，否则"增肌减脂"会先命中"增肌"
        self.register(Skill(
            name="fat_loss",
            description="减脂训练计划生成",
            triggers=["减脂", "减重", "瘦", "刷脂", "塑形", "体脂"],
            retrieval_filters={"rep_range": "12-15", "rest": "30-60s"},
            plan_template="上下肢分化/全身"
        ))
        self.register(Skill(
            name="muscle_building",
            description="增肌训练计划生成",
            triggers=["增肌", "增重", "变大", "维度", "增肌塑形"],
            retrieval_filters={"rep_range": "6-12", "rest": "60-90s"},
            plan_template="四分化/五分化"
        ))

    def register(self, skill: Skill):
        """
        注册一个新技能（或覆盖已有技能）。

        参数：
            skill: Skill  - Skill 数据类实例

        返回值：None

        核心逻辑：
        将技能以 name 为 key 存入 self.skills 字典。同名技能后注册的会覆盖先注册的，
        这允许运行时动态更新技能配置而无需修改代码。
        """
        self.skills[skill.name] = skill

    def match(self, user_input: str) -> str | None:
        """
        根据用户输入文本匹配最佳技能名称。

        参数：
            user_input: str  - 用户的原始输入文本

        返回值：
            str | None       - 匹配到的技能名称（如 "muscle_building"），
                              无匹配时默认返回 "muscle_building"（即永远不返回 None）

        核心逻辑：
        1. 遍历所有技能的触发词列表。
        2. 对每个触发词，用子串匹配（trigger in user_input）检查。
        3. 命中第一个匹配的技能就立即返回（短路匹配）。
           这意味着触发词可能有优先级问题——如果两个技能的触发词有重叠，
           先注册的技能优先被匹配。
        4. 无任何匹配时，默认兜底返回 "muscle_building"。

        为什么默认返回增肌而不是返回 None：
        健身场景下用户说不清具体需求的概率很高（如只发"给我一个计划"），
        此时增肌是最常见的默认需求。返回 None 会导致流程中断，用户体验差。
        """
        for name, skill in self.skills.items():
            for trigger in skill.triggers:
                if trigger in user_input:
                    return name
        # 无匹配时兜底使用增肌技能（最常用场景）
        return "muscle_building"

    def get(self, name: str) -> Skill | None:
        """
        按技能名称获取技能完整配置。

        参数：
            name: str  - 技能名称（如 "muscle_building"）

        返回值：
            Skill | None  - Skill 对象，不存在时返回 None

        使用场景：
        match() 返回技能名称后，调用 get() 获取该技能的 retrieval_filters
        和 plan_template，然后传递给后续的 Retriever / Writer 流程。
        """
        return self.skills.get(name)
