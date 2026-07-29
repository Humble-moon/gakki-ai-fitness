---
name: muscle_building
description: >
  增肌训练计划生成。用户想增肌/变大/增重/增加维度时选择。
  检索过滤 rep_range=6-12, rest=60-90s, 计划模板=四分化/五分化。
  可用工具: calc_1rm(最大重量估算), calc_bmi(BMI计算),
  calc_calorie_target(热量目标), check_volume(训练量检查),
  check_progressive_overload(渐进超负荷检查)。
  关联知识: 增肌原理/分化训练/蛋白质/周期化/平台期突破。

# 触发条件
triggers:
  - 增肌
  - 增重
  - 变大
  - 维度
  - 增肌塑形
  - 练粗
  - 练大
  - 练背
  - 练胸
  - 胸肌
  - 背肌
  - 胳膊粗
  - 倒三角

# 检索策略
retrieval_filters:
  rep_range: "6-12"
  rest: "60-90s"

# 计划模板 (Markdown格式, Writer注入时使用)
plan_template: |
  ## 增肌训练计划
  ### 周期策略
  {periodization}

  ### 每周安排
  - 训练频率: {days_per_week} 天/周
  - 训练模式: {split}
  - 每肌群每周: 10-20 组

  ### 训练参数
  - 次数范围: 6-12 次/组（增肌黄金区间）
  - 组间休息: 60-90 秒

  ### 每日计划
  {daily_plan}

  ### 渐进策略
  {progression_plan}

  ### 饮食配合
  每日热量盈余约 300-500 kcal, 蛋白质 {protein_g}g/天

# 参数约束
params:
  days_per_week:
    min: 3
    max: 6
  rep_range:
    options: ["6-12", "5-8", "8-15"]
  split:
    options: ["四分化", "五分化", "推拉腿", "上下肢分化", "全身"]

# 可组合的 Skill
composes_with:
  - exercise_analysis
---
