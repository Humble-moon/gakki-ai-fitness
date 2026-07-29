---
name: fat_loss
description: >
  减脂训练计划生成。用户想减脂/减重/瘦身/刷脂/塑形时选择。
  检索过滤 rep_range=12-15, rest=30-60s, 计划模板=上下肢分化/全身。
  可用工具: calc_bmi, calc_calorie_target, check_volume。
  关联知识: 减脂原理/平台期/心理策略/饮食。

triggers:
  - 减脂
  - 减重
  - 瘦
  - 刷脂
  - 塑形
  - 体脂
  - 减肥
  - 变细
  - 燃脂
  - 有氧减
  - 减减

retrieval_filters:
  rep_range: "12-15"
  rest: "30-60s"

plan_template: |
  ## 减脂训练计划
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
  每日热量目标约 {daily_kcal} kcal, 蛋白质 {protein_g}g/天

params:
  days_per_week:
    min: 3
    max: 6
  rep_range:
    options: ["12-15", "10-12", "15-20"]

composes_with:
  - exercise_analysis
---
