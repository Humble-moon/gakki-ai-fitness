---
name: exercise_analysis
description: >
  动作分析与安全诊断。用户描述动作问题/疼痛/伤病/姿势纠正/体态矫正时选择。
  安全优先规则：涉及伤病/疼痛/功能障碍时必须选此技能。
  无检索过滤, 计划模板=分析报告。
  可用工具: check_progressive_overload。
  关联知识: 常见伤病/下背痛/肩袖保护/膝盖健康/深蹲硬拉卧推FAQ/反复性劳损。

triggers:
  # 疼痛/不适
  - 疼
  - 痛
  - 不舒服
  - 咔咔响
  - 弹响
  - 撕裂感
  # 姿势/动作纠正
  - 姿势
  - 纠正
  - 借力
  - 错误
  - 不对
  # 发力感/肌肉感知
  - 找不到
  - 没感觉
  - 泵感
  - 发力感
  # 伤病/诊断关键词
  - 损伤
  - 间盘
  - 腰突
  - 半月板
  - 髌骨
  - 脱臼
  - 腱鞘炎
  - 网球肘
  - 肩峰撞击
  - 跟腱炎
  - 手术
  - 恢复期
  - 炎症
  # 疑问句式
  - 是不是
  - 怎么办
  - 哪个更
  - 哪个好
  - 哪个
  - 区别
  - 能不能
  - 会不会加重
  - 怎么纠正
  - 怎么改进
  - 怎么判断
  - 怎么安全
  # 能力/限制
  - 做不了
  - 算不算
  # 术后
  - 术后
  - 重建
  # 通用
  - 动作

retrieval_filters: {}

plan_template: |
  ## 动作分析报告
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
  {alternatives}

params: {}

composes_with:
  - muscle_building
  - fat_loss
---
