# 2026-09-13：用户指定Mean100 final800为正式基线

用户决定：“把当前mean100作为新的baseline基线并记录，我需要其他codex窗口也能同步知晓这一变化”。
本记录对应此前分组图中的Mean100，也就是Mean主分支＋E3＋mean≤100%＋Set的800步final，不是旧400步或best740。

新ID：`mean100_v5_mean_anchor_mixed_real_no_p12_800_20260913`。
旧ID：`e3_mean100_v3_400_20260908_run01`，原JSON/Markdown按原文归档于本目录；归档内部的current字样是历史快照，不能覆盖根目录CURRENT_BASELINE记录。

核验：final800和checkpoint_last的SHA256均为`ae61ff74cb40cc34dfeac0c221b13cdbfc6e4e9b84514e5282e3ea6ed7dafa03`；best为740步。
checkpoint内anchor=mean_rl3，use_set_branch=true；配置max_steps800，训练P01–P11＋45/55，排除P12。
两个权重、冻结配置、真实清单哈希已在CPU上核对；本次未启动GPU任务。

登记更新：CURRENT_BASELINE.md/json、README顶部、CURRENT_NETWORK_AND_EXPERIMENT_PLAN顶部、训练手册顶部。
新增根目录AGENTS.md，要求后续任务每次重新读取基线注册。已运行窗口需主动重读；不同克隆/worktree需先同步文件。
本次没有可用的跨任务发送消息工具，未宣称其他窗口已经确认；也未提交/推送Git或改变历史标签。

局限：用户选择作为后续参照不等于已证明统计显著、真实深度正确或分辨率全面提高。45/55为训练视场，无独立真实测试。
best740测试CSV不能当final800测试。保留旧实验源码的ID保护，不改写历史实验条件。
