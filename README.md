# Q学习算法
强化学习算法实现了一个确定性的《冰湖问题》（FrozenLake），这是一种“网格世界”问题，其中Q学习智能体通过学习一个预定的策略，能够在冰湖中找到最佳路径。程序使用Python编写了两个类：一个用于设置环境状态，另一个用于设置智能体。Q值表示状态-动作对的价值，算法通过根据这些Q值来选择当前状态下的最佳动作。执行该动作后，智能体会观察到相应的奖励和下一状态，并根据这些信息更新Q值。通过多次迭代，算法能够学习出最优路径，只要能够正确平衡“探索”（Exploration）和“利用”（Exploitation）。

**网格示意图：**

![网格](https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Images/grid.PNG?raw=true)

**方法：**
Q学习算法采用了“ε贪心”方法（epsilon greedy），即10%的时间随机选择动作，其余90%的时间选择最佳动作。Q值的计算公式如下：
\[
Q\_值 = (1-\alpha) \times Q[(i,j,动作)] + \alpha \times (奖励 + \gamma \times Q_{max}[下一个状态动作])
\]
其中：
- \(\alpha\) 是学习率（即Q值更新的步长）
- \(\gamma\) 是折扣因子，用于权衡未来奖励与当前奖励
- \(Q_{max}\) 是下一状态的最大Q值

每个回合中的Q值都会根据公式进行计算和更新。如果当前状态是结束状态，Q值将设为对应的奖励值——失败时为-5，成功时为+1，并且状态会重置为初始状态(0,0)。

**最优解：**

![最优解](https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Images/optimal_solution.PNG?raw=true)

**每回合奖励变化：**
为了观察奖励的变化，算法在10,000个回合中运行。结果显示，尽管算法刚开始时表现不佳，但它逐渐快速地学会了最佳路径。由于算法在10%的时间内会执行随机动作，因此它并不会总是停留在最优解上。为了优化这个过程，可以考虑随着训练的进行减少探索的比例，从而让算法更专注于利用当前已学到的最优策略。

![每回合奖励](https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Images/RewardPerEpisode.png?raw=true)
