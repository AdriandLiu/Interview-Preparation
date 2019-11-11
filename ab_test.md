## Metrics:
1. Practical significance: magnitude of the difference, which is known as the effect size. Cohen's *d* is the measure of practical significance, 
2. Statistical significance, typically 0.05
3. Binomial distribution
4. Confidence interval, typically 95% 

5. Pooled standard deviriation for two groups of data
6. Sensitivity/recall, true positive rate (correctly identify a patient who has the condition)
7. Statistical power (1-beta), ususally 0.8. know the relationship between power and sample size. (Statistical power is affected chiefly by the size of the effect and the size of the sample used to detect it. Bigger effects are easier to detect than smaller effects, while large samples offer greater test sensitivity than small samples.)
Larger sample size narrow the distribution of test statistics, so power increases as sample size n increases. 

8. Significant level/alpha: alpha increase, power increases.
9. Click-through-probability (unique clicks / unique visits)
10. Estimated difference, d_hat, typically 0.02 (if d_min > 0.02, difference is ok for launch): X_experiment/N_experiment - X_control/N_control
11. Margin of error, m, SE_pool * Z_score(1.96). Margin of error is usually defined as the "radius" (or half the width) of a confidence interval. The larger the margin of error, the less confidence one should have that the poll's reported results are close to the "true" figures; that is, the figures for the whole population
12. Confidence interval, (d_hat - m, d_hat + m). Presentation: if d_hat - m > d_min, it means it is highly likely that the difference of click-through-prob is at least d_min%, say 2%. 
**IF m < statistical significant level AND d_hat - m > d_min, which means they are both significant, then LAUNCH**
![](./pics/confidence_level.PNG)
![](./pics/power.png)

![](./pics/calculations.PNG)


## Policy
### Risk
Take the minimal risk, such as any tests on health or finance related are not in the minimal risk

### Benefits
这项研究的益处是什么？即使风险很小，但是研究结果有什么意义呢？

### Alternative plan
参与者还有什么其他选择？比如，您要测试一个搜索引擎的变化，参与者始终可以选择使用其他搜索引擎。 主要问题在于参与者的选择越少，强制性以及参与者是否能选择参与与否的问题就越多，以及这些问题如何在风险和益处之间实现平衡。

例如，医学临床试验要测试治疗癌症的新药，鉴于大多数参与者面临的另一个主要选择是死亡，那么在知情同意的情况下，参与者的风险还是很高的。

对于在线实验，需要考虑的问题是用户可以使用的其他服务有哪些，以及转换服务的成本是多少，包括时间、金钱、信息等。

### Data Sensitivity
- 参与者了解从他们那里收集的数据是什么吗？
- 如果公开这些数据，会给他们带来什么伤害？
- 他们是否期望这些数据被当做隐私和保密信息？
例如，如果参与者在公共环境中接受观察（如：足球场），那就没有隐私可言。如果这项研究是针对现有公共数据，也就谈不上更进一步的保密了。

Main three concerns:
- 对于新收集和存储的数据，数据的敏感性如何？处理数据时采取了什么内部防御措施？例如设置了何种访问控制？如何捕捉和管理违反该防御措施的行为等？
- 然后，如何使用收集的数据？如何保护参与者的数据？如何向参与者保证为了此研究而向他们收集的数据不会被用于其他目的？随着数据敏感性的增强，这一点越来越重要。
- 最后，哪些数据会更加广泛地公开，是否会给参与者带来任何额外风险



## Metrics
1. How to define metrics in the early stage? First to make a high-level concepts for metrics
For example Audicity
- Business objective:
  - Helping students get jobs
  - Financial sustainability

2. Choosing metrics






