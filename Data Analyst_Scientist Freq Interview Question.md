## Question Scope



Title | 介绍 | 面试内容 |
|-------|--------|----------|
|Data Scientist(偏算法) | 一般意义上的DS，需要承担研究、优化算法或支持组内其他与数据相关的业务。 | Machine learning算法（较细致和具体），统计，online python coding test,  SQL（不多）|
|Data Scientist(偏产品) | 以FB的DS-analytics岗位为代表的产品分析方向的data scientist，介乎于DA和硬核DS之间的一个职位. | Machine learning算法（中等难度），统计，产品case分析，online SQL/Python coding test, A/B testing|
|Data Analyst | 和上一个DS比较相似，但更加少了些算法的问题，多了些具体情景的SQL的Online coding test和case | 简单的Machine learning算法问答（有的没有），统计，必考SQL，产品case分析, product sense, A/B testing|


* **Machine Learning**
  - Frequenct interview questions
     1). What is overfitting?  / Please briefly describe what is bias vs. variance.
      2). How do you overcome overfitting? Please list 3-5 practical experience.    / What is 'Dimension Curse'? How to prevent?
      3). Please briefly describe the Random Forest classifier. How did it work? Any pros and cons in practical implementation?
      4). Please describe the difference between GBM tree model and Random Forest.
      5). What is SVM? what parameters you will need to tune during model training? How is different kernel changing the classification result?
      6). Briefly rephrase PCA in your own way. How does it work? And tell some goods and bads about it.
      7). Why doesn't logistic regression use R^2?
      8). When will you use L1 regularization compared to L2?
      9). List out at least 4 metrics you will use to evaluate model performance and tell the advantage for each of them. (F1 score, ROC curve, recall, etc…)
    10). What would you do if you have > 30% missing value in an important field before building the model?

* **Statistics, Probability, A/B test**
      1). What is p-value? What is confidence interval? Explain them to a product manager or non-technical person.. (*cannot just answer 5% cutoff in the distribution, need more details and plain language*)
      2). How do you understand the "Power" of a statistical test?
      3). If a distribution is right-skewed, what's the relationship between medium, mode, and mean?
      4). When do you use T-test instead of Z-test? List some differences between these two.
      5). Dice problem-1: How will you test if a coin is fair or not? How will you design the process(may need coding)? what test would you use?
      6). Dice problem-2: How to simulate a fair coin with one unfair coin?
      7). [3 door questions](https://www.theproblemsite.com/games/treasure-hunt/door-hint) (classic questions)
      8). [Bayes Questions](http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability6.html):  Tom takes a cancer test and the test is advertised as being 99% accurate: if you have cancer you will test positive 99% of the time, and if you don't have cancer, you will test negative 99% of the time. If 1% of all people have cancer and Tom tests positive, what is the prob that Tom has the disease? (Very classic *cancer screen* question. Other similar questions could be solved if comprehend this one)
      9). How do you calculate the sample size for an A/B testing?
    10). If after running an A/B testing you find the fact that the desired metric(i.e, Click Through Rate) is going up while another metric is decreasing(i.e., Clicks). How would you make a decision?
    11). Now assuming you have an A/B testing result reflecting your test result is kind of negative (i.e, p-value ~= 20%). How will you communicate with the product manager?
           If given the above 20% p-value, the product manager still decides to launch this new feature, how would you claim your suggestions and alerts?


* **Resources**
      1). Coursera: [Machine learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
          ***Classic but useful***
      2). [15 hours of expert ML videos](https://www.dataschool.io/15-hours-of-expert-machine-learning-videos/.) 
          ***Quickly go through ML fundamental algorithm***
      3). [An Introduction to Statistical Learning](https://www-bcf.usc.edu/~gareth/ISL/ISLR%20First%20Printing.pdf)
          ***One of the best book for Stat and ML***
      4). 《Practical Statistics for Data Scientists: 50 Essential Concepts》
          ***Practical book, tiny concepts but useful. Not deep but more boardly understanding of ML***
      5).   Medium: Towards Data Science
           Especially: [Machine Learning 101](https://medium.com/machine-learning-101)
          ***Clear and understandable. Interpreting abstruct algorithms intuitively, such as KNN, Random Forest, Decision Tree, SVM, Adaboost***
      6). [StackOverflow](https://stackoverflow.com/)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwODcxODU4OThdfQ==
-->