## Degree of Freedom
Imagine you’re a fun-loving person who loves to wear hats. You couldn't care less what a degree of freedom is. You believe that variety is the spice of life.

Unfortunately, you have constraints. You have only 7 hats. Yet you want to wear a different hat every day of the week.

7 hats

On the first day, you can wear any of the 7 hats. On the second day, you can choose from the 6 remaining hats, on day 3 you can choose from 5 hats, and so on.

When day 6 rolls around, you still have a choice between 2 hats that you haven’t worn yet that week. But after you choose your hat for day 6, you have no choice for the hat that you wear on Day 7. You must wear the one remaining hat. You had 7-1 = 6 days of “hat” freedom—in which the hat you wore could vary!

That’s kind of the idea behind degrees of freedom in statistics. **Degrees of freedom are often broadly defined as the number of "observations" (pieces of information) in the data that are free to vary when estimating statistical parameters.**

## KNN

Few ideas on picking a value for ‘K’
1) Firstly, there is no physical or biological way to determine the best value for “K”, so we have to try out a few values before settling on one. We can do this by pretending part of the training data is “unknown”
2) **Small values for K can be noisy and subject to the effects of outliers.**
3) **Larger values** of K will have smoother decision boundaries which mean **lower variance but increased bias.**
4) Another way to choose K is though **cross-validation.** First is using some kind of validation process(cross-validation, leave-one-out,...) for K=1, 2, 3, ... As K increases, the error usually goes down, then stabilizes, and then raises again. Set **optimum K at the beginning of the stable zone.**
One way to select the cross-validation dataset from the training dataset. Take the small portion from the training dataset and call it a validation dataset, and then use the same to evaluate different possible values of K. This way we are going to predict the label for every instance in the validation set using with K equals to 1, K equals to 2, K equals to 3.. and then we look at what value of K gives us the best performance on the validation set and then we can take that value and use that as the final setting of our algorithm so we are minimizing the validation error .
5) In general, practice, choosing the value of **k is k = sqrt(N) where N stands for the number of samples in your training dataset.**
6) Try and **keep the value of k odd** in order to avoid confusion between two classes of data


## Expected Value or Expectation vs Avg or Mean
The concept of expectation value or expected value may be understood from the following example. Let X represent the outcome of a roll of an unbiased six-sided die. The possible values for X are 1, 2, 3, 4, 5, and 6, each having the probability of occurrence of 1/6. The expectation value (or expected value) of X is then given by

(X)expected=1(1/6)+2⋅(1/6)+3⋅(1/6)+4⋅(1/6)+5⋅(1/6)+6⋅(1/6)=21/6=3.5
Suppose that in a sequence of ten rolls of the die, if the outcomes are 5, 2, 6, 2, 2, 1, 2, 3, 6, 1, then the average (arithmetic mean) of the results is given by

(X)average=(5+2+6+2+2+1+2+3+6+1)/10=3.0
We say that the average value is 3.0, with the distance of 0.5 from the expectation value of 3.5. If we roll the die N times, where N is very large, then the average will converge to the expected value, i.e.,(X)average=(X)expected. This is evidently because, when N is very large each possible value of X (i.e. 1 to 6) will occur with equal probability of 1/6, turning the average to the expectation value.
