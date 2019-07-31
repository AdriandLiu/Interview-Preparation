## KNN

Few ideas on picking a value for ‘K’
1) Firstly, there is no physical or biological way to determine the best value for “K”, so we have to try out a few values before settling on one. We can do this by pretending part of the training data is “unknown”
2) **Small values for K can be noisy and subject to the effects of outliers.**
3) **Larger values** of K will have smoother decision boundaries which mean **lower variance but increased bias.**
4) Another way to choose K is though **cross-validation.** First is using some kind of validation process(cross-validation, leave-one-out,...) for K=1, 2, 3, ... As K increases, the error usually goes down, then stabilizes, and then raises again. Set **optimum K at the beginning of the stable zone.**
One way to select the cross-validation dataset from the training dataset. Take the small portion from the training dataset and call it a validation dataset, and then use the same to evaluate different possible values of K. This way we are going to predict the label for every instance in the validation set using with K equals to 1, K equals to 2, K equals to 3.. and then we look at what value of K gives us the best performance on the validation set and then we can take that value and use that as the final setting of our algorithm so we are minimizing the validation error .
5) In general, practice, choosing the value of **k is k = sqrt(N) where N stands for the number of samples in your training dataset.**
6) Try and **keep the value of k odd** in order to avoid confusion between two classes of data
