## Implementation and Evaluation of the Road Less Scheduled
#### Authors: Jingyu Wang, Hongyi Yu
As we have been studying the learning rate scheduling combined with various constraint optimization methods like Adam and SGD in deep learning applications, we are interested in further exploring how to improve the selection of learning rate schedules, given the sensitivity of the optimizers to the learning rates. There are concerns related to the commonly used optimization methods. For example, for the stepsizes in SGD applied to the Lipschitz continuous convex function, research reflects that the schedules conducted by convergence theory may not be optimal in practice. The performance of existing learning rate schedules highly depends on the optimization stopping step T. In other words, even though the dedicated hyperparameter tuning might take significant effort, the final stage might still be suboptimal. 

Recently, we discovered that one of the most recent methods might dramatically save energy spent on hyperparameter tuning while, at the same time, allowing the model to achieve the same or even outperform the benchmark. 

Aaron Defazio et al. proposed an approach that does not need to specify the optimization stopping step T or other relevant hyperparameters while demonstrating exceptional performance from convex issues to large-scale deep learning challenges. The only input for the optimizer is the learning rate, and no scheduler is needed. This idea raises our interest given the potential to save efforts on tuning learning rate without sacrificing the model performance. 

Thus, based on the paper The Road Less Scheduled, we propose the project to implement a schedule-free optimizer on multiple different tasks to ensure the reproduction of use cases indicated in the paper, evaluate the model performance, and explore new user cases: we decided to implement a complete training process for two non-convex deep learning problems, including one computer vision and one natural language processing task with the proposed schedule-free optimizer, and two convex problems to ensure that the advantage of the new methodology is only visible to non-convex problems. Then, we will compare the performance of the schedule-free optimizer to the standard optimizers and evaluate its efficiency, accuracy, and impact on hyperparameter tuning efforts. 

Generally, as indicated by our experiments, we do see the potential of the method in large-scale deep learning applications by potentially providing more flexibility on learning rate options with no additional hyperparameters required and possibly speeding up the convergence.

#### Project References
[1] Defazio, A., Yang, X. A., Mehta, H., Mishchenko, K., Khaled, A.,& Cutkosky, A. (2024). The Road Less Scheduled. arXiv preprint arXiv:2405.15682.

[2] Facebook Research. (n.d.). Schedule-Free Optimization [GitHub repos-itory]. Retrieved November 21, 2024, from https://github.com/facebookresearch/schedule_free/tree/main

[3] Zhang, A., Lipton, Z. C., Li, M., Smola, A. J., & others. (2021). Dive into Deep Learning. Retrieved from https://d2l.ai

[4] Zhang, A., Lipton, Z. C., Li, M., Smola, A. J., & others. (2021). Dive into Deep Learning [GitHub repository]. Retrieved from https://github.com/d2l-ai/d2l-en

[5] ManyThings.org. (n.d.). Anki Files for Sentence Pair Translations. Retrieved December 8, 2024, from https://www.manythings.org/anki/

[6] Polyak, Boris T. and Juditsky, Anatoli B., “Acceleration of stochastic approximation by averaging,” SIAM Journal on Control and Optimization,vol. 30, no. 4, pp. 838–855, 1992.

[7] Defazio, Aaron, “Momentum via Primal Averaging: Theoretical In-sights and Learning Rate Schedules for Non-Convex Optimization,” arXiv preprint arXiv:2010.00406, 2020.

[8] Elad Hazan, Adam Kalai, Satyen Kale, and Amit Agarwal, “Logarithmic Regret Algorithms for Online Convex Optimization,” In Learning Theory - 19th Annual Conference on Learning Theory, COLT 2006, Proceedings (pp. 499-513).Springer Verlag. https://doi.org/10.1007/11776420_37
