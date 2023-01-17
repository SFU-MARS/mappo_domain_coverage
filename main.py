import time
import numpy as np
from matplotlib import pyplot as plt

from common.env_utils import make_env
from common.env_config import square, hexagon, polygon_1, polygon_2, polygon_3, polygon_4

from common.SubprocVecEnv import SubprocVecEnv
from mappo import MAPPO

if __name__ == '__main__':

    # running 18 parallel environments, contains six types of default shape across 7 to 9 agents.
    vec_env = SubprocVecEnv([make_env(**square), make_env(**square), make_env(**square),
                             make_env(**polygon_4), make_env(**polygon_4), make_env(**polygon_4),
                             make_env(**polygon_3), make_env(**polygon_3), make_env(**polygon_3),
                             make_env(**polygon_2), make_env(**polygon_2), make_env(**polygon_2),
                             make_env(**polygon_1), make_env(**polygon_1), make_env(**polygon_1),
                             make_env(**hexagon), make_env(**hexagon), make_env(**hexagon)])

    """ 
      For the demo project, the default setup is to continue training from a pre-trained model, where the pre-trained 
      model is obtained by doing imitation learning to an classical controller, this controller will perform average 
      episodic reward as about -12000. The default setup will give the converged result about 200 iterations.
    """
    agent = MAPPO(env=vec_env, num_steps=1200, alpha=0.001, beta=0.003, max_std=0.5, clip=0.2, gamma=0.99, k_epochs=16,
                  use_bcloss=False, bcloss_weight=1, use_init_model=True)

    """ 
      Defined the number of iteration performed for the training. In each iteration we running 18 parallel environments,
      and each environment will collect a "num_steps" amount of sampled time steps. The default termination setup for 
      this project is 300, the total number of trajectories collected in the buffer is  num_env * (num_steps / 300 ).
    """
    num_iteration = 1000

    """ 
      The baseline is used to determine the use the behavior cloning loss, this value is determined from the 
      expert data. For the demo project, this value is obtained from the average episodic reward of 10000 trajectories 
      sampled from the classical controller(this controller is used for imitation learning to get pre-trained model).
    """
    baseline = -12000

    """ 
      if continue training after some breakpoint is needed， uncomment the following two lines and give the checkpoint.
      the checkpoint should be the last saved model named as "generic_policy_{}".format(checkpoint).
    """
    # path, checkpoint = "./trained_model/", checkpoint
    # load_model(path, checkpoint)

    # start the training process
    training_log = []  # used to store some training information over the whole training process.
    for i in range(num_iteration):
        start = time.time()

        agent.rollout()
        log_info = agent.update(baseline)
        agent.clear_buffer()

        end = time.time()
        training_log.append(log_info)

        print("=======================================================================================")
        print("Training Information for Iteration " + str(i + 1) + ":")
        print("---------------------------------------------------------------------------------------")
        print("Running Time: {:.1f}s".format(end - start))
        print("Average Episodic Rewards for current iterations are:")
        print(str(np.mean(np.asarray(log_info[0], dtype=float))))
        print("Current Performance Ratio: " + str(log_info[1]))
        print("=======================================================================================\n")

        # save the training_log and model after some a certain amount of training iteration.
        if ((i + 1) % 100) == 0:
            np.save(r"./log/training_log.npy", np.asarray(training_log, dtype=float))
            agent.save_model(i + 1)

    # plotting the learning curve.
    t = np.arange(num_iteration)
    t = np.asarray(t, dtype=int)
    episodic_returns = np.mean(np.asarray(training_log, dtype=float), axis=1).squeeze()

    plt.plot(t, episodic_returns, linewidth=1)
    plt.title(label="Episodic Reward Vs Number of Iterations")
    plt.show()
