from isaaclab.utils import configclass
from rsl_rlex.symmetry.modules import modules_cfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class G1CfgRNN(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 1000
    experiment_name = "g121_symmetry"
    empirical_normalization = False
    policy = modules_cfg.SymmetryRecurrentCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        symmetry_objs="HistorySymmetry",

        rnn_type='lstm',
        rnn_hidden_dim=128,
        rnn_num_layers=1
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,# 0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1CfgV2RNN(G1CfgRNN):
    experiment_name = "g121_symmetryv2"
