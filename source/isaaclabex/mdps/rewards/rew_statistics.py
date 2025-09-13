from .statistics import mean_joints, var_joints

mirror_or_synchronize = mean_joints.mirror_or_synchronize

rew_mean_zero = mean_joints.rew_mean_zero           # L1 penalty for deviation from desired joint positions.
rew_mean_symmetry = mean_joints.rew_mean_symmetry        # Norm penalty for joint positions.
rew_mean_step_symmetry = mean_joints.rew_mean_step_symmetry
rew_mean_constraint = mean_joints.rew_mean_constraint

rew_variance_zero = var_joints.rew_variance_zero
rew_variance_symmetry = var_joints.rew_variance_symmetry
rew_variance_constraint = var_joints.rew_variance_constraint
