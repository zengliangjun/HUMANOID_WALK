from .statistics import mean_joints, var_joints, bodies, steps

mirror_or_synchronize = mean_joints.mirror_or_synchronize

rew_mean_zero = mean_joints.rew_mean_zero           # L1 penalty for deviation from desired joint positions.
rew_mean_symmetry = mean_joints.rew_mean_symmetry        # Norm penalty for joint positions.
rew_mean_step_symmetry = mean_joints.rew_mean_step_symmetry
rew_mean_constraint = mean_joints.rew_mean_constraint

rew_variance_zero = var_joints.rew_variance_zero
rew_variance_symmetry = var_joints.rew_variance_symmetry
rew_variance_constraint = var_joints.rew_variance_constraint

rew_bodies_symmetry = bodies.rew_bodies_symmetry

rew_feetpose_symmetry = steps.rew_feetpose_symmetry
rew_bodiespose_symmetry = steps.rew_bodiespose_symmetry
rew_jointspose_symmetry = steps.rew_jointspose_symmetry
rew_times_symmetry = steps.rew_times_symmetry

from .pbrs import pbrs_statistics

pbrs_mean_zero = pbrs_statistics.mean_zero           # L1 penalty for deviation from desired joint positions.
pbrs_mean_symmetry = pbrs_statistics.mean_symmetry        # Norm penalty for joint positions.
pbrs_mean_step_symmetry = pbrs_statistics.mean_step_symmetry
pbrs_mean_constraint = pbrs_statistics.mean_constraint

pbrs_variance_zero = pbrs_statistics.variance_zero
pbrs_variance_symmetry = pbrs_statistics.variance_symmetry
pbrs_variance_constraint = pbrs_statistics.variance_constraint
