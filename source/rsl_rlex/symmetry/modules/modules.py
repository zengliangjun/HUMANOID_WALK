

from __future__ import annotations
from tasks.unitreeg121.statistic_locations.mdps.mdps_symmetry21dofs import  HistorySymmetry
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories

def symmetry_subclass(parent_class):

    class Symmetry(parent_class):

        def __init__(
            self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            **kwargs,
        ):
            assert "symmetry_objs" in kwargs
            symmetry_objs: str = kwargs.pop("symmetry_objs")
            self.symmetry_objs = eval(symmetry_objs)()

            super(Symmetry, self).__init__(
                num_actor_obs - 1,
                num_critic_obs - 1,
                num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation,
                init_noise_std=init_noise_std,
                noise_std_type = noise_std_type,
                **kwargs,
            )

        def act(self, observations, **kwargs):
            symmetry_flags = observations[..., -1]
            symmetry_flags = symmetry_flags > 0.1

            observations = observations[..., :-1].clone()
            observations[symmetry_flags] = self.symmetry_objs.mirror_policy(observations[symmetry_flags])

            actions = super(Symmetry, self).act(observations, **kwargs)

            if "masks" in kwargs:
                symmetry_flags = unpad_trajectories(symmetry_flags[..., None], kwargs["masks"])[..., 0]

            actions[symmetry_flags] = self.symmetry_objs.mirror_action(actions[symmetry_flags])

            self.symmetry_flags = symmetry_flags.clone()
            return actions

        def get_actions_log_prob(self, actions):
            actions = actions.clone()
            actions[self.symmetry_flags] = self.symmetry_objs.mirror_action(actions[self.symmetry_flags])

            # work
            probs = super(Symmetry, self).get_actions_log_prob(actions)

            # probs[self.symmetry_flags] = self.symmetry_objs.mirror_action(probs[self.symmetry_flags])
            return probs

        def act_inference(self, observations):
            symmetry_flags = observations[..., -1]
            symmetry_flags = symmetry_flags > 0.1

            observations = observations[..., :-1].clone()
            observations[symmetry_flags] = self.symmetry_objs.mirror_policy(observations[symmetry_flags])

            actions = super(Symmetry, self).act_inference(observations)

            actions[self.symmetry_flags] = self.symmetry_objs.mirror_action(actions[self.symmetry_flags])

            return actions

        def evaluate(self, critic_observations, **kwargs):
            symmetry_flags = critic_observations[..., -1]
            symmetry_flags = symmetry_flags > 0.1

            observations = critic_observations[..., :-1].clone()
            observations[symmetry_flags] = self.symmetry_objs.mirror_critic(observations[symmetry_flags])

            return super(Symmetry, self).evaluate(observations, **kwargs)

    return Symmetry


from rsl_rl.modules import actor_critic, actor_critic_recurrent

SymmetryActorCritic = symmetry_subclass(actor_critic.ActorCritic)
SymmetryActorCriticRecurrent = symmetry_subclass(actor_critic_recurrent.ActorCriticRecurrent)




