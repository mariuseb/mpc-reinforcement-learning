from abc import ABC, abstractmethod
from typing import Collection, Dict, Generic, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc

from mpcrl.agents.agent import ActType, Agent, ObsType, SymType
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.util.types import GymEnvLike

ExpType = Tuple[
    ObsType, ActType, float, ObsType, Solution[SymType], Optional[Solution[SymType]]
]


class LearningAgent(Agent[SymType], ABC, Generic[SymType]):
    """Base class for a learning agent with MPC as policy provider where the main method
    `update`, which is called to update the learnable parameters of the MPC according to
    the underlying learning methodology (e.g., Bayesian Optimization, RL, etc.) is
    abstract and must be implemented by inheriting classes.

    Note: this class makes no assumptions on the learning methodology used to update the
    MPC's learnable parameters."""

    __slots__ = ("_experience", "_learnable_pars")

    def __init__(
        self,
        mpc: Mpc[SymType],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: Optional[str] = None,
    ) -> None:
        """Instantiates the learning agent.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place to create the approximations of the state function `V(s)`
            and action value function `Q(s,a)`, so it is recommended not to modify it
            further after initialization of the agent. Moreover, some parameter and
            constraint names will need to be created, so an error is thrown if these
            names are already in use in the mpc. These names are under the attributes
            `perturbation_parameter`, `action_parameter` and `action_constraint`.
        learnable_parameters : LearnableParametersDict
            A special dict containing the learnable parameters of the MPC, together with
            their bounds and values. This dict is complementary with `fixed_parameters`,
            which contains the MPC parameters that are not learnt by the agent.
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        exploration : ExplorationStrategy, optional
            Exploration strategy for inducing exploration in the MPC policy. By default
            `None`, in which case `NoExploration` is used in the fixed-MPC agent.
        experience : ExperienceReplay, optional
            The container for experience replay memory. If `None` is passed, then a
            memory wtih length 1 is created, i.e., it keeps only the latest memoery
            transition.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(mpc, fixed_parameters, exploration, warmstart, name)
        self._experience = (
            ExperienceReplay(maxlen=1) if experience is None else experience
        )
        self._learnable_pars = learnable_parameters

    @property
    def experience(self) -> ExperienceReplay[ExpType]:
        """Gets the experience replay memory of the agent."""
        return self._experience

    @property
    def learnable_parameters(self) -> LearnableParametersDict[SymType]:
        """Gets the parameters of the MPC that can be learnt by the agent."""
        return self._learnable_pars

    def store_experience(self, item: ExpType) -> None:
        """Stores the given item in the agent's memory for later experience replay. If
        the agent has no memory, then the method does nothing.

        Parameters
        ----------
        item : tuple of state-action-reward-new_state-V(s)-Q(s,a) (last can be `None`)
            Item to be stored in memory.
        """
        if self._experience is not None:
            self._experience.append(item)

    def on_training_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """Callback called at the beginning of the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        """

    def on_training_end(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """Callback called at the end of the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent has been trained on.
        """

    def on_episode_start(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """Callback called at the beginning of each episode in the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        """

    def on_episode_end(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """Callback called at the end of each episode in the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        """

    def on_env_step(
        self, env: GymEnvLike[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """Callback called after each `env.step`.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        timestep : int
            Time instant of the current training episode.
        """

    def on_udpate(self) -> None:
        """Callaback called after each `agent.update`. Use this callback for, e.g.,
        decaying exploration probabilities or learning rates."""

    @abstractmethod
    def update(self) -> Optional[str]:
        """Updates the learnable parameters of the MPC according to the agent's learning
        algorithm.

        Returns
        -------
        errormsg : str or None
            In case the update fails, an error message is returned to be raised as error
            or warning; otherwise, `None` is returned.
        """

    @abstractmethod
    def train(
        self,
        env: GymEnvLike[ObsType, ActType],
        episodes: int,
        update_frequency: int,
        seed: Union[None, int, Iterable[int]] = None,
        raises: bool = True,
    ) -> npt.NDArray[np.double]:
        """Train the agent on an environment.

        Parameters
        ----------
        env : GymEnvLike[ObsType, ActType]
            A gym-like environment where to train the agent in.
        episodes : int
            Number of training episodes.
        update_frequency : int
            The frequency of timesteps (i.e., every `env.step`) at which to perform
            updates to the learning parameters.
        seed : int or iterable of ints, optional
            Each env's seed for RNG.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Returns
        -------
        array of doubles
            The cumulative returns for each training episode.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on `raises`) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on `raises`) if the update fails.
        """
