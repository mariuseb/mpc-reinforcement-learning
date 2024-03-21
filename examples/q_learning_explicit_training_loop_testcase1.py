"""Reproduces the first numerical example in [1]

References
----------
[1] Gros, S. and Zanon, M., 2019. Data-driven economic NMPC using reinforcement
    learning. IEEE Transactions on Automatic Control, 65(2), pp. 636-648.
"""

import logging
from typing import Any, Optional

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from ocp.mpc import MPC
from ocp.tests.utils import Bounds, get_boptest_config_path, get_opt_config_path
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.optim import NetwonMethod
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcrl.util.seeding import mk_seed
import pandas as pd
from project1_boptest_gym.examples.test_and_plot import plot_results
# boptest:
from project1_boptest_gym.boptestGymEnv import BoptestGymEnv
import os

# first, create classes for environment and mpc controller

"""
Set up boptest:
"""
bop_config_base = get_boptest_config_path()
boptest_cfg = os.path.join(bop_config_base, "ZEBLL_config.json")
mpc_cfg = os.path.join("1R1C_MPC.json")
url = 'http://bacssaas_boptest:5000'
# Use boptest gym env:
dt = 900
boptest = BoptestGymEnv(boptest_cfg,
                        name                  = "testcase1",
                        url                   = url,
                        actions               = ['oveAct_u'],
                        observations          = {'TRooAir_y':(280.,310.)}, 
                        #observations          = {'TRooAir_y':(280.,310.), 'TDryBul':(280.,310.)}, 
                        random_start_time     = False,
                        max_episode_length    = 24*3600,
                        #predictive_period     = 3600,
                        predictive_period     = dt*12,
                        warmup_period         = 0,
                        step_period           = dt)
boptest.reset()
"""
Set up MPC, simplest way
to obtain integrator:
"""
kwargs = {
    "x_nom": 30,
    "x_nom_b": 280.15,
    "u_nom": 10000,
    "r_nom": 30,
    "r_nom_b": 280.15,
    "p_nom": [1, 1],
    #"slack": Trues
    "slack": False
}
params = [1e-2,1E6]
_mpc = MPC(config=mpc_cfg,
          param_guess=params,
          **kwargs,
        )
"""
From this, we can get
properly scaled A, B
matrices.
"""
A = _mpc.get_Ad(900, p=params)
B = _mpc.get_Bd(900, p=params)
print(A)

_A = np.array([[-1/(1e-2*1e6)]])
_B = np.array([[1e-6],[1e-4]]).T
C = np.array([[1]])
D = np.array([[0]])
from scipy.signal import cont2discrete, lti, dlti, dstep
Ad, Bd, _, _, _ = cont2discrete((_A, _B, C, D), dt=900, method="zoh")

"""
Verify A using scipy:
"""

class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""
    horizon = 12
    discount_factor = 0.9

    def __init__(self, 
                 learnable_pars_init,
                 **kwargs
                 ) -> None:
        
        self.learnable_pars_init = learnable_pars_init
        N = self.horizon
        gamma = self.discount_factor
        w = np.array([1e6])
        nx, nu, nr = 1, 1, 1
        x_bnd = (np.asarray([[293.15]]), np.asarray([[296.15]]))  # bounds of state
        a_bnd = (-10000, 10000)  # bounds of control input
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        V0 = self.parameter("V0")
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        A = self.parameter("A", (nx, nx))
        B = self.parameter("B", (nx, nu))
        R = self.parameter("R", (nx, nr))

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)
        r = self.disturbance("r", nr) #, lb=280.15, ub=280.15)

        # dynamics
        self.set_dynamics(lambda x, u, r: A @ x + B @ u + R @ r + b, n_in=3, n_out=1)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)
        #self.constraint("x_lb", x_bnd[0] + x_lb, "<=", x[:, 1:])
        #self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub)
        # constraints on disturbances:
        #self.constraint("r_eq", r_bnd, "==", r[:, :])
        

        # objective
        A_init, B_init, R_init = self.learnable_pars_init["A"], \
                                 self.learnable_pars_init["B"], \
                                 self.learnable_pars_init["R"]
        #S = cs.DM(dlqr(A_init, B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0
            #+ cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 500,
                "sb": "yes",
                "print_level": 5,
            },
        }
        self.init_solver(opts, solver="ipopt")


learnable_pars_init = {
    "V0": np.asarray(0.0),
    "x_lb": np.asarray([0]),
    "x_ub": np.asarray([0]),
    "b": np.zeros(1), # nx
    "f": np.zeros(2), # nx + nu
    "A": A,
    "B": np.array([[B[0, 0]]]),
    "R": np.array([[B[0, 1]]]),
    #"r": data["Ta"].values[:12]
}

# now, let's create the instances of such classes and start the training
mpc = LinearMpc(learnable_pars_init)
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)

#env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=int(2e3)))
env = MonitorEpisodes(TimeLimit(boptest, max_episode_steps=int(2e3)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            discount_factor=mpc.discount_factor,
            update_strategy=100,
            optimizer=NetwonMethod(learning_rate=5e-2),
            hessian_type="approx",
            record_td_errors=True,
            remove_bounds_on_initial_action=True,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1000},
)
"""
Replace below with explicit training loop,
for one episode:
"""
#agent.train(env=env, episodes=1, seed=69)
episode = 0
raises = True
rng = np.random.default_rng(69)
agent.reset(rng)
agent._raises = raises
returns = np.zeros(1, float)
state, _ = env.reset(seed=mk_seed(rng))

#r = agent.train_one_episode(env, episode, state, True)
"""
Training one episode:
"""
timestep = 0
reward = 0
rewards = []
# get forecast:
data = boptest.forecast()
r = data["Ta"].values[:12].reshape((1,12))
#r = np.array([280.15])
action, solV = agent.state_value(
                                 state,
                                 False,
                                 {"r": r}
                                 )
if not solV.success:
    agent.on_mpc_failure(episode, None, solV.status, raises)
truncated = terminated = False
#while not (truncated or terminated):
for k in range(96):
    # compute Q(s,a)

    solQ = agent.action_value(state,
                              action,
                              {"r": r}
                              )

    action = pd.Series(data=float(action))
    
    # step the system with action computed at the previous iteration
    new_state, cost, truncated, terminated, _ = env.step(action)
    agent.on_env_step(env, episode, timestep)

    # get forecast:
    data = boptest.forecast()
    r = data["Ta"].values[:12].reshape((1,12))
    # compute V(s+) and store transition:
    new_action, solV = agent.state_value(new_state,
                                         False,
                                         {"r": r}
                                         )
    if not agent._try_store_experience(cost, solQ, solV):
        agent.on_mpc_failure(
            episode, timestep, f"{solQ.status} (Q); {solV.status} (V)", raises
        )

    # increase counters
    state = new_state
    action = new_action
    reward += float(cost)
    rewards.append(float(cost))
    timestep += 1
    agent.on_timestep_end(env, episode, timestep)

returns[episode] = reward

plot_results(
            boptest,
            rewards
            )

# plot the results
X = env.observations[0].squeeze().T
U = env.actions[0].squeeze()
R = env.rewards[0]
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[0])
axs[1].plot(X[1])
axs[2].plot(U)
for i in range(2):
    axs[0].axhline(env.x_bnd[i][0], color="r")
    axs[1].axhline(env.x_bnd[i][1], color="r")
    axs[2].axhline(env.a_bnd[i], color="r")
axs[0].set_ylabel("$s_1$")
axs[1].set_ylabel("$s_2$")
axs[2].set_ylabel("$a$")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(agent.td_errors, "o", markersize=1)
axs[1].semilogy(R, "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

_, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(np.asarray(agent.updates_history["b"]))
axs[0, 1].plot(
    np.stack(
        [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
    ),
)
axs[1, 0].plot(np.asarray(agent.updates_history["f"]))
axs[1, 1].plot(np.asarray(agent.updates_history["V0"]))
axs[2, 0].plot(np.asarray(agent.updates_history["A"]).reshape(-1, 4))
axs[2, 1].plot(np.asarray(agent.updates_history["B"]).squeeze())
axs[0, 0].set_ylabel("$b$")
axs[0, 1].set_ylabel("$x_1$")
axs[1, 0].set_ylabel("$f$")
axs[1, 1].set_ylabel("$V_0$")
axs[2, 0].set_ylabel("$A$")
axs[2, 1].set_ylabel("$B$")

plt.show()
print(mpc)
