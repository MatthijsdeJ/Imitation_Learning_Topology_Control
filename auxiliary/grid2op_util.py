#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util file for Grid2Op functionality.

@author: Matthijs de Jong
"""
# Standard library imports
from typing import Sequence, Tuple, List, Optional, Callable, Dict
import math

# Third party imports
import numpy as np
import grid2op
import torch
from grid2op.dtypes import dt_int

# Project imports
from auxiliary.config import get_config
import auxiliary.util as util


def tv_groupby_subst(tv: Sequence, sub_info: Sequence[int]) -> \
        List[Sequence]:
    """
    Group a sequence the shape of the topology vector by the substations.

    Parameters
    ----------
    tv : Sequence
        Sequence the shape of the topology vector.
    sub_info : Sequence[int]
        Sequence with elements containing the number of object connected to each substation.

    Returns
    -------
    List[Sequence]
        List, each element corresponding to a Sequence of objects in tv that belong to a particular substation.
    """
    i = 0
    gs = []
    for ss in sub_info:
        gs.append(tv[i:i + ss])
        i += ss
    return gs


def select_single_substation_from_topovect(topo_vect: torch.Tensor,
                                           sub_info: torch.Tensor,
                                           f: Callable = torch.sum,
                                           select_nothing_condition: Callable = lambda tv: all(tv < 0.5)) \
        -> Tuple[torch.Tensor, Optional[int]]:
    """
    Given a topology vector, select the substation whose objects maximize some function. From this substation, the mask
    in the topology vector and the index are returned. If a certain condition is met, the function can also select
    no substation, returning a zeroed array and an index of None.

    Parameters
    ----------
    topo_vect : torch.Tensor
        The vector based on which select a substation.
    sub_info : torch.Tensor
        Vector describing the number of objects per substation. Used to group topo_vect into objects of separate
        substations.
    f : Callable
        Function, based on the argmax of which the substation is selected.
    select_nothing_condition : Callable
        A condition on the topo_vect, which, if it holds true, will select no substation.

    Returns
    -------
    torch.Tensor
        The mask of the selected substation (one at the substation, zero everywhere else). Fully zeroed if 
        select_nothing_condition evaluates to true.
    Optional[int]
        Index of the substation. None if select_nothing_condition evaluates to true.
    """
    assert len(topo_vect) == sum(sub_info), "Length of topo vect should correspond to the sum of the " \
                                            "substation objects."

    if select_nothing_condition(topo_vect):
        return torch.zeros_like(topo_vect), None

    topo_vect_grouped = tv_groupby_subst(topo_vect, sub_info)
    selected_substation_idx = util.argmax_f(topo_vect_grouped, f)
    selected_substation_mask = torch.cat([(torch.ones_like(sub)
                                           if i == selected_substation_idx
                                           else torch.zeros_like(sub))
                                          for i, sub in enumerate(topo_vect_grouped)]).bool()

    return selected_substation_mask, selected_substation_idx


def init_env() -> grid2op.Environment.Environment:
    """
    Prepares the Grid2Op environment from a dictionary containing configuration setting.

    Returns
    -------
    env : TYPE
        The Grid2Op environment.
    """
    config = get_config()
    data_path = config['paths']['rte_case14_realistic']
    scenario_path = data_path + 'chronics/'

    # Required for topology reversal
    param = grid2op.Parameters.Parameters()
    param.MAX_SUB_CHANGED = 14

    env = grid2op.make(dataset=data_path,
                       chronics_path=scenario_path,
                       gamerules_class=grid2op.Rules.DefaultRules,
                       param=param,
                       test=True)

    # for reproducible experiments
    env.seed(config['simulation']['seed'])

    # Set custom thermal limits
    thermal_limits = config['rte_case14_realistic']['thermal_limits']
    env.set_thermal_limit(thermal_limits)

    return env


def ts_to_day(ts: int, ts_in_day: int) -> int:
    """
    Calculate what day (as a number) a timestep is in.

    Parameters
    ----------
    ts : int
        The timestep.
    ts_in_day : int
        The number of timesteps in a day.

    Returns
    -------
    int
        The day.
    """
    return math.floor(ts / ts_in_day)


def skip_to_next_day(env: grid2op.Environment.Environment, ts_in_day: int, chronic_id: int, disable_line: int):
    """
    Skip the environment to the next day.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The environment to fast-forward to the next day in.
    ts_in_day : int
        The number of timesteps in a day.
    chronic_id : int
        The current chronic id.
    disable_line : int
        The index of the line to be disabled.

    Raises
    -------
    grid2op.Exception.DivergingPowerFlowException
    """
    # Reset environment
    ts_next_day = ts_in_day * (1 + ts_to_day(env.nb_time_step, ts_in_day))
    env.set_id(chronic_id)
    env.reset()

    # Fast-forward to the day, disable lines if necessary
    env.fast_forward_chronics(ts_next_day - 1)
    if disable_line != -1:
        env_step_raise_exception(env, env.action_space({"set_line_status": (disable_line, -1)}))
    else:
        env_step_raise_exception(env, env.action_space())


def env_step_raise_exception(env: grid2op.Environment.Environment, action: grid2op.Action.BaseAction) \
        -> grid2op.Observation.BaseObservation:
    """
    Performs a step in the grid2op environment. Raises exceptions if they occur in the grid.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The grid2op environment.
    action : grid2op.Action.BaseAction
        The action to take in the former environment.

    Raises
    -------
    ExceptionGroup

    Returns
    -------
    obs : grid2op.Observation.CompleteObservation
        The observation resulting from the action.
    """
    obs, _, _, info = env.step(action)

    if len(info['exception']) > 1:
        raise ExceptionGroup('Exceptions', info['exception'])
    elif len(info['exception']) == 1:
        raise info['exception'][0]

    return obs
