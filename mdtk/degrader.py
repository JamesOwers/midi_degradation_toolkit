"""A degrader object can be used to easily degrade data points on the fly
according to some given parameters."""
import json
import logging

import numpy as np

import mdtk.degradations as degs


class Degrader:
    """A Degrader object can be used to easily degrade musical excerpts
    on the fly."""

    def __init__(
        self,
        seed=None,
        degradations=tuple(degs.DEGRADATIONS.keys()),
        degradation_dist=np.ones(len(degs.DEGRADATIONS)),
        clean_prop=1 / (len(degs.DEGRADATIONS) + 1),
        config=None,
    ):
        """
        Create a new degrader with the given parameters.

        Parameters
        ----------
        seed : int
            A random seed for numpy.

        degradations : list(string)
            A list of the names of the degradations to use (and in what order
            to label them).

        degradation_dist : list(float)
            A list of the probability of each degradation given in
            degradations. This list will be normalized to sum to 1.

        clean_prop : float
            The proportion of degrade calls that should return clean excerpts.

        config : string
            The path of a json config file (created by measure_errors.py).
            If given, degradations, degradation_dist, and clean_prop will
            all be overwritten by the values in the json file.
        """
        if seed is not None:
            np.random.seed(seed)

        # Load config
        if config is not None:
            with open(config, "r") as file:
                config = json.load(file)

            if "degradation_dist" in config:
                degradation_dist = np.array(config["degradation_dist"])
                degradations = list(degs.DEGRADATIONS.keys())
            if "clean_prop" in config:
                clean_prop = config["clean_prop"]

        # Check arg validity
        assert len(degradation_dist) == len(degradations), (
            "Given degradation_dist is not the same length as degradations:"
            f"\nlen({degradation_dist}) != len({degradations})"
        )
        assert (
            min(degradation_dist) >= 0
        ), "degradation_dist values must not be negative."
        assert (
            sum(degradation_dist) > 0
        ), "Some degradation_dist value must be positive."
        assert 0 <= clean_prop <= 1, "clean_prop must be between 0 and 1 (inclusive)."

        self.degradations = degradations
        self.degradation_dist = np.array(degradation_dist)
        self.clean_prop = clean_prop
        self.failed = np.zeros(len(degradations))

    def degrade(self, note_df):
        """
        Degrade the given note_df.

        Parameters
        ----------
        note_df : pd.DataFrame
            A note_df to degrade.

        Returns
        -------
        degraded_df : pd.DataFrame
            A degraded version of the given note_df. If self.clean_prop > 0,
            this can be a copy of the given note_df.

        deg_label : int
            The label of the degradation that was performed. 0 means none,
            and larger numbers mean the degradation
            "self.degradations[deg_label-1]" was performed.
        """
        if self.clean_prop > 0 and np.random.rand() <= self.clean_prop:
            return note_df.copy(), 0

        degraded_df = None
        this_deg_dist = self.degradation_dist.copy()
        this_failed = self.failed.copy()

        # First, sample from failed degradations
        while np.any(this_failed > 0):
            # Select a degradation proportional to how many have failed
            deg_index = np.random.choice(
                len(self.degradations), p=this_failed / np.sum(this_failed)
            )
            deg_fun = degs.DEGRADATIONS[self.degradations[deg_index]]

            # Try to degrade
            logging.disable(logging.WARNING)
            degraded_df = deg_fun(note_df)
            logging.disable(logging.NOTSET)

            # Check for success!
            if degraded_df is not None:
                self.failed[deg_index] -= 1
                return degraded_df, deg_index + 1

            # Degradation failed -- 0 out this deg and continue
            this_failed[deg_index] = 0

        # No degradations have remaining failures. Draw from standard dist
        while np.any(this_deg_dist > 0):
            # Select a degradation proportional to the distribution
            deg_index = np.random.choice(
                len(self.degradations), p=this_deg_dist / np.sum(this_deg_dist)
            )
            # This deg would have already failed in the above loop.
            # But we want to sample it and count it as another failure.
            if self.failed[deg_index] > 0:
                self.failed[deg_index] += 1
                continue
            deg_fun = degs.DEGRADATIONS[self.degradations[deg_index]]

            # Try to degrade
            logging.disable(logging.WARNING)
            degraded_df = deg_fun(note_df)
            logging.disable(logging.NOTSET)

            # Check for success!
            if degraded_df is not None:
                return degraded_df, deg_index + 1

            # Degradation failed -- add 1 to failure and continue
            self.failed[deg_index] += 1

        # Here, all degradations (with dist > 0) failed
        return note_df.copy(), 0
