"""
Class to create psths
"""

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# from dreye.stimuli.variables import SYNCED_DELAY_KEY, DUR_KEY, PAUSE_KEY
#todo define these variables here; they're just strings


PSTH_COL = 'psth'
TIME_COL = 'time'


DUR_KEY = 'dur'
SYNCED_DUR_KEY = 'synced_dur'
PAUSE_KEY = 'pause'

class Psths:
    """
    Class to create a PSTH long-format dataframe

    Parameters
    ----------
    timestamps : numpy.ndarray
        Array with same length as `signal`
    signal : numpy.ndarray
        Array with same length as `timestamps`. (1D-array)
    events : pandas.DataFrame
        A dataframe where each row represents an event
    delay_key : str
        The column in `events` that corresponds to the start/alignement
        of the event.
    dur_key : str
        The column in `events` that corresponds to the length of the event.
    pause_key : str
        The column in `events` that corresponds to the length of time
        between the end of the event and a period of time where no
        other events occur.
    """

    def __init__(
        self,
        timestamps,
        signal,
        events,
        delay_key=SYNCED_DELAY_KEY,
        dur_key=None,
        pause_key=None,
        end_key=None,
    ):
        self.timestamps = timestamps
        self.signal = signal
        self.events = events
        self.delay_key = delay_key
        self.dur_key = dur_key
        self.pause_key = pause_key
        self.end_key = end_key

    def create_interpolate(
        self, before=0, after=0, interpolate_dt=0.1,
    ):
        """
        Create long PSTH dataframe and interpolate signal array.

        Parameters:
            before (float): Time before the start of each event to include in the interpolation. Default is 0.
            after (float): Time after the end of each event to include in the interpolation. Default is 0.
            interpolate_dt (float): Time step size for the interpolation. Default is 0.1.

        Returns:
            pandas.DataFrame: Long-format PSTH dataframe.
        """

        start = np.array(self.events[self.delay_key]) - before
        if self.dur_key is not None:
            after = after + np.array(self.events[self.dur_key])
        if self.pause_key is not None:
            after = after + np.array(self.events[self.pause_key])

        assert np.all(after > 0), (
            "Start times not smalled than end times for PSTH creation."
        )

        self.events['_start'] = start
        self.events['_after'] = after

        def create_helper(event):

            aligned = self.timestamps - event['_start']
            interpolator = interp1d(
                aligned, self.signal, bounds_error=False
            )
            t = np.arange(
                0, event['_after'] + interpolate_dt + before, interpolate_dt
            )
            psth = interpolator(t)
            t -= before

            df = pd.DataFrame(
                np.array([t, psth]).T,
                columns=[TIME_COL, PSTH_COL]
            )

            for k, v in event.items():
                # TODO non-hashables
                try:
                    df[k] = v
                except ValueError as e:
                    warnings.warn(
                        f"For value `{v}`, this error occured "
                        f"during assignment:\n{e}\n\n"
                        "Trying to assign differently!"
                    )
                    df[k] = [v] * len(df)

            return df

        psth_df = pd.DataFrame()
        for idx, event in self.events.iterrows():
            psth_df = psth_df.append(
                create_helper(event), ignore_index=True, sort=True
            )

        return psth_df


def psth_interpolate(
    timestamps,
    signal,
    events,
    before=0.5, after=0, interpolate_dt=0.1,
    delay_key=SYNCED_DELAY_KEY,
    dur_key=DUR_KEY,
    pause_key=PAUSE_KEY,
):
    """
    Create PSTH long-format dataframe and interpolate to a given time step size.

    Parameters:
        timestamps (numpy.ndarray): Array with the same length as `signal`.
        signal (numpy.ndarray): Array with the same length as `timestamps` (1D array).
        events (pandas.DataFrame): A dataframe where each row represents an event.
        before (float): Time before the start of each event to include in the interpolation. Default is 0.5.
        after (float): Time after the end of each event to include in the interpolation. Default is 0.
        interpolate_dt (float): Time step size for the interpolation. Default is 0.1.
        delay_key (str): The column in `events` that corresponds to the start/alignment of the event. Default is SYNCED_DELAY_KEY.
        dur_key (str): The column in `events` that corresponds to the length of the event. Default is DUR_KEY.
        pause_key (str): The column in `events` that corresponds to the length of time between the end of the event and a period of time where no other events occur. Default is PAUSE_KEY.

    Returns:
        pandas.DataFrame: Long-format PSTH dataframe.
    """

    p = Psths(
        timestamps, signal, events,
        delay_key=delay_key,
        dur_key=dur_key,
        pause_key=pause_key
    )
    df = p.create_interpolate(
        before=before, after=after,
        interpolate_dt=interpolate_dt
    )
    return df
