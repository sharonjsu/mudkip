"""
Class to handle dataframes
"""

import warnings
import numpy as np
import pandas as pd
from datajoint import AndList
import puffbird as pb
from dreye.io import read_json
from dreye.core.photoreceptor import Photoreceptor
from dreye.core.signal_container import SignalsContainer
import ast
from sklearn.cluster import KMeans
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool
import uuid

from axolotl.schema import ffleds, imaging, subjects, color
from axolotl.utils.rounding import signif
from axolotl.utils.ffleds_df_helpers import _dataframe_led_descriptors
from axolotl.utils.dataframe_manipulation import parallel_concat, write_file


# TODO round time up to ms and use integers
LED_PRECISION = 3  # to the third decimal - convert to integer for better handling
TIME = 'time'
R = 'psth'
AMP = 'amp'
NORM = 'norm'
MNAME = 'measurement_name'
MSPECTRA = 'measured_spectra'
CELL_COLS = [
    'recording_id', 'cell_id', 'roi',
    # cell specific
    'genotype_id', 'subject_id',
    'measurement_name', 'opsin_set_name', 'pr_type',
    'pr_params',
    'stimulus_id'
]
STIM_SUBSET_COLS = ['led_mixer_name', 'stim_category']
# TODO get from database instead of hard-coded
OPSINS = ['rh1', 'rh3', 'rh4', 'rh5', 'rh6']
LEDS = ['duv', 'uv', 'violet', 'rblue', 'lime', 'orange']
LMAP = {
    "duv": "magenta",
    "uv": "purple",
    "violet": "blueviolet",
    "rblue": "navy",
    "lime": "springgreen",
    "orange": "orange"
}
OMAP = {
    "rh1": "gray",
    "rh3": "magenta",
    "rh4": "purple",
    "rh5": "blue",
    "rh6": "green"
}
PR_MODEL_KEY = {
    'opsin_set_name': 'published_sharkey2020_sta1993',
    'pr_type': 'LogPhotoreceptor',
    'pr_params': 'noiseless'
}

TABLE_INFO = [
    {
        'table': ffleds.LedMixerStimulus,
        'proj_args': [
            'opsin_set_name', 'pr_type',
            'pr_params', 'measurement_name'
        ]
    },
    {
        'table': ffleds.LedStimulusInformation,
        'proj_args': [
            'background_name',
            'background_intensity'
        ]
    },
    {'table': imaging.TwoPhotonRecording, 'proj_args': ['subject_id', 'recording_solution', 'neuron_section', 'brain_area', 'recording_time']},
    {'table': subjects.FlySubject, 'proj_args': ['genotype_id', 'sex', 'age', 'rearing_method', 'experimenter']},
]


def join_tables(keys, *table_dicts):
    """
    Join multiple tables using keys and projection arguments.
    
    Parameters:
    - keys: A dictionary specifying the keys for joining the tables.
    - table_dicts: Variable number of dictionaries specifying the tables and projection arguments.
    
    Returns:
    - joined: The joined table.
    """
    joined = None
    for table_dict in table_dicts:
        table = (table_dict['table'] & keys)
        if 'proj_args' in table_dict or 'proj_kwargs' in table_dict:
            table = table.proj(
                *table_dict.get('proj_args', []),
                **table_dict.get('proj_kwargs', {})
            )
        if joined is None:
            joined = table
        else:
            joined = joined * table
    return joined


class Ffleds:

    def __init__(
        self, 
        pr_model=None,
        led_round=LED_PRECISION,
        opsins=OPSINS, 
        leds=LEDS, 
        lmap=LMAP, 
        omap=OMAP
    ):
        """
        Initialize the Ffleds class.
        
        Parameters:
        - pr_model: The photoreceptor model.
        - led_round: The precision for rounding LED values.
        - opsins: List of opsins.
        - leds: List of LEDs.
        - lmap: LED color mapping.
        - omap: Opsin color mapping.
        """
        self.pr_model = pr_model
        self.led_round = led_round
        self.opsins = opsins
        self.leds = leds
        self.lmap = lmap
        self.omap = omap
        self.load_pr_model(pr_model)

    @property
    def rel_leds(self):
        """
        Get the list of relative LED names.
        """
        return [f'rel_{led}' for led in self.leds]

    @property
    def fitted_rel_leds(self):
        """
        Get the list of fitted relative LED names.
        """
        return [f'fitted_{led}' for led in self.rel_leds]

    @property
    def fitted_leds(self):
        """
        Get the list of fitted LED names.
        """
        return [f'fitted_{led}' for led in self.leds]

    @property
    def bg_leds(self):
        """
        Get the list of background LED names.
        """
        return [f'bg_{led}' for led in self.leds]

    @property
    def c_leds(self):
        """
        Get the list of contrast LED names.
        """
        return [f'c_{led}' for led in self.leds]

    @property
    def weber_leds(self):
        """
        Get the list of Weber contrast LED names.
        """
        return [f'weber_{led}' for led in self.leds]

    @property
    def fitted_opsins(self):
        """
        Get the list of fitted opsins.
        """
        return [f'fitted_{opsin}' for opsin in self.opsins]

    @property
    def q_opsins(self):
        """
        Get the list of quenched opsins.
        """
        return [f'q_{opsin}' for opsin in self.opsins]

    @property
    def fitted_q_opsins(self):
        """
        Get the list of fitted quenched opsins.
        """
        return [f'fitted_{opsin}' for opsin in self.q_opsins]

    def load_pr_model(self, pr_model=None):
        """
        Load the photoreceptor model.
        
        Parameters:
        - pr_model: The photoreceptor model.
        """
        if pr_model is None:
            self.pr_model = (
                color.PhotoreceptorModel
                & PR_MODEL_KEY
            ).fetch1('pr_model')
        elif isinstance(pr_model, dict):
            self.pr_model = (
                color.PhotoreceptorModel
                & pr_model
            ).fetch1('pr_model')
        elif isinstance(pr_model, Photoreceptor):
            self.pr_model = pr_model
        else:
            raise TypeError(f"`pr_model` is of type `{type(pr_model)}`.")

        return self.pr_model

    def calculate_amplitude(
        self, df,
        amp_interval=(0.4, 0.5),
        baseline_interval=(-0.2, -0.1),
        groupby=CELL_COLS+STIM_SUBSET_COLS,
        keepcols=None,
        rcol=R,
        tcol=TIME,
        ampcol=AMP,
        aggfunc='mean',
        comparefunc=lambda x,y: x-y, 
        aggfunc_noise='var', 
        comparefunc_noise=lambda x,y: x+y,
        _always_keep=None
    ):
        """
        Calculate the amplitude for each PSTH.
    
        Parameters:
        psth (numpy.ndarray): Array of PSTH values.
        
        Returns:
        numpy.ndarray: Array of amplitudes.
        """
        # get index pivot points
        groupby = groupby + self.leds + self.bg_leds + ['delay']

        groupby = list(set(groupby) & set(df.columns))
        assert groupby

        # pivot columns
        pdf = pd.pivot_table(
            df,
            rcol,
            groupby,
            tcol
        )

        def agg_helper(aggfunc, comparefunc):
            # calculate amplitude
            try:
                baseline = pdf.loc[:, slice(*baseline_interval)].agg(aggfunc, axis=1)
                amp = pdf.loc[:, slice(*amp_interval)].agg(aggfunc, axis=1)
            except KeyError:
                t = pdf.columns.to_numpy()
                baseline = pdf.iloc[:, _interval_helper(t, baseline_interval)].agg(aggfunc, axis=1)
                amp = pdf.iloc[:, _interval_helper(t, amp_interval)].agg(aggfunc, axis=1)
            amp = comparefunc(amp, baseline)
            amp.name = ampcol
            amp = amp.reset_index()
            return amp

        amp = agg_helper(aggfunc, comparefunc)
        noise = agg_helper(aggfunc_noise, comparefunc_noise)
        amp[ampcol+'_snr'] = amp[ampcol]**2 / noise[ampcol]  # mean ** 2 / var

        # add other columns
        if _always_keep is None:
            _always_keep = self.leds + self.bg_leds + self.rel_leds + self.led_measures
        if keepcols is None:
            keepcols = _always_keep
        else:
            keepcols = _always_keep + keepcols
        # just those in df
        keepcols = list((set(keepcols) & set(df.columns)) - set(groupby))
        if keepcols:
            # just choose first value
            to_merge = df.groupby(groupby)[keepcols].first().reset_index()
            amp = pd.merge(amp, to_merge)

        return amp

    def cluster_labels(
        self,
        df,
        featurecols=None,
        samplecols=['recording_id', 'cell_id'],
        rcol=AMP,
        cluster_model=lambda x: KMeans(3).fit(x).labels_,
        decomp_model=None,
        labels_attr='labels_',
        labels_name='labels',
        groupby=['genotype_id', 'bg', 'background_name']
    ):
        """
        Assign clusters
        """
        if featurecols is None:
            featurecols = self.bg_leds + self.rel_leds

        assert labels_name not in df.columns

        if groupby is not None:
            dfnew = pd.DataFrame()
            for _, grouped in df.groupby(groupby):
                df_ = self.cluster_labels(
                    grouped,
                    featurecols=featurecols,
                    samplecols=samplecols,
                    rcol=rcol,
                    cluster_model=cluster_model,
                    decomp_model=decomp_model,
                    labels_attr=labels_attr,
                    labels_name=labels_name,
                    groupby=None
                )
                dfnew = dfnew.append(df_, ignore_index=True)
            return dfnew

        Xdf = pd.pivot_table(
            df,
            rcol,
            samplecols,
            featurecols
        )
        X = Xdf.to_numpy()

        assert np.all(np.isfinite(X)), "Nans in the pivoted X array, change `featurecols` or `samplecols`"

        if decomp_model is None:
            Xt = X
        else:
            if callable(decomp_model):
                Xt = decomp_model(X)
            elif hasattr(decomp_model, 'fit') and hasattr(decomp_model, 'transform'):
                decomp_model.fit(X)
                Xt = decomp_model.transform(X)
            else:
                raise TypeError("`decomp_model` must be callable or sci-kit learn transformer.")

        if callable(cluster_model):
            labels = cluster_model(Xt)
        elif hasattr(cluster_model, 'fit') and labels_attr is not None:
            labels = getattr(cluster_model.fit(Xt), labels_attr)
        elif hasattr(cluster_model, 'fit') and hasattr(cluster_model, 'predict'):
            labels = cluster_model.fit(Xt).predict(Xt)
        else:
            raise TypeError("`cluster_model` must be callable or sci-kit learn type estimator.")

        labels = pd.Series(
            labels, index=Xdf.index,
            name=labels_name
        ).reset_index()

        return pd.merge(df, labels, on=samplecols)

    def calc_snr(
        self,
        df,
        rcol=AMP,
        groupby=None,
        cellcols=CELL_COLS,
        stim_category='neural_calibration',
    ):
        df = df[df['stim_category'] == stim_category]

        groupby = (CELL_COLS + self.leds if groupby is None else groupby)
        groupby = list((set(groupby) | set(cellcols)) & set(df.columns))

        df = df.groupby(groupby).agg(
            signal=pd.NamedAgg(column=rcol, aggfunc=lambda x: np.mean(x ** 2)),
            noise=pd.NamedAgg(column=rcol, aggfunc=lambda x: np.var(x, ddof=1)),
            norm=pd.NamedAgg(column=rcol, aggfunc='mean')
        )
        df = df.reset_index()
        df['snr'] = df['signal'] / df['noise']
        return df.groupby(
            cellcols
        )[['snr', 'signal', 'noise', 'norm']].mean().reset_index()

    def normalize(
        self, df,
        rcol=AMP,
        aggcols=None,
        meancols=None,
        aggfunc=None,
        aggkws=None,
        ncols=None,
        nfunc=None,
        newcol=NORM
    ):
        """
        normalize dataframe along groups
        """

        assert newcol not in df.columns, f"Column `{newcol}` already in dataframe columns, change `newcol` argument."
        aggfunc = (np.linalg.norm if aggfunc is None else aggfunc)
        aggkws = ({} if aggkws is None else aggkws)
        aggcols = (CELL_COLS if aggcols is None else aggcols)
        ncols = ([rcol] if ncols is None else ncols)
        nfunc = (np.divide if nfunc is None else nfunc)

        if meancols is None:
            dfm = df.set_index(aggcols)[rcol]
        else:
            meancols = list(set(meancols) | set(aggcols))
            dfm = df.groupby(meancols)[rcol].mean()

        norm = dfm.groupby(aggcols).aggregate(aggfunc, **aggkws)
        norm.name = newcol

        df = pd.merge(df, norm.reset_index(), on=aggcols)
        for ncol in ncols:
            df[ncol] = nfunc(df[ncol].to_numpy(), df[newcol].to_numpy())
        return df

    def hierarchical_bootstrap(self, df, estimator=np.mean, ci=95, n=1000):
        """
        bootstrap a dataframe
        """
        raise NotImplementedError('hierarchical_bootstrap')

    @property
    def led_measures(self):
        return ['bg', 'single_led', 'rel_type', 'abs_sum', 'rel_abs_sum',
                'background_name', 'led_combos', 'led_number',
                'unidirectional', 'diff']

    def round_dataset(
        self, df, digit=3, cols=None, add=True
    ):
        default_cols = self.leds + self.bg_leds + self.rel_leds + ['diff', 'bg']
        if cols is None:
            cols = default_cols
        if add:
            cols = cols + default_cols
        df[cols] = signif(df[cols], digit)
        return self

    def filter_data_with_snr(
        self,
        df, 
        amp_col=AMP, 
        snr_col=AMP+'_snr', 
        identifier_cols=[
            'background_name', 
            'genotype_id', 
            'subject_id', 
            'recording_id', 
            'roi'
        ], 
        groupby_cols=None, 
        top_responses=20,
        snr_thresh=1, 
        return_X=False, 
        ignore_nans=False, 
        agg_method='min', 
    ):
        """
        columns: amp_snr, amp
        """
        if groupby_cols is None:
            groupby_cols = self.rel_leds

        if ignore_nans:
            snr_index = df.groupby(
                groupby_cols
            )[amp_col].mean().abs().sort_values(
                ascending=False
            ).index[:top_responses]
        else:
            snr_index = df.groupby(
                groupby_cols
            )[amp_col].agg(
                lambda x: (
                    np.mean(x)
                    if np.all(np.isfinite(x))
                    else np.nan
                )
            ).abs().sort_values(
                ascending=False, 
                na_position='last'
            ).index[:top_responses]

        Xsnr = pd.pivot_table(
            df, 
            snr_col, 
            groupby_cols, 
            identifier_cols
        )
        snr_lim = getattr(Xsnr.loc[snr_index], agg_method)() > snr_thresh

        if not np.any(snr_lim):
            warnings.warn("All recordings are too noisy", RuntimeWarning)

        if return_X:
            X = pd.pivot_table(
                df, 
                amp_col, 
                groupby_cols, 
                identifier_cols
            )
            return X.loc[:, snr_lim]
        else:
            snr_lim = snr_lim.reset_index()
            snr_lim = snr_lim[snr_lim[0]].drop(columns=0)
            return pd.merge(df, snr_lim, how='inner')

    def format_data(self, df):
        format_data(
            df, 
            self.opsins, self.fitted_opsins, 
            self.q_opsins, self.fitted_q_opsins, 
            self.leds, self.fitted_leds, 
            self.rel_leds, self.fitted_rel_leds, 
            self.bg_leds, self.c_leds, self.led_round
        )
        return self

    def motyxia_format_swls(self, data):
        assert set(['spectrum_id', 'luminant_multiple']).issubset(set(data.columns))
        try:
            # if string
            data.loc[~data['spectrum_id'].isnull(), 'spectrum_id'] = data.loc[~data['spectrum_id'].isnull(), 'spectrum_id'].apply(ast.literal_eval)
        except:
            pass
        swl_info = data.groupby(
            self.bg_leds + self.leds + ['luminant_multiple']
        )['spectrum_id'].first().apply(
            pd.Series
        ).fillna(0).rename(columns=lambda x: float(x.replace('peak', ''))).sort_index(1)
        swl_info.columns.name = 'wl_peak'

        swl_stacked = swl_info.stack()
        swl_stacked = swl_stacked[swl_stacked != 0]
        swl_stacked.name = 'wl_prop'
        swl_stacked = swl_stacked.reset_index()
        swl_stacked['swl'] = False
        swl_stacked.loc[swl_stacked['wl_prop'] == 1, 'swl'] = True

        mix_stacked = swl_stacked.loc[~swl_stacked['swl']]
        swl_stacked = swl_stacked.loc[swl_stacked['swl']]
        mix_stacked = mix_stacked.groupby(
            self.bg_leds + self.leds + ['luminant_multiple']
        ).apply(
            lambda x: (
                list(np.round(x['wl_peak'].to_numpy()[np.argsort(x['wl_peak'])], 0))
                +
                list(np.round(x['wl_prop'].to_numpy()[np.argsort(x['wl_peak'])], 2))
            )
        ).apply(pd.Series, index=['wl_peak', 'wl_peak2', 'wl_prop', 'wl_prop2'])
        mix_stacked['swl'] = False
        mix_stacked = mix_stacked.reset_index()
        stacked = pd.concat([mix_stacked, swl_stacked])
        return data.merge(stacked, validate='many_to_one')

    def bg_uniques(self, df):
        return df[['background_name', 'bg'] + self.bg_leds].drop_duplicates()


class FfledsDf(Ffleds):

    @property
    def table_info(self):
        table_info = [
            {'table': self.psth_table},
            {'table': self.stimuli_table, 'proj_args': ['stimulus_id']},
        ] + TABLE_INFO.copy()
        return table_info

    def __init__(
        self,
        *keys,
        stimuli_table=ffleds.DreyeStimuli,
        psth_table=ffleds.DreyePsths,
        raw_import='pv_import',
        motion_correct='rigid_pw_mc1_blur',
        extraction='watershedd_contrastive_interpolate',
        psths='interpolate_psth_v2',
        pr_model=None,
        led_round=LED_PRECISION,
        opsins=OPSINS, 
        leds=LEDS, 
        lmap=LMAP, 
        omap=OMAP
    ):
        super().__init__(
            pr_model=pr_model, 
            led_round=led_round, 
            opsins=opsins, 
            leds=leds, 
            lmap=lmap, omap=omap
        )

        self.populate_keys = {
            imaging.RawTwoPhotonData().settings_name: raw_import,
            imaging.MotionCorrectedData().settings_name: motion_correct,
            imaging.ExtractedData().settings_name: extraction,
            psth_table().settings_name: psths
        }
        self.keys = AndList(list(keys) + [self.populate_keys])
        self.stimuli_table = stimuli_table
        self.psth_table = psth_table
        self.raw_import = raw_import
        self.motion_correct = motion_correct
        self.extraction = extraction
        self.psths = psths

        self._uuid_table = None
        self._uuids = None
        self._subject_table = None
        self._subject_df = None
        self._table = None
        self._data = None
        self._neuralcal = None
        self._ledsubs = None
        self._exploration = None
        self._measurement_table = None
        self._measurement_df = None

    def populate(
        self,
        suppress_errors=False,
        verbose=True,
        skip=[]
    ):
        if 'raw' not in skip:
            imaging.RawTwoPhotonData.populate(
                self.raw_import,
                self.keys,
                suppress_errors=suppress_errors,
                verbose=verbose
            )

        if 'mc' not in skip:
            imaging.MotionCorrectedData.populate(
                self.motion_correct,
                self.keys,
                suppress_errors=suppress_errors,
                verbose=verbose
            )

        if 'extract' not in skip:
            imaging.ExtractedData.populate(
                self.extraction,
                self.keys,
                suppress_errors=suppress_errors,
                verbose=verbose
            )

        if 'stim' not in skip:
            self.stimuli_table.populate(self.keys, suppress_errors=suppress_errors)
            ffleds.LedStimulusInformation.populate(suppress_errors=suppress_errors)

        self.psth_table.populate(
            self.psths,
            self.keys,
            suppress_errors=suppress_errors,
            verbose=verbose
        )

    @property
    def uuid_table(self):
        if self._uuid_table is None:
            self._uuid_table = (
                ffleds.LedMixerStimulus()
                & self.table.proj()
            )
        return self._uuid_table

    @property
    def uuids(self):
        if self._uuids is None:
            self._uuids = self.uuid_table.proj().fetch('stimulus_id')
        return self._uuids

    @property
    def measurement_table(self):
        if self._measurement_table is None:
            self._measurement_table = (
                color.LedMeasurement
                & self.uuid_table.proj(MNAME)
            ).proj('measurement_file')
        return self._measurement_table

    @property
    def measurement_df(self):
        if self._measurement_df is None:
            df = self.measurement_table.fetch(format='frame').reset_index()
            mss = []
            for index, row in df.iterrows():
                ms = read_json(row['measurement_file'])
                ms = ms.measured_spectra
                mss.append(ms)
            df.loc[:, MSPECTRA] = pd.Series(mss)
            self._measurement_df = df
        return self._measurement_df

    @property
    def subject_table(self):
        if self._subject_table is None:
            self._subject_table = (
                (
                    imaging.TwoPhotonRecording
                    & self.keys
                ).proj('subject_id', 'recording_file_id')
                * (
                    subjects.FlySubject & self.keys
                ).proj('genotype_id', 'experimenter')
            )
        return self._subject_table

    @property
    def subject_df(self):
        if self._subject_df is None:
            self._subject_df = self.subject_table.fetch(format='frame').reset_index()
        return self._subject_df

    @property
    def table(self):
        if self._table is None:
            self._table = join_tables(self.keys, *self.table_info)
        return self._table

    @property
    def data(self):
        if self._data is None:
            # TODO improve with handling columns properly
            length = len(self.table)
            if length == 0:
                raise RuntimeError("No tables to fetch - check your restrictions and see if you populated the tables.")
            load = length / 30
            processes = min([32, cpu_count()])
            print(f"Loading dataset - Highest load approx. {load}GB.")
            files = process_map(
                process_df, 
                self.table, 
                args_iterable(
                    length, 
                    [
                        self.opsins, self.fitted_opsins, 
                        self.q_opsins, self.fitted_q_opsins, 
                        self.leds, self.fitted_leds, 
                        self.rel_leds, self.fitted_rel_leds, 
                        self.bg_leds, self.c_leds, self.led_round
                    ]
                ), 
                max_workers=processes, 
                chunksize=10
            )

            print("Concatenating dataset...")
            df = parallel_concat(files)
            self._data = df
        return self._data

    def __getattr__(self, name):
        if '_data' in vars(self) and hasattr(self._data, name):
            return getattr(self._data, name)
        raise AttributeError(
            f"{type(self).__name__} does not have attribute `{name}`."
        )

    @property
    def ledsubs(self):
        if self._ledsubs is None:
            self._ledsubs = self.data[
                self.data['stim_category'].isin(['LED_substitution'])
            ]
        return self._ledsubs

    @property
    def neuralcal(self):
        if self._neuralcal is None:
            self._neuralcal = self.data[
                self.data['stim_category'].isin(['neural_calibration'])
            ]
        return self._neuralcal

    @property
    def exploration(self):
        if self._exploration is None:
            self._exploration = self.data[
                self.data['stim_category'].isin([
                    'single_led_axis',
                    'mixture_led_axis',
                    'achromatic_led_axis'
                ])
            ]
        return self._exploration

    # def plot_exploration
    # def plot_ledsubs
    # def plot_neuralcal

    def get_spectra(self, df, ledcols=None, **kwargs):
        """
        """
        ledcols = (self.leds if ledcols is None else ledcols)
        assert MNAME in df.columns, '`measurement_name` is not in dataframe columns'

        container = []
        for index, row in self.measurement_df.iterrows():
            df_ = df[df[MNAME] == row[MNAME]]
            labels = df_.index
            measured_spectra = row[MSPECTRA]
            spectra = measured_spectra.ints_to_spectra(
                df_[ledcols].to_numpy(), **kwargs
            )
            spectra.labels = labels
            container.append(spectra)
        spectra = SignalsContainer(container).signals.loc[df.index]
        spectra.domain_axis = 1
        return spectra

    def bg_uniques(self, df=None):
        if df is None:
            df = self.data
        return super().bg_uniques(df)


def _interval_helper(t, interval):
    """
    Check if each value in array `t` falls within the specified interval.

    Parameters:
        t (numpy.ndarray or list): Array of values.
        interval (tuple): Interval represented as a tuple of (start, end).

    Returns:
        numpy.ndarray: Boolean array indicating whether each value falls within the interval.
    """
    return (t >= interval[0]) & (t <= interval[1])


def process_df(entry, args=None):
    """
    Process the DataFrame by dropping unnecessary columns and adding missing columns.

    Parameters:
        entry (dict or tuple): Dictionary or tuple containing the DataFrame and additional parameters.
        args (tuple or None): Additional parameters for format_data function. Default is None.

    Returns:
        str: Result of the write_file function.
    """
    if args is None:
        entry, args = entry
    df_ = entry.pop('psths')
    drop = list(
        set(df_.columns) 
        & set([
            '_after', '_start', 'repeat', 'stim_index', 
            'synced_delay', 
            'values_index_0', 'index', 'iter'
        ])
    )
    df_.drop(columns=drop, inplace=True)
    for k, v, in entry.items():
        if k in df_.columns:
            continue
        if k == 'psths':
            continue
        if k.endswith('populate_settings'):
            continue
        df_[k] = v

    format_data(df_, *args)
    return write_file(df_)

def format_data(df, opsins, fitted_opsins, q_opsins, fitted_q_opsins, leds, fitted_leds, rel_leds, fitted_rel_leds, bg_leds, c_leds, led_round):
    """
    Format the data in the DataFrame by adding/replacing columns and dropping unnecessary columns.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        opsins (list): List of opsins column names.
        fitted_opsins (list): List of fitted opsins column names.
        q_opsins (list): List of q opsins column names.
        fitted_q_opsins (list): List of fitted q opsins column names.
        leds (list): List of LED column names.
        fitted_leds (list): List of fitted LED column names.
        rel_leds (list): List of relative LED column names.
        fitted_rel_leds (list): List of fitted relative LED column names.
        bg_leds (list): List of background LED column names.
        c_leds (list): List of C LED column names.
        led_round (int): Number of digits to round LED values.

    Returns:
        None
    """
    # add/rename columns
    df['roi'] = df['recording_id'] * 1000 + df['cell_id']
    df[leds] = df[fitted_leds].to_numpy()

    # TODO use measurement_name in events to calculate new opsins values
    if set(fitted_opsins).issubset(set(df.columns)):
        # TODO df[[MNAME] + self.leds + self.bg_leds].drop_duplicates()
        df[opsins] = df[fitted_opsins]

    # drop columns inplace
    drop_cols = (
        fitted_leds
        + fitted_rel_leds
        + fitted_opsins
        + q_opsins
        + fitted_q_opsins
    )
    drop_cols = list(set(drop_cols) & set(df.columns))

    df.drop(
        columns=drop_cols,
        inplace=True
    )
    # remove other fitted columns
    fitted_cols = {
        col: col.replace('fitted_', '')
        for col in df.columns
        if col.startswith('fitted_')
        and col.replace('fitted_', '') in df.columns
    }
    df[list(fitted_cols.values())] = df[list(fitted_cols)]
    df.drop(columns=list(fitted_cols), inplace=True)

    _dataframe_led_descriptors(df, leds, bg_leds, rel_leds, c_leds, led_round)

def df_iterable(df):
    """
    Iterate over the DataFrame and yield each row.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Yields:
        pandas.Series: Each row in the DataFrame.
    """
    for _, row in df.reset_index().iterrows():
        yield row

def args_iterable(length, args):
    """
    Iterate over a given length and yield the provided arguments.

    Parameters:
        length (int): Length of the iteration.
        args (any): Arguments to be yielded.

    Yields:
        any: The provided arguments.
    """
    for _ in range(length):
        yield args