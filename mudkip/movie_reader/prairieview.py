"""functions for reading PrairieView imaging data
"""
from datetime import timedelta
from glob import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from numpy.lib.format import open_memmap
from warnings import warn
from PIL import Image
from joblib import Parallel, delayed

from mudkip.err import AxolotlError, XmlStructureError

REC_TYPES_DICT = {
    'tseries': 'TSeries',
    'tseries_2channel': 'TSeries',
    'zseries': 'ZSeries',
    'singleimage': 'SingleImage'
}
REC_TYPES = ['TSeries', 'ZSeries', 'SingleImage']
V_NAMES = {
    'VoltageRecording': 'Vin',
    'VoltageOutput': 'Vout'
}
PV_POSITIONS = [
    'positionCurrentXAxis0',
    'positionCurrentYAxis0',
    'positionCurrentZAxis0'
]
PV_MPP = [
    'micronsPerPixelXAxis',
    'micronsPerPixelYAxis'
]
# TODO make prairieView reader class
# TODO infer formatting of tiff loading from frames dataframe in read_xml
# TODO create own tiff reading function which parallelizes process
# TODO make prairieview class


def _get_column(df, col):
    if isinstance(col, str):
        assert col in df.columns, f"Column `{col}` not in dataframe with {list(df.columns)}"
        return df[col]
    elif isinstance(col, (list, tuple)):
        assert len(set(col) & set(df.columns)) == 1, (
            f"No or more than one column exists in set {col} "
            f"that also exists in dataframe with {list(df.columns)}"
        )
        for icol in col:
            if icol in df.columns:
                return df[icol]
    else:
        raise TypeError(
            f"`col` is wrong type (str or list/tuple): {type(col)}"
        )


def _memmap_tiffs(
    filenames,
    rec_folder_path,
    n_jobs=None,
    memmap_file='_movie_memmap.npy'
):
    """
    From single tiff files create a memmap numpy file.
    """
    # make listlike
    filenames = np.array(filenames)

    # Load movie
    tiff_filepath = os.path.join(rec_folder_path, filenames[0])
    frame = np.array(Image.open(tiff_filepath))
    shape = filenames.shape + frame.shape
    dtype = frame.dtype

    filepath = os.path.join(rec_folder_path, memmap_file)
    movie = open_memmap(
        filepath, mode='w+', dtype=dtype, shape=shape
    )
    movie[0] = frame

    def assign_helper(idx, filename, movie):
        tiff_filepath = os.path.join(rec_folder_path, filename)
        frame = np.array(Image.open(tiff_filepath))
        movie[idx + 1] = frame

    if n_jobs is None:
        for idx, filename in enumerate(filenames[1:]):
            assign_helper(idx, filename, movie)

    else:
        Parallel(n_jobs=n_jobs)(
            delayed(assign_helper)(idx, filename, movie)
            for idx, filename in enumerate(filenames[1:])
        )
    movie.flush()
    ###
    return movie


def get_prairieview_data_for_database(
    datafolders,
    recording_type,
    file_extension,
    date,
    days_before=0,
    date_format='%m%d%Y',
    ext_fill=3,
    file_format='{recording_type}-{date}-????-{file_extension}',
    errors='raise',
    n_jobs=None,
    channel=None,
    # trigger related things
    trigger_time_column='Time(ms)',
    trigger_time_conversion=10**-3,
    trigger_column='trigger',
    add_absolute_trigger_time=True
):
    """
    Get raw two photon data from prairieView recording to insert
    into the imaging.RawTwoPhotonData table in loris/datajoint.

    This is for 2D recordings.
    
    Parameters
    ----------
    datafolders : str or list
        Path to 2p raw dta
    recording_type : str 
        Type of recording (from imaging.TwoPhotonRecording)
        Will also need a mapping to prairiveiw type 
    file_extension : str
        The ending of your PrairieView file recording number.
    date: datetime
        Date of recording
    days_before: int
        Days before date recording to find prairieview files
    date_format : str format
        Datetime format of prairieiview saves
    ext_fill : int
        minimium length of file extension id. 
    file_format : str
    

    """

    rec_folder_path = find_recording(
        datafolders=datafolders,
        recording_type=recording_type,
        file_extension=file_extension,
        date=date,
        days_before=days_before,
        date_format=date_format,
        ext_fill=ext_fill,
        file_format=file_format,
        errors=errors
    )
    # split path
    imaging_folder_path, rec_folder_name = os.path.split(rec_folder_path)
    # get xml data
    xmlpath = os.path.join(rec_folder_path, rec_folder_name + '.xml')
    pv_settings, sequences = read_xml(xmlpath, channel)

    if 'absoluteTimeVin' in pv_settings and 'dataFileVin' in pv_settings:
        vin_csv = os.path.join(rec_folder_path, pv_settings['dataFileVin'])
        csvdata = pd.read_csv(vin_csv).rename(columns=lambda x: str(x).strip())
        vin_data = csvdata

        # this is hard-coded
        tt = np.array(
            # get column allows to define multiple column names that
            # may be acceptable if the previous ones did not work.
            _get_column(vin_data, trigger_time_column) * trigger_time_conversion
        )
        if add_absolute_trigger_time:
            tt += pv_settings['absoluteTimeVin']
        trigger = np.array(_get_column(vin_data, trigger_column))
    else:
        raise AxolotlError('No voltage input exists, '
                           'recording will not be able to be '
                           'synced with stimuli.')

    movie = _memmap_tiffs(
        sequences['filename'], rec_folder_path, n_jobs=n_jobs
    )

    #
    data2insert = dict(
        rate=1 / pv_settings['framePeriod'],
        timestamps=np.array(sequences['absoluteTime']),
        movie=np.array(movie),
        tiff_folder_location=rec_folder_path,
        imaging_offset=sequences['absoluteTime'].iloc[0],
        trigger=trigger,
        trigger_timestamps=tt,
        field_of_view=np.array([
            pv_settings['pixelsPerLine'] * pv_settings[PV_MPP[0]],
            pv_settings['linesPerFrame'] * pv_settings[PV_MPP[1]]
        ]),
        pmt_gain=pv_settings['pmtGain1'],
        scan_line_rate=1 / pv_settings['scanLinePeriod'],
        dimension=np.array([
            pv_settings['pixelsPerLine'],
            pv_settings['linesPerFrame'],
        ]),
        location=np.array([
            pv_settings[pos] for pos in PV_POSITIONS
        ]),
        laser_power=pv_settings['laserPower0'],
        laser_wavelength=pv_settings['laserWavelength0'],
        dwell_time=pv_settings['dwellTime'],
        microns_per_pixel=np.array([
            pv_settings[mpp] for mpp in PV_MPP
        ]),
        metadata_collection={
            'pv_settings': pv_settings,
            'pv_sequence': sequences,
            'pv_vin': vin_data
        }
    )
    return data2insert


def find_recording(
    datafolders, recording_type, file_extension, date,
    days_before=0, date_format='%m%d%Y', ext_fill=3,
    file_format='{recording_type}-{date}-????-{file_extension}',
    errors='raise'
):
    """
    Find the correct PraireView folder corresponding
    to a recording.

    Parameters
    ----------
    datafolders : str or iterable
        Path(s) to data storage folder
    recording_type : str
        Type of recording (TSeries, ZSeries, SingleImage)
    file_extension : str or int
        The extension to the file
    date : datetime
        date of the recording
    days_before : int
        The days before the date to look for the recording
    file_format : str
        Formating of recording folder.

    Returns
    -------
    recording_folder : str
        Name recording folder.
    """
    recording_type = REC_TYPES_DICT.get(recording_type, recording_type)
    if recording_type not in REC_TYPES:
        raise AxolotlError(f"Unknown recording type: {recording_type}.")

    if isinstance(datafolders, str):
        datafolders = [datafolders]

    check_dates = [
        (date - timedelta(days=d)).strftime(date_format)
        for d in range(days_before + 1)
    ]
    for datafolder in datafolders:
        matches_found = []
        for check_date in check_dates:
            matches_found.extend(glob(os.path.join(
                datafolder,
                file_format.format(
                    recording_type=recording_type,
                    date=check_date,
                    file_extension=str(file_extension).zfill(ext_fill)
                )
            )))

        if len(matches_found) == 1:
            break

    if len(matches_found) == 0:
        if errors == 'warn':
            warn("Could not find folder matching "
                 f"{file_extension} on the {date}.")
            return None
        elif errors == 'ignore':
            return None
        else:
            raise NameError("no match found for file extension "
                            f"{file_extension} on the {date}.")
    elif len(matches_found) != 1:
        raise NameError("multiple matches found "
                        f"for recording: {matches_found}.")
    else:
        return matches_found[0]


# TODO make class
def read_xml(xmlpath, channel=None):
    """
    Read the metadata from the prairie view xml file.

    Returns
    -------
    pv_settings : dictionary
        General settings of prairie view.
    sequences : pandas.DataFrame
        Dataframe of each frame's metadata
    """
    def shard_helper(stateshard):
        # check if right tag
        if stateshard.tag != 'PVStateShard':
            raise XmlStructureError(f"{stateshard.tag}")
        # initialize settings dictionary
        settings = {}
        # iterate over shards
        for statevalue in list(stateshard):
            if statevalue.tag != 'PVStateValue':
                raise XmlStructureError(f"{statevalue.tag}")
            elif len(statevalue) == 0:
                attrib = statevalue.attrib
                settings[attrib['key']] = attrib['value']
                continue
            statekey = statevalue.attrib['key']
            for indexvalue in list(statevalue):
                if indexvalue.tag == 'IndexedValue':
                    attrib = indexvalue.attrib
                    settings[statekey + attrib['index']] = attrib['value']
                    continue
                elif indexvalue.tag != 'SubindexedValues':
                    raise XmlStructureError(f"{indexvalue.tag}")
                indexkey = indexvalue.attrib['index']
                for subindexvalue in list(indexvalue):
                    if subindexvalue.tag != 'SubindexedValue':
                        raise XmlStructureError(f"{subindexvalue.tag}")
                    attrib = subindexvalue.attrib
                    settings[
                        statekey + indexkey + attrib['subindex']
                    ] = attrib['value']
        return settings
    ###
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    date = root.attrib['date']
    date = pd.to_datetime(date).to_pydatetime()
    #
    if root[0].tag == 'PVStateShard':
        pv_settings = shard_helper(root[0])
    elif root[1].tag == 'PVStateShard' and root[0].tag == 'SystemIDs':
        pv_settings = shard_helper(root[1])
        pv_settings['SystemID'] = root[0].get('SystemID')
    else:
        raise XmlStructureError("Unknown root structure for xml")
    pv_settings['date'] = date
    sequences = pd.DataFrame()
    #
    for sequence in root.findall('Sequence'):
        # initialize frames
        frames = []
        seq_attrib = sequence.attrib
        # checks that if there are mutliple voltage inputs/outputs
        for vname, vshort in V_NAMES.items():
            # Find all with vname and check only one exists
            vs = sequence.findall(vname)
            if len(vs) == 0:
                v_attrib = {}
            else:
                assert len(vs) == 1, f"{vs}"
                v = vs[0]
                v_attrib = {k + vshort: v for k, v in v.items()}
            seq_attrib.update(v_attrib)
        ### go through each frame
        for frame in sequence.findall('Frame'):
            f_attrib = frame.attrib  # dictionary
            # update dictionary with info for each filename (etc.)
            # and extra parameters
            extra_param = frame.findall('ExtraParameters')
            assert len(extra_param) == 1, "More than one extra parameter"
            extra_param = extra_param[0].attrib
            f_attrib.update(extra_param)
            # update with frame file info - keep in mind multiple channels
            if channel is None:
                finfo = frame.findall('File')
                assert len(finfo) == 1, "More than one channel recorded"
                finfo = finfo[0].attrib
                f_attrib.update(finfo)
            elif isinstance(channel, int):
                finfo = frame.findall('File')
                for finfo_ in finfo:
                    # each file attribute should have a channel keyword
                    channel_ = finfo_.attrib.get('channel', None)
                    if channel_ is None:
                        raise NameError(f"File dictionary does not have channel key: {finfo_.attrib}")
                    if int(channel_) == channel:
                        f_attrib.update(finfo_.attrib)
                        break
                else:
                    raise ValueError(f"Channel number {channel} not found.")
            else:
                raise NotImplementedError("Loading multiple channels at once.")

            shard_dict = shard_helper(frame.find('PVStateShard'))
            f_attrib.update(shard_dict)
            frames.append(f_attrib)
        # make frames dataframe and add seq_attrib
        frames = pd.DataFrame(frames)
        for k, v in seq_attrib.items():
            frames[k] = v
        #
        sequences = sequences.append(frames, ignore_index=True)
    # add singly unique columns to pv_settings if not in
    # change string of integers and floats if possible
    for k, v in pv_settings.items():
        if isinstance(v, str):
            try:
                pv_settings[k] = int(v)
            except ValueError:
                try:
                    pv_settings[k] = float(v)
                except ValueError:
                    pass
    # iterate over sequence DataFrame and convert data types
    for column, series in sequences.iteritems():
        series = pd.to_numeric(series, errors='ignore')
        sequences[column] = series
        # add columns with one unique value to pv_settings
        if column not in pv_settings:
            u = series.unique()
            if len(u) == 1:
                pv_settings[column] = u[0]
    ###
    return pv_settings, sequences
