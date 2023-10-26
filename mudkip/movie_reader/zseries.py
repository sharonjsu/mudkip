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

import scipy.io

from mudkip.err import AxolotlError, XmlStructureError

from mudkip.movie_reader.prairieview import find_recording, _get_column, _memmap_tiffs

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

def get_prairieview_data_for_piezo(
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
    add_absolute_trigger_time=True,
    save_mat = False
):
    """
    Get raw two photon data from prairieView recording to insert
    into the imaging.RawTwoPhotonData table in loris/datajoint.

    This is for 3d recordings using a piezo.
    
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
    pv_settings, sequences = read_piezo_xml(xmlpath, channel)
    
    trigger_time_column='Time(ms)',
    trigger_time_conversion=10**-3,
    trigger_column='trigger',
    add_absolute_trigger_time=True
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

    zplane=[]
    for i in sequences['index'].drop_duplicates():
        print(f'Processing file {file_extension}, plane #{i}')
        filenames = sequences[sequences['index']==i]['filename'].sort_index()
        filenames = np.array(filenames)
        
        zplane.insert(i,np.array(_memmap_tiffs(filenames, rec_folder_path,n_jobs=n_jobs)))
    
    
    data2insert = dict(
        rate=1 / pv_settings['framePeriod'],
        timestamps=np.array(sequences['absoluteTime']),
        movie=np.array(zplane),
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
        # laser_wavelength=pv_settings['laserWavelength0'], #can't control wl on the new 2p
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
    if save_mat:
        print(f'Saving data to mat file for {date}-{file_extension} at {os.getcwd()}.')
        scipy.io.savemat(f'tzseries-{file_extension}.mat', {'imageStack': np.transpose(zplane,(2,3,1,0)), 'imagingOffset': data2insert['imaging_offset'], 'timestamps':data2insert['timestamps']})
    
    return data2insert


def get_prairieview_data_for_matlab(
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
    pv_settings, sequences = read_piezo_xml(xmlpath, channel)

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
        print('By the way, this doesnt have a voltage input for some reason... just fyi...')
    #     raise AxolotlError('No voltage input exists, '
    #                        'recording will not be able to be '
    #                        'synced with stimuli.')
                           
    zplane=[]
    for i in sequences['index'].drop_duplicates():
        print(f'Processing file {file_extension}, plane #{i}')
        filenames = sequences[sequences['index']==i]['filename'].sort_index()
        filenames = np.array(filenames)
        
        zplane.insert(i,np.array(_memmap_tiffs(filenames, rec_folder_path,n_jobs=n_jobs)))
        
    np.shape(zplane)
    zplane = np.transpose(zplane,(2,3,1,0))
    
    print('Saving data to mat file for...')
    scipy.io.savemat(f'{date}-{file_extension}.mat', {'imageStack':zplane})
    return 



def read_piezo_xml(xmlpath, channel=None):
    """
    Read the metadata from the prairie view xml file for piezos.

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
    
    for sequence in root.findall('Sequence'): ### ITS BECAUSE THERE ARE MULTIPLE SEQUENCES
        # initialize frames
        frames = []
        seq_attrib = sequence.attrib
        # checks that if there are mutliple voltage inputs/outputs
        for vname, vshort in V_NAMES.items():
            # Find all with vname and check only one exists
            vs = sequence.findall(vname)
            # print(len(vs))
            if len(vs) != 0:
                # v_attrib = {}
                # print(f'this is when lenvs is zero: {v_attrib}')
            # else:
                assert len(vs) == 1, f"{vs}"
                v = vs[0]
                v_attrib = {k + vshort: v for k, v in v.items()}
                # print(v_attrib)
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
