################################
### Motion correct functions ###
################################

try:
    import caiman as cm
except ImportError as e:
    raise ImportError(
        f"{e}\n\nMotion correction package requires caiman. "
        "Install caiman using conda: `conda install -c conda-forge caiman`."
    )
from multiprocessing.sharedctypes import Value
from time import time
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter as med_filt
from sklearn.preprocessing import SplineTransformer
import numpy as np
from warnings import warn
import os
import glob
import tifffile
import h5py
from skimage import transform as image_transform
from glob import glob

# try:
#     import deepinterpolation as de
#     from deepinterpolation.cli.training import Training
#     from deepinterpolation.cli.inference import Inference
#     DEEP_INTERP = True
# except ImportError as e:
#     DEEP_INTERP = False
#     warn("Could not import deep interpolation library.")


from .band_filtering import remove_diagonal_bands



def debleach(mov):
    """ Debleach by fiting a model to the median intensity.
    Copied from caiman package.

    Parameter
    ---------
       movie : 3D numpy array
        The first dimension is time, and the remaining dimensions are spatial dimensions (height and width)

    Returns 
    -------
        Debleached movie as a 3D numpy array
    """
    if not isinstance(mov[0, 0, 0], np.float32):
        mov = np.asanyarray(mov, dtype=np.float32)

    t, _, _ = mov.shape
    x = np.arange(t)
    y = np.nanmedian(mov.reshape(t, -1), axis=1)

    def expf(x, a, b, c):
        return a * np.exp(-b * x) + c

    def linf(x, a, b):
        return a * x + b

    try:
        p0 = (y[0] - y[-1], 1e-6, y[-1])
        popt, _ = curve_fit(expf, x, y, p0=p0)
        y_fit = expf(x, *popt)
    except Exception:
        p0 = ((y[-1] - y[0]) / (x[-1] - x[0]), y[0])
        popt, _ = curve_fit(linf, x, y, p0=p0)
        y_fit = linf(x, *popt)

    norm = y_fit - np.nanmedian(y[:])
    mov = mov - norm[:, None, None]

    return mov


def motion_correction_rigid(
    rate,
    movie,
    timestamps,
    drop=5,  # frames of start to drop
    template_interval=None,
    n_jobs=None,
    top=0, bottom=0, left=10, right=0,  # to cut movie
    denoise=100,
    batch_denoise=1000,
    shifts_opencv=True,
    overlaps=(100, 100),
    strides=(120, 120),
    min_mov=0,
    max_shifts=(40, 40),
    border_nan=False,
    niter_rig=1,
    splits_rig=14,
    num_splits_to_process_rig=None,
    splits_els=14,
    num_splits_to_process_els=[7, None],
    upsample_factor_grid=4,
    max_deviation_rigid=3,
    nonneg_movie=False,
    gSig_filt=None,
    pw_rigid=False,
    remove_artifacts=False, 
    remove_artifacts_kwargs={},
    avg_frames=None,
    avg_type='mean',
    cut_post_mc=False,
    correct_for_borders=False,
    debleach_movie=False, 
    replace_nans=False, 
    median_filter=False,
    median_time=False,
):
    """
     Standard motion correction for medulla and lobula recordings.
     Assumes flyback device (left=10)

     Parameters
     ----------
     rate : 
        Frame rate of the movie
     movie : 3D float64 numpy array 
        Movie to be motion corrected 
     timestamps : 1D int array
        Timestamps corresponding to each frame of the movie
     drop : int, optional
        Number of seconds of movie to be dropped at the start
     template_interval : int, optional
        The interval in seconds or frames to be used to calculate the template.
     n_jobs : int, optional
        Number of parallel job to run
     top, bottom, left, right : int, optional 
        Number of pixels to be removed from around the image
     denoise : int, optional
        Number of principal components to be used for denoising
     batch_denoise : int, optional
        Number of frames to be denoised at once 
     shifts_opencv : Boolean, optional
        Whether to use OpenCV's implementation of motion correction
     overlaps : int couple, optional
        Number of pixels to be overlapped between patches
     strides : int couple, optional
        Number of pixels to be skipped between patches
     min_mov : int, optional
        The minimum value of the movie after subtracting the template.
     max_shifts : int couple, optional
        The maximum absolute value of shifts allowed in pixels
    border_nan : boolean, optional
        Whether to set the border to NaN
    niter_rig : int, optional
        Number of iterations of rigid motion correction
    splits_rig : int, optional
        Number of splits to be used for parallelization of rigid motion correction
    num_splits_to_process_rig : int, optional
        Number of splits to be used for parallelization of rigid motion correction, starting from the first split
    splits_els : int, optional
        Number of splits to be used for parallelization of elastic motion correction
    num_splits_to_process_els : optional
        Number of splits to be used for parallelization of elastic motion correction, starting from the first split
    upsample_factor_grid : int, optional
        Factor by which to upsample the grid for elastic motion correction
    max_deviation_rigid : int, optional
        Maximum deviation from the template allowed for rigid motion correction
    nonneg_movie : boolean, optional
        Whether to set negative values in the movie to 0
    gSig_filt : int, optional
        Size of the Gaussian filter to be applied to the movie before motion correction
    pw_rigid : boolean, optional
        Whether to use piecewise-rigid motion correction
    remove_artifacts : boolean, optional
        Whether to remove diagonal lines in the movie caused by the flyback device
    avg_frames : int, optional
        Number of frames to be averaged together
    avg_type : str, optional 
        Type of averaging to be used if avg_frames is not None

     Returns 
     -------
    Movie : Filename of the motion corrected movie
    Timestamps : Numpy array of movie timestamps
    Rate : Frame rate of the movie

    """
    # get frames to drop
    drop_frames = int(drop * rate)
    # cut movie
    movie = movie[
        drop_frames:,
        top:movie.shape[1] - bottom,
        left:movie.shape[2] - right
    ]
    # cut timestamps
    timestamps = timestamps[drop_frames:]
    
    if remove_artifacts:
        movie = remove_diagonal_bands(movie, **remove_artifacts_kwargs)
        
    if median_filter:
        if median_time:
            movie = med_filt(movie, size=int(median_filter))
        else:
            for idx, f in enumerate(movie):
                movie[idx] = med_filt(f, size=int(median_filter))
    
    if avg_frames is not None:
        shape = movie.shape
        rest = shape[0] % avg_frames
        reshape = shape[0] // avg_frames
        movie = movie[rest:]
        timestamps = timestamps[rest:]
        
        movie = movie.reshape(reshape, avg_frames, *shape[1:])
        timestamps = timestamps.reshape(reshape, avg_frames).mean(1)
        rate = rate / avg_frames
        
        if avg_type == 'mean':
            movie = np.mean(movie, axis=1)
        elif avg_type == 'min':
            movie = np.min(movie, axis=1)
        elif avg_type == 'median':
            movie = np.median(movie, axis=1)
        else:
            raise NameError(f"Unknown average type {avg_type}")

    mc_movie = movie.astype(np.float32)

    if template_interval is None:
        template = np.nanmean(mc_movie, axis=0)
    elif isinstance(template_interval, int):
        template = np.nanmean(mc_movie[:template_interval], axis=0)
    elif isinstance(template_interval, float):
        template = np.nanmean(mc_movie[:int(rate * template_interval)], axis=0)
    else:
        t0 = timestamps - np.min(timestamps)
        tbool_template = (
            (t0 >= template_interval[0])
            & (t0 < template_interval[1])
        )
        template = np.nanmean(mc_movie[tbool_template], axis=0)

    # motion correction keywords
    mc_kws = dict(
        min_mov=min_mov,
        max_shifts=max_shifts,
        niter_rig=niter_rig,
        border_nan=border_nan,
        splits_rig=splits_rig,
        num_splits_to_process_rig=num_splits_to_process_rig,
        strides=strides,
        overlaps=overlaps,
        splits_els=splits_els,
        num_splits_to_process_els=num_splits_to_process_els,
        upsample_factor_grid=upsample_factor_grid,
        max_deviation_rigid=max_deviation_rigid,
        shifts_opencv=shifts_opencv,
        nonneg_movie=nonneg_movie,
        gSig_filt=gSig_filt,
        use_cuda=False,
        pw_rigid=False
    )

    # create clusters for parallelization
    if n_jobs is not None:
        try:
            # setup cluster for parallel processing
            _, dview, _ = cm.cluster.setup_cluster(
                backend='multiprocessing',
                n_processes=n_jobs
            )
            mc_kws['dview'] = dview
            created_cluster = True
        except Exception:
            created_cluster = False
            warn("Could not create cluster")
    else:
        created_cluster = False

    # apply motion correction
    mc_obj = cm.motion_correction.MotionCorrect(
        mc_movie,
        **mc_kws
    )
    if pw_rigid:
        mc_obj.motion_correct(
            save_movie=True,
            template=(None if template_interval is None else template)
        )
        for i in range(int(pw_rigid)):
            mc_obj.pw_rigid = True
            mc_obj.template = mc_obj.mmap_file
            mc_obj.motion_correct(
                save_movie=True,
                template=mc_obj.total_template_rig
            )
        mc_filename = mc_obj.fname_tot_els
        bord_px = np.ceil(
            np.maximum(
                np.max(np.abs(mc_obj.x_shifts_els)),
                np.max(np.abs(mc_obj.y_shifts_els))
                )
            ).astype(np.int)
        final_size = np.subtract(
            mc_obj.total_template_els.shape, 
            2 * bord_px
        )
    else:
        mc_obj.motion_correct(template=template, save_movie=True)
        mc_filename = mc_obj.mmap_file
        bord_px = np.ceil(np.max(mc_obj.shifts_rig)).astype(np.int)
        final_size = np.subtract(
            mc_obj.total_template_rig.shape, 
            2 * bord_px
        )

    # incorrect
    # mc_filename = mc_obj.mmap_file

    # terminate cluster if exists
    if created_cluster:
        dview.terminate()

    # load motion corrected file
    m = cm.load(mc_filename)

    m = m.astype(np.float32)
    if denoise:
        m = m.IPCA_denoise(
            components=denoise,
            batch=batch_denoise
        )

    mc_movie[:] = m[:]
    mov = np.array(mc_movie)
    
    # delete motion correction files
    for filename in mc_filename:
        os.remove(filename)
    
    if correct_for_borders:
        final_size_x, final_size_y = final_size
        max_shft_x = np.int(np.ceil((np.shape(mov)[1] - final_size_x) / 2))
        max_shft_y = np.int(np.ceil((np.shape(mov)[2] - final_size_y) / 2))
        max_shft_x_1 = - ((np.shape(mov)[1] - max_shft_x) - (final_size_x))
        max_shft_y_1 = - ((np.shape(mov)[2] - max_shft_y) - (final_size_y))
        if max_shft_x_1 == 0:
            max_shft_x_1 = None
        if max_shft_y_1 == 0:
            max_shft_y_1 = None
            
        mov = mov[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
        
    if debleach_movie:
        if isinstance(debleach_movie, bool):
            mov = debleach(mov)
        else:
            mov_mean = mov.mean((1, 2))  # mean across time
            X = timestamps[:, None]
            knots_per_second = float(debleach_movie)
            n_knots = int(np.round((timestamps.max()-timestamps.min())*knots_per_second, 0))
            spline = SplineTransformer(n_knots, degree=1)
            linear_model = LinearRegression()
            pipe = make_pipeline(spline, linear_model)
            pipe.fit(X, mov_mean)
            mov_mean = pipe.predict(X)
            
            mov = mov - mov_mean[:, None, None]
          
    # hard cutting of borders  
    if cut_post_mc:
        max_shift = np.max(max_shifts)
        mov = mov[:, max_shift:-max_shift, max_shift:-max_shift]
        
    if replace_nans:
        mov[np.isnan(mov)] = 0

    return {
        'movie': mov,
        'timestamps': timestamps,
        'rate': rate
    }


def motion_correction_supreme(
    rate,
    movie,
    timestamps,
    recording_id,
    drop=5,  # frames drop at the beginning in seconds
    # cutting of movie frames before motion correction
    top=0, 
    bottom=0, 
    left=60, 
    right=0,
    # motion correction arguments
    num_frames_split=100,
    max_shifts=(6, 6),
    overlaps=(48, 48),
    strides=(24, 24),
    max_deviation_rigid=3,
    border_nan='copy', 
    # filtering before motion correction (just used to motion correct, but not for denoising)
    q_cut=(1, 99),  # cut movie according to the percentiles
    median_filter=3, 
    median_time=True,
    # denoising parameters
    output_dir='/mnt/engram/deepinterp_models',
    resize='full', 
    size=256,
    steps_per_epoch=30,
    pre_frame=30,
    post_frame=30,
    batch_size=50,
    test_start_frame=0,
    test_end_frame=1000,
    train_start_frame=0,
    train_end_frame=-1,
    lr=0.0001, 
    loss='mean_absolute_error', 
    denoise_movie=True, 
    load_previous_training=False, 
    load_previous_inference=False
):
    """
     Standard motion correction for medulla and lobula recordings.
     Assumes flyback device (left=10)

     Parameters
     ----------
     rate: the frame rate of the movie in frames per second
     movie: the movie to be corrected, filtered, and denoised, as a 3D NumPy array with dimensions (time, height, width)
     timestamps: the timestamps for each frame in the movie, as a 1D NumPy array of floats
     recording_id: an identifier for the recording
     drop: the number of seconds to drop from the beginning of the movie (default 5)
     top, bottom, left, right: the number of pixels to cut off from each side of the movie (default 0 for top, bottom, and right, and 60 for left)
     num_frames_split: the number of frames to split the movie into for motion correction (default 100)
     max_shifts: the maximum amount of shift allowed for each frame during motion correction, as a tuple of integers (default (6, 6))
     overlaps: the amount of overlap between frames during motion correction, as a tuple of integers (default (48, 48))
     strides: the amount of stride between frames during motion correction, as a tuple of integers (default (24, 24))
     max_deviation_rigid: the maximum allowed deviation from a rigid transformation during motion correction (default 3)
     border_nan: how to handle pixels that are outside the bounds of the movie during motion correction (default 'copy')
     q_cut: the percentiles to use for cutting the movie before motion correction, as a tuple of integers (default (1, 99))
     median_filter: the size of the median filter to apply to the movie before motion correction (default 3)
     median_time: whether to apply the median filter over time (default True)
     output_dir: the directory to save the denoised movie in (default '/mnt/engram/deepinterp_models')
     resize: how to resize the movie before denoising (default 'full')
     size: the size to resize the movie to before denoising (default 256)
     steps_per_epoch: the number of steps per epoch during denoising (default 30)
     pre_frame: the number of frames to use as context before the current frame during denoising (default 30)
     post_frame: the number of frames to use as context after the current frame during denoising (default 30)
     batch_size: the batch size to use during denoising (default 50)
     test_start_frame: the starting frame to use for testing during denoising (default 0)
      test_end_frame: the ending frame to use for testing during denoising (default 1000)
     train_start_frame: the starting frame to use for training during denoising (default 0)
     train_end_frame: the ending frame to use for training during denoising (default -1)
     lr: the learning rate to use during denoising (default 0.0001)
     loss: the loss function to use during denoising (default 'mean_absolute_error')
     denoise_movie: whether or not to denoise the movie (default True)
     load_previous_training: whether or not to load the weights of a previously trained denoising model (default False)
     load_previous_inference: whether or not to load the weights of a previously trained inference model

     Returns
     -------
     None

    """
    if load_previous_inference:
        warn("If load_previous_inference, then a lot of unnecessary steps will be taken "
             "and it is assumed that the rest of the parameters must match")

    output_dir = os.path.join(output_dir, f"rec{recording_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    drop_filenames = []
    # get frames to drop
    drop_frames = int(drop * rate)
    # cut movie
    movie = movie[
        drop_frames:,
        top:movie.shape[1] - bottom,
        left:movie.shape[2] - right
    ]
    # cut timestamps
    timestamps = timestamps[drop_frames:]
    
    print('save original movie as temporary npy file.')
    omovie_filename = './tmp_omovie.npy'
    omovie = movie
    drop_filenames.append(omovie_filename)
    np.save(omovie_filename, movie)
       
    print('process original movie for motion correction')
    movie = movie.copy() 
    # preprocessing of movie for motion correction
    if q_cut is not None:
        lower = np.percentile(movie, q=q_cut[0])
        upper = np.percentile(movie, q=q_cut[1])
        movie[movie < lower] = 0
        movie[movie > upper] = upper
    
    if median_filter:
        if median_time:
            movie = med_filt(movie, size=median_filter)
        else:
            for idx, f in enumerate(movie):
                movie[idx] = med_filt(f, size=median_filter)

    # convert datatype of movie
    movie = movie.astype(np.float32)
    # motion correction keywords
    mc_kws = dict(
        max_shifts=max_shifts,
        border_nan=border_nan,
        strides=strides,
        overlaps=overlaps,
        max_deviation_rigid=max_deviation_rigid,
        shifts_opencv=True,
        num_frames_split=num_frames_split,
        nonneg_movie=True,
        use_cuda=False,
        pw_rigid=True
    )

    # apply motion correction
    print('init motion correction object')
    mc_obj = cm.motion_correction.MotionCorrect(
        movie,
        **mc_kws
    )
    print('motion correct')
    # perform rigid motion correction
    mc_obj.motion_correct(save_movie=False)
    
    # apply shifts to original movie
    print('applying shifts to original movie')
    m = mc_obj.apply_shifts_movie(omovie_filename)
    
    drop_filenames.extend([
        getattr(mc_obj, 'fname', None), 
        getattr(mc_obj, 'fname_tot_rig', None), 
        getattr(mc_obj, 'mmap_file', None), 
        getattr(mc_obj, 'fname_tot_els', None)
    ])
    
    print('save movie as temporary tif')
    if border_nan != 'copy':
        raise NotImplementedError("border_nan not copy")
        # this is a bug and will give you nonsense
        # output_file = cm.save_memmap([m], border_to_0=mc_obj.border_to_0)
        # drop_filenames.append(output_file)
        # Y, dims, T = cm.load_memmap(output_file)
        # m = np.reshape(Y.T, [T] + list(dims), order='F')
    
    # denoise_movie
    if denoise_movie:
        print('denoising')
        now = datetime.datetime.now()
        run_uid = now.strftime("%Y_%m_%d_%H_%M")
        mov, timestamps = denoise_complete_movie(
            m, timestamps, output_dir, run_uid, 
            resize=resize, 
            size=size,
            steps_per_epoch=steps_per_epoch,
            pre_frame=pre_frame,
            post_frame=post_frame,
            batch_size=batch_size,
            test_start_frame=test_start_frame,
            test_end_frame=test_end_frame,
            train_start_frame=train_start_frame,
            train_end_frame=train_end_frame,
            lr=lr, 
            loss=loss, 
            load_previous_inference=load_previous_inference, 
            load_previous_training=load_previous_training
        )
    else:
        mov = m
    
    print('saving metadata')
    # save a bunch of metadata  
    metadata = {
        'shifts_rig': np.array(mc_obj.shifts_rig), 
        'x_shifts_els': np.array(mc_obj.x_shifts_els), 
        'y_shifts_els': np.array(mc_obj.y_shifts_els), 
        'border_to_0': mc_obj.border_to_0,
        'median_after2': np.nanmedian(mov, axis=(1, 2)), 
        'median_after1': np.nanmedian(m, axis=(1, 2)), 
        'median_before': np.nanmedian(omovie, axis=(1, 2)), 
        'mean_after2': np.nanmean(mov, axis=(1, 2)), 
        'mean_after1': np.nanmean(m, axis=(1, 2)), 
        'mean_before': np.nanmean(omovie, axis=(1, 2)), 
        'image_after2': np.nanmean(mov, axis=0),
        'image_after1': np.nanmean(m, axis=0), 
        'image_before': np.nanmean(omovie, axis=0), 
        'std_after2': np.nanstd(mov, axis=0), 
        'std_after1': np.nanstd(m, axis=0), 
        'std_before': np.nanstd(omovie, axis=0), 
        'corr_after2': cm.local_correlations(mov.T), 
        'corr_after1': cm.local_correlations(m.T), 
        'corr_before': cm.local_correlations(omovie.T)
    }
    
    print('deleting temporary files')
    # delete filenames
    delete_filenames(drop_filenames)
    
    print('done')
    # data to return
    return {
        'movie': mov,
        'timestamps': timestamps,
        'rate': rate, 
        'MotionCorrectedDataMeta': {'metadata': metadata}
    }

def size_image(mov, nshape, oshape):
    """
    Resizes an input image or video frame to a specified output shapeu 
    using the image_transform.resize function from the scikit-image library.

    Parameters
    ----------
    mov (numpy.ndarray): A 3D numpy array representing the input image or video frame
    nshape (tuple): A tuple of 2 integers representing the current shape of the input image
    oshape (tuple): A tuple of 2 integers representing the desired output shape of the image

    Returns
    -------
    numpy.ndarray: A 3D numpy array representing the resized image or video frame.

    """
    if nshape == oshape:
        return mov
    return image_transform.resize(mov.T, output_shape=oshape[::-1]).T


def resize_image(mov, resize, size):
    """
    Resize image of a movie

    Parameters
    ----------
    mov : 3D float64 numpy array
        Movie to resize
    resize : str
        None, 'expand' or 'full' to set the specific resizing
    size : int
        Size at which to resize the image if resize = 'full'
    """
    oshape = mov.shape[1:]
    if resize is None:
        nshape = oshape
    elif resize == 'expand':
        expand_by = int(size / np.max(oshape))
        assert expand_by, "image larger than possible"
        if expand_by != 1:
            nshape = tuple(np.array(oshape) * expand_by)
            mov = image_transform.resize(mov.T, output_shape=nshape[::-1]).T
    elif resize == 'full':
        nshape = (size, size)
        if oshape != nshape:
            mov = image_transform.resize(mov.T, output_shape=nshape[::-1]).T
    else:
        raise NameError("Resizing name {resize} not recognized.")

    return mov, nshape, oshape        

# delete a bunch of files after processing  
def delete_filenames(filenames):
    """
    Deletes files 

    Parameters
    ----------
    filenames : str
        url of the file to delete

    Returns 
    -------
    None
    """
    if filenames is None:
        return
    if isinstance(filenames, str):
        if os.path.exists(filenames):
            os.remove(filenames)
    elif hasattr(filenames, '__iter__'):
        for filename in filenames:
            delete_filenames(filename)
    else:
        raise ValueError(f"Filename type: type(filenames), filenames")
