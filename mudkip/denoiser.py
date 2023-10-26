import warnings
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
import numpy as np  
import cvxpy as cp
from tqdm import tqdm


class KernelPCADenoiser(RegressorMixin, BaseEstimator):
    """
    Two-Photon movie denoiser class.
    """

    def __init__(
        self, 
        n_components=512,
        k=5, 
        alpha=0.01,
        kernel='rbf', 
        n_jobs=None,
        scaling='features', 
        reconstruction_method='mean',
        nonneg=False, 
        decimate=None,
    ):
        """
        A class for denoising 2-photon calcium imaging movie data using NMF (Non-negative matrix factorization).

        Parameters
        ----------
         n_components (int): the number of components to extract
         (int): the sparsity of the component's loadings
         alpha (float): regularization strength for sparsity
        kernel (str): kernel used for NMF
         n_jobs (int or None): number of parallel jobs to run, None means using all processors
        scaling (str): scaling method for the data matrix
        reconstruction_method (str): method used for reconstructing the denoised data
        nonneg (bool): whether to enforce non-negativity on the factors
        decimate (int or None): factor by which to reduce the size of the data along each dimension
 
        """
        self.n_components = n_components
        self.alpha = alpha
        self.kernel = kernel
        self.k = k
        self.n_jobs = n_jobs
        self.scaling = scaling
        self.reconstruction_method = reconstruction_method
        self.nonneg = nonneg
        self.decimate = decimate
        
    def _format_X(self, X):
        """
        Format the input data `X` for processing by the denoising algorithm.
    
        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape `(n_samples, n_features)`.
    
        Returns
        -------
            numpy.ndarray
            Formatted input data of shape `(n_samples, n_features * k)` if `k > 1`,
            or the original input data `X` if `k == 1`.
        """
        if self.k > 1:
            X = np.hstack([X[i:X.shape[0]-self.k+i+1] for i in range(self.k)])
        return X
    
    def _reformat_X(self, X):
        """
        Reformat the input data X depending on the reconstruction method and the value of k.
    
        If k == 1, return X as is.
        If reconstruction_method is 'mean', average the k frames around each timepoint.
        If reconstruction_method is 'middle', return a new array by stacking k/2 frames at the start and end of X, 
        then the middle k-2 frames of X.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to reformat.
        
        Returns:
        --------
        Xnew : array-like, shape (n_samples_new, n_features_new)
            The reformatted data X.
        """
        if self.k == 1:
            return X
        elif self.reconstruction_method == 'mean':
            count = np.zeros(X.shape[0]+self.k-1)
            Xnew = np.zeros((X.shape[0]+self.k-1, self.n_features_), dtype=X.dtype)
            for i in range(X.shape[0]):
                for ik in range(self.k):
                    Xnew[i+ik] += X[i, ik*self.n_features_:(ik+1)*self.n_features_]
                    count[i+ik] += 1
            Xnew = Xnew / count[:, None]
            return Xnew
        elif self.reconstruction_method == 'middle':
            # make sure the newX has the same shape of X before stacking timepoints
            # get "badly" estimated frames from top and bottom
            topX = np.zeros((self.k//2, self.n_features_))
            bottomX = topX.copy()
            for i in range(self.k // 2):
                topX[i] = X[
                    i, 
                    i*self.n_features_:(i+1)*self.n_features_
                ]
                bottomX[-(i+1)] = X[
                    -(i+1), 
                    self.k*self.n_features_-(i+1)*self.n_features_
                    :self.k*self.n_features_-i*self.n_features_
                ]
            # get middle estimated frames
            X = X[:, (self.k // 2 * self.n_features_):((self.k // 2 + 1) * self.n_features_)]
            # stack all arrays
            X = np.vstack([bottomX, X, topX])
            return X
        else:
            raise NameError(f"Reconstruction method: {self.reconstruction_method}") 
        
    def fit(self, X, y=None, sample_weight=None, X2=None):
        """
        Fits the model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples, n_features), default=None
            Unused parameter.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Unused parameter.
        
        X2 : array-like of shape (n_samples, n_features), default=None
            Additional training input samples to concatenate with `X`.
        
        Returns:
        --------
        self : object
            Returns an instance of the class.
        """
        assert sample_weight is None, "sample weight must be None"
        if y is None:
            y = X
        assert X.shape == y.shape, "X and y must be same shape"
        assert (self.k % 2) == 1, "k must be odd"
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        
        pca = KernelPCA(
            n_components=self.n_components, 
            kernel=self.kernel, 
            n_jobs=self.n_jobs, 
            alpha=self.alpha,
            fit_inverse_transform=True, 
            copy_X=False  # save memory
        )
        
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            self.scaler_ = StandardScaler(copy=False)
            X = self.scaler_.fit_transform(X)
        elif self.scaling == 'overall':
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        
        if X2 is not None:
            print("Adding X2")
            self.x2_scaler_ = StandardScaler()
            X2 = self.x2_scaler_.fit_transform(X2)
            self.x2_features_ = X2.shape[1]
            X = np.hstack([X2, X])
        else:
            self.x2_features_ = 0
            
        self.n_features_ = X.shape[-1]
        
        X = self._format_X(X)
        
        if self.decimate is not None:
            X = X[::self.decimate]
        
        pca.fit(X)
        self.pca_ = pca
        
        return self
    
    def predict(self, X, X2=None):
        """
        Predict the reconstruction of the input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to reconstruct.

        X2 : array-like of shape (n_samples, n_features_x2), default=None
            Additional input data to concatenate with X.

        Returns
        -------
        X_pred : ndarray of shape (n_samples, n_features)
        The reconstructed data.
        """
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        # scale inputs
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.transform(X)
        elif self.scaling == 'overall':
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        
        if self.x2_features_:
            assert X2 is not None, "X2 was used"
            X2 = self.x2_scaler_.fit_transform(X2)
            X = np.hstack([X2, X])
        # format X given self.k
        X = self._format_X(X)
        # prediction as transformation
        X = self.pca_.inverse_transform(self.pca_.transform(X))
        # format back to original
        X = self._reformat_X(X)
        
        if self.x2_features_:
            # cut out X2 features
            X = X[:, self.x2_features_:]

        # inversely transform X scaling
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.inverse_transform(X)
        elif self.scaling == 'overall':
            X = (X * self.std_) + self.mean_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        # ensure non-negativity
        if self.nonneg:
            X[X < 0] = 0
        return X
    
    
class KernelRidgeDenoiser(RegressorMixin, BaseEstimator):
    """
    Two-Photon movie denoiser class using kernel ridge regression
    """

    def __init__(
        self, 
        n_components=None,
        reconstruction_method=None,
        k=5, 
        alpha=0.01,
        kernel='rbf', 
        scaling='features', 
        nonneg=False, 
        decimate=None,
        pad=True, 
        model='kernel',
        verbose=1
    ):
        self.n_components = n_components
        self.reconstruction_method = reconstruction_method
        self.alpha = alpha
        self.kernel = kernel
        self.k = k
        self.scaling = scaling
        self.nonneg = nonneg
        self.decimate = decimate
        self.pad = pad
        self.model = model
        self.verbose = verbose
        
    def _format_X(self, X):
        if self.verbose:
            print("assembling X")
        if self.pad:
            y = X
            X = np.vstack([
                np.zeros((self.k//2, X.shape[-1])), 
                X, 
                np.zeros((self.k//2, X.shape[-1])), 
            ])
        else:
            y = X[self.k//2:-self.k//2]
        
        X = np.hstack([
            X[i:X.shape[0]-self.k+i+1] for i in range(self.k)
            if i != (self.k//2)  # skip middle - to predict
        ])
        return X, y  
        
    def fit(self, X, y=None, sample_weight=None, X2=None):
        assert X2 is None, "X2 not implemented for kernel ridge"
        if self.n_components is not None:
            warnings.warn("n_components will be ignored for kernel ridge.")
        assert sample_weight is None, "sample weight must be None"
        if y is None:
            y = X
        assert X.shape == y.shape, "X and y must be same shape"
        assert (self.k % 2) == 1, "k must be odd"
        assert (self.k > 3), "k must be larget than 3"
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        
        if self.verbose:
            print('scaling X')
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            self.scaler_ = StandardScaler(copy=False)
            X = self.scaler_.fit_transform(X)
        elif self.scaling == 'overall':
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        
        if self.model == 'kernel':
            if self.verbose:
                print("kernel model for prediction")
            ridge = KernelRidge(
                alpha=self.alpha, 
                gamma=1/(X.shape[-1]*(self.k-1)), 
                kernel=self.kernel
            )
        elif self.model == 'linear':
            if not self.alpha:
                if self.verbose:
                    print("linear model for prediction")
                ridge = LinearRegression(copy_X=False, n_jobs=-1)
            else:
                if self.verbose:
                    print("ridge model for prediction")
                ridge = Ridge(alpha=self.alpha, copy_X=False)
        elif self.model == 'svr':
            if self.verbose:
                print("svr model for prediction")
            assert self.scaling != 'features', "No feature scaling for SVR!!!"
            assert X.shape[0]/(1 if self.decimate is None else self.decimate) < 1000, "Too many samples for SVR"          
            ridge = SVR(
                kernel=self.kernel, 
                # tenth percentile is background for sure
                epsilon=np.percentile(np.std(X, axis=0), q=10), 
                C=1/self.alpha, 
                cache_size=200
            )
            ridge = MultiOutputRegressor(ridge, n_jobs=-1)
        else:
            raise NameError(f"AHHH no model {self.model}")
            
        self.n_features_ = X.shape[-1]
        X, y = self._format_X(X)
        
        if self.decimate is not None:
            if self.verbose:
                print(f"decimating by {self.decimate}")
            X = X[::self.decimate]
            y = y[::self.decimate]
        
        ridge.fit(X, y)
        self.ridge_ = ridge
        
        return self
    
    def predict(self, X, X2=None):
        assert X2 is None, "X2 not implemented for kernel ridge"
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        # scale inputs
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.transform(X)
        elif self.scaling == 'overall':
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        # format X given self.k
        X, _ = self._format_X(X)
        # prediction as transformation
        if self.verbose:
            print("predicting")
        X = self.ridge_.predict(X)
        if self.verbose:
            print('rescaling')

        # inversely transform X scaling
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.inverse_transform(X)
        elif self.scaling == 'overall':
            X = (X * self.std_) + self.mean_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        # ensure non-negativity
        if self.nonneg:
            X[X < 0] = 0
        return X
       

class ShallowDenoiser(TransformerMixin, BaseEstimator):
    """
    Two-Photon movie denoiser class using shallow interpolation
    """
    
    def __init__(
        self, 
        spatial_rad=1, 
        temporal_rad=4,
        loss='mse', 
        nonneg=False, 
        solve_kws=None, 
        huberM=1, 
        decimate=10,
        fit_intercept=True
    ):
        self.spatial_rad = spatial_rad
        self.temporal_rad = temporal_rad
        self.loss = loss
        self.nonneg = nonneg
        self.solve_kws = solve_kws
        self.huberM = huberM
        self.decimate = decimate
        self.fit_intercept = fit_intercept
        
    def fit_transform(self, X):
        assert X.ndim == 3, "X must be movie"
        if self.nonneg:
            assert np.all(X >= 0), "X must be nonneg"
        else:
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
            X = (X - self.mean_)/self.std_
            
        solve_kws = ({} if self.solve_kws is None else self.solve_kws)
        
        self.n_pixels_ = np.prod(X.shape[1:])
        
        self.n_samples_ = int(np.ceil(X.shape[0] / self.decimate))
        self.k_ = (self.temporal_rad * 2) + 1
        
        self.n_features_ = ((self.spatial_rad*2+1)**2) * self.temporal_rad * 2
        
        # zeropad X
        # TODO memory efficient
        Xpad = np.pad(
            X, (
                (self.temporal_rad, self.temporal_rad), 
                (self.spatial_rad, self.spatial_rad), 
                (self.spatial_rad, self.spatial_rad)
            )
        )
        
        assert self.n_features_ < (self.n_samples_ / 4), f"Overfitting!!! {self.n_features_} vs. {self.n_samples_}"
        
        if self.nonneg:
            A = cp.Variable((self.n_features_), name='A', pos=True)
            Xin = cp.Parameter((self.n_samples_, self.n_features_), name='Xin', pos=True)
            yout = cp.Parameter((self.n_samples_), name='yout', pos=True)
            if self.fit_intercept:
                bias = cp.Variable(name='bias', pos=True)
            else:
                bias = 0.0
        else:
            A = cp.Variable((self.n_features_), name='A')
            Xin = cp.Parameter((self.n_samples_, self.n_features_), name='Xin')
            yout = cp.Parameter((self.n_samples_), name='yout')
            if self.fit_intercept:
                bias = cp.Variable(name='bias')
            else:
                bias = 0.0
            
        ypred = Xin @ A + bias
            
        if self.loss == 'mse':
            loss = cp.sum_squares(yout - ypred)
        elif self.loss == 'mae':
            loss = cp.norm(yout - ypred, 1)
        elif self.loss == 'huber':
            loss = cp.sum(cp.huber(yout - ypred, self.huberM))
        elif self.loss == 'kl':
            # TODO - is this dpp?
            assert self.nonneg, "For KL problem must be nonneg"
            loss = cp.sum(cp.kl_div(ypred+1, yout+1))
        else:
            raise NameError(f"Unknown loss name: {self.loss}")
            
        # TODO regularize
        reg = 0.0
        
        obj = cp.Minimize(loss+reg)
        prob = cp.Problem(obj)
        
        assert prob.is_dpp(), "BUG problem not dpp!"
        
        Xout = np.zeros_like(X)
        As = np.zeros((self.n_features_, *X.shape[1:]))
        biases = np.zeros(X.shape[1:])
        for h, w in tqdm(np.ndindex(X.shape[1:]), desc='Fitting', total=self.n_pixels_):
            yout.value = X[:, h, w][::self.decimate]
            xpatch = Xpad[
                :, h:h+1+self.spatial_rad*2, w:w+1+self.spatial_rad*2
            ].reshape(Xpad.shape[0], -1)
            Xin_ = np.hstack([
                xpatch[i:xpatch.shape[0]-self.k_+i+1] for i in range(self.k_)
                if i != (self.temporal_rad)  # skip middle - to predict
            ])
            Xin.value = Xin_[::self.decimate]
            
            l = prob.solve(**solve_kws)
            As[:, h, w] = A.value
            if self.fit_intercept:
                biases[h, w] = bias.value
            # predict
            Xout[:, h, w] = Xin_ @ As[:, h, w] + biases[h, w]
            
        self.As_ = As
        self.biases_ = biases
        
        if not self.nonneg:
            Xout = (Xout * self.std_) + self.mean_
        return Xout            
            
    def fit(self, X, y=None):
        self.fit_transform(X)
        return self
    
    def transform(self, X):
        if self.nonneg:
            assert np.all(X >= 0), "X must be nonneg"
        else:
            X = (X - self.mean_)/self.std_
        # zeropad X
        # TODO memory efficient
        Xpad = np.pad(
            X, (
                (self.temporal_rad, self.temporal_rad), 
                (self.spatial_rad, self.spatial_rad), 
                (self.spatial_rad, self.spatial_rad)
            )
        )
        Xout = np.zeros_like(X)
        for h, w in tqdm(np.ndindex(X.shape[1:]), desc='Fitting', total=self.n_pixels_):
            xpatch = Xpad[
                :, h-self.spatial_rad:h+1+self.spatial_rad, w-self.spatial_rad:w+1+self.spatial_rad
            ].reshape(Xpad.shape[0], -1)
            Xin = np.hstack([
                xpatch[i:xpatch.shape[0]-self.k_+i+1] for i in range(self.k_)
                if i != (self.temporal_rad)  # skip middle - to predict
            ])
            Xout[:, h, w] = Xin @ self.As_[:, h, w] + self.biases_[h, w]
            
        if not self.nonneg:
            Xout = (Xout * self.std_) + self.mean_
        return Xout
                            
            
            
            
            
        
