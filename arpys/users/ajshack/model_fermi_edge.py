"""Definitions of models involving Fermi edges."""

from lmfit.models import update_param_vals
from scipy import stats
import numpy as np
import lmfit as lf
from scipy.ndimage import gaussian_filter
from scipy.special import erfc
import xarray as xr
from multiprocessing import pool, Pool


class HotPool:
    _pool: pool.Pool

    @property
    def pool(self) -> pool.Pool:
        if self._pool is not None:
            return self._pool

        self._pool = Pool()
        return self._pool

    def __del__(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None


def broadcast_model(
    model_cls,
    data,
    broadcast_dims,
    params=None,
    progress=True,
    weights=None,
    safe=False,
    prefixes=None,
    window=None,
    parallelize=None,
):
    """Perform a fit across a number of dimensions.

    Allows composite models as well as models defined and compiled through strings.

    Args:
        model_cls: The model specification
        data: The data to curve fit
        broadcast_dims: Which dimensions of the input should be iterated across as opposed
          to fit across
        params: Parameter hints, consisting of plain values or arrays for interpolation
        progress: Whether to show a progress bar
        weights: Weights to apply when curve fitting. Should have the same shape as the input data
        safe: Whether to mask out nan values
        window: A specification of cuts/windows to apply to each curve fit
        parallelize: Whether to parallelize curve fits, defaults to True if unspecified and more
          than 20 fits were requested
        trace: Controls whether execution tracing/timestamping is used for performance investigation

    Returns:
        An `xr.Dataset` containing the curve fitting results. These are data vars:

        - "results": Containing an `xr.DataArray` of the `lmfit.model.ModelResult` instances
        - "residual": The residual array, with the same shape as the input
        - "data": The original data used for fitting
        - "norm_residual": The residual array normalized by the data, i.e. the fractional error
    """
    if params is None:
        params = {}

    if isinstance(broadcast_dims, str):
        broadcast_dims = [broadcast_dims]

    cs = {}
    for dim in broadcast_dims:
        cs[dim] = data.coords[dim]

    other_axes = set(data.dims).difference(set(broadcast_dims))
    template = data.sum(list(other_axes))
    template.values = np.ndarray(template.shape, dtype=np.object)
    n_fits = np.prod(np.array(list(template.S.dshape.values())))

    if parallelize is None:
        parallelize = n_fits > 20

    print("Copying residual")
    residual = data.copy(deep=True)
    residual.values = np.zeros(residual.shape)

    print("Parsing model")
    model = parse_model(model_cls)

    serialize = parallelize
    fitter = mp_fits.MPWorker(
        data=data,
        uncompiled_model=model,
        prefixes=prefixes,
        params=params,
        safe=safe,
        serialize=serialize,
        weights=weights,
        window=window,
    )

    if parallelize:
        trace(f"Running fits (nfits={n_fits}) in parallel (n_threads={os.cpu_count()})")

        print("Running on multiprocessing pool... this may take a while the first time.")
        from .hot_pool import hot_pool

        pool = hot_pool.pool
        exe_results = list(
            wrap_progress(
                pool.imap(fitter, template.G.iter_coords()), total=n_fits, desc="Fitting on pool..."
            )
        )
    else:
        trace(f"Running fits (nfits={n_fits}) serially")
        exe_results = []
        for _, cut_coords in wrap_progress(
            template.G.enumerate_iter_coords(), desc="Fitting", total=n_fits
        ):
            exe_results.append(fitter(cut_coords))

    if serialize:
        trace("Deserializing...")
        print("Deserializing...")

        def unwrap(result_data):
            # using the lmfit deserialization and serialization seems slower than double pickling with dill
            # result = lmfit.model.ModelResult(compiled_model, compiled_model.make_params())
            # return result.loads(result_data)
            return dill.loads(result_data)

        exe_results = [(unwrap(res), residual, cs) for res, residual, cs in exe_results]
        print("Finished deserializing")

    trace(f"Finished running fits Collating")
    for fit_result, fit_residual, coords in exe_results:
        template.loc[coords] = wrap_for_xarray_values_unpacking(fit_result)
        residual.loc[coords] = fit_residual

    trace("Bundling into dataset")
    return xr.Dataset(
        {
            "results": template,
            "data": data,
            "residual": residual,
            "norm_residual": residual / data,
        },
        residual.coords,
    )


def dict_to_parameters(dict_of_parameters) -> lf.Parameters:
    params = lf.Parameters()

    for param_name, param in dict_of_parameters.items():
        params[param_name] = lf.Parameter(param_name, **param)

    return params


class XModelMixin(lf.Model):
    """A mixin providing curve fitting for ``xarray.DataArray`` instances.

    This amounts mostly to making `lmfit` coordinate aware, and providing
    a translation layer between xarray and raw np.ndarray instances.

    Subclassing this mixin as well as an lmfit Model class should bootstrap
    an lmfit Model to one that works transparently on xarray data.

    Alternatively, you can use this as a model base in order to build new models.

    The core method here is `guess_fit` which is a convenient utility that performs both
    a `lmfit.Model.guess`, if available, before populating parameters and
    performing a curve fit.

    __add__ and __mul__ are also implemented, to ensure that the composite model
    remains an instance of a subclass of this mixin.
    """

    n_dims = 1
    dimension_order = None

    def guess_fit(
        self,
        data,
        params=None,
        weights=None,
        guess=True,
        debug=False,
        prefix_params=True,
        transpose=False,
        **kwargs
    ):
        """Performs a fit on xarray data after guessing parameters.

        Params allows you to pass in hints as to what the values and bounds on parameters
        should be. Look at the lmfit docs to get hints about structure
        """
        if params is not None and not isinstance(params, lf.Parameters):
            params = dict_to_parameters(params)

        if transpose:
            assert (
                len(data.dims) == 1
                and "You cannot transpose (invert) a multidimensional array (scalar field)."
            )

        coord_values = {}
        if "x" in kwargs:
            coord_values["x"] = kwargs.pop("x")

        real_data, flat_data = data, data

        new_dim_order = None
        if isinstance(data, xr.DataArray):
            real_data, flat_data = data.values, data.values
            assert len(real_data.shape) == self.n_dims

            if self.n_dims == 1:
                coord_values["x"] = data.coords[list(data.indexes)[0]].values
            else:

                def find_appropriate_dimension(dim_or_dim_list):
                    if isinstance(dim_or_dim_list, str):
                        assert dim_or_dim_list in data.dims
                        return dim_or_dim_list

                    else:
                        intersect = set(dim_or_dim_list).intersection(data.dims)
                        assert len(intersect) == 1
                        return list(intersect)[0]

                # resolve multidimensional parameters
                if self.dimension_order is None or all(d is None for d in self.dimension_order):
                    new_dim_order = data.dims
                else:
                    new_dim_order = [
                        find_appropriate_dimension(dim_options)
                        for dim_options in self.dimension_order
                    ]

                if list(new_dim_order) != list(data.dims):
                    print("Transposing data for multidimensional fit.")
                    data = data.transpose(*new_dim_order)

                coord_values = {k: v.values for k, v in data.coords.items() if k in new_dim_order}
                real_data, flat_data = data.values, data.values.ravel()

        real_weights = weights
        if isinstance(weights, xr.DataArray):
            if self.n_dims == 1:
                real_weights = real_weights.values
            else:
                if new_dim_order is not None:
                    real_weights = weights.transpose(*new_dim_order).values.ravel()
                else:
                    real_weights = weights.values.ravel()

        if transpose:
            cached_coordinate = list(coord_values.values())[0]
            coord_values[list(coord_values.keys())[0]] = real_data
            real_data = cached_coordinate
            flat_data = real_data

        if guess:
            guessed_params = self.guess(real_data, **coord_values)
        else:
            guessed_params = self.make_params()

        if params is not None:
            for k, v in params.items():
                if isinstance(v, dict):
                    if prefix_params:
                        guessed_params[self.prefix + k].set(**v)
                    else:
                        guessed_params[k].set(**v)

            guessed_params.update(params)

        result = None
        try:
            result = super().fit(
                flat_data, guessed_params, **coord_values, weights=real_weights, **kwargs
            )
            result.independent = coord_values
            result.independent_order = new_dim_order
        except Exception as e:
            print(e)
            if debug:
                import pdb

                pdb.post_mortem(e.__traceback__)
        finally:
            return result

    def xguess(self, data, **kwargs):
        """Tries to determine a guess for the parameters."""
        x = kwargs.pop("x", None)

        real_data = data
        if isinstance(data, xr.DataArray):
            real_data = data.values
            assert len(real_data.shape) == 1
            x = data.coords[list(data.indexes)[0]].values

        return self.guess(real_data, x=x, **kwargs)


def affine_bkg(x: np.ndarray, lin_bkg=0, const_bkg=0) -> np.ndarray:
    """An affine/linear background.

    Args:
        x:
        lin_bkg:
        const_bkg:

    Returns:
        Background of the form
          lin_bkg * x + const_bkg
    """
    return lin_bkg * x + const_bkg


def gaussian(x, center=0, sigma=1, amplitude=1):
    """Some constants are absorbed here into the amplitude factor."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def g(x, mu=0, sigma=0.1):
    """TODO, unify this with the standard Gaussian definition because it's gross."""
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(1 / 2) * ((x - mu) / sigma) ** 2)


def lorentzian(x, gamma, center, amplitude):
    """A straightforward Lorentzian."""
    return amplitude * (1 / (2 * np.pi)) * gamma / ((x - center) ** 2 + (0.5 * gamma) ** 2)


def fermi_dirac(x, center=0, width=0.05, scale=1):
    """Fermi edge, with somewhat arbitrary normalization."""
    return scale / (np.exp((x - center) / width) + 1)


def gstepb(x, center=0, width=1, erf_amp=1, lin_bkg=0, const_bkg=0):
    """Fermi function convoled with a Gaussian together with affine background.

    This accurately represents low temperature steps where thermal broadening is
    less substantial than instrumental resolution.

    Args:
        x: value to evaluate function at
        center: center of the step
        width: width of the step
        erf_amp: height of the step
        lin_bkg: linear background slope
        const_bkg: constant background

    Returns:
        The step edge.
    """
    dx = x - center
    return const_bkg + lin_bkg * np.min(dx, 0) + gstep(x, center, width, erf_amp)


def gstep(x, center=0, width=1, erf_amp=1):
    """Fermi function convolved with a Gaussian.

    Args:
        x: value to evaluate fit at
        center: center of the step
        width: width of the step
        erf_amp: height of the step

    Returns:
        The step edge.
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(1.66511 * dx / width)


def band_edge_bkg(
    x, center=0, width=0.05, amplitude=1, gamma=0.1, lor_center=0, offset=0, lin_bkg=0, const_bkg=0
):
    """Lorentzian plus affine background multiplied into fermi edge with overall offset."""
    return (lorentzian(x, gamma, lor_center, amplitude) + lin_bkg * x + const_bkg) * fermi_dirac(
        x, center, width
    ) + offset


def fermi_dirac_affine(x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1):
    """Fermi step edge with a linear background above the Fermi level."""
    # Fermi edge with an affine background multiplied in
    return (scale + lin_bkg * x) / (np.exp((x - center) / width) + 1) + const_bkg


def gstep_stdev(x, center=0, sigma=1, erf_amp=1):
    """Fermi function convolved with a Gaussian.

    Args:
        x: value to evaluate fit at
        center: center of the step
        sigma: width of the step
        erf_amp: height of the step
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(np.sqrt(2) * dx / sigma)


def twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg):
    """A double lorentzian model.

    This is typically not necessary, as you can use the
    + operator on the Model instances. For instance `LorentzianModel() + LorentzianModel(prefix='b')`.

    This mostly exists for people that prefer to do things the "Igor Way".

    Args:
        x
        gamma
        t_gamma
        center
        t_center
        amp
        t_amp
        lin_bkg
        const_bkg

    Returns:
        A two peak structure.
    """
    L1 = lorentzian(x, gamma, center, amp)
    L2 = lorentzian(x, t_gamma, t_center, t_amp)
    AB = affine_bkg(x, lin_bkg, const_bkg)
    return L1 + L2 + AB


class AffineBroadenedFD(XModelMixin):
    """A model for fitting an affine density of states with resolution broadened Fermi-Dirac occupation."""

    @staticmethod
    def affine_broadened_fd(
        x, fd_center=0, fd_width=0.003, conv_width=0.02, const_bkg=1, lin_bkg=0, offset=0
    ):
        """Fermi function convoled with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            fd_center: center of the step
            fd_width: width of the step
            conv_width: The convolution width
            const_bkg: constant background
            lin_bkg: linear background slope
            offset: constant background
        """
        dx = x - fd_center
        x_scaling = x[1] - x[0]
        fermi = 1 / (np.exp(dx / fd_width) + 1)
        return (
            gaussian_filter((const_bkg + lin_bkg * dx) * fermi, sigma=conv_width / x_scaling)
            + offset
        )

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.affine_broadened_fd, **kwargs)

        self.set_param_hint("offset", min=0.0)
        self.set_param_hint("fd_width", min=0.0)
        self.set_param_hint("conv_width", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Make some heuristic guesses.

        We use the mean value to estimate the background parameters and physically
        reasonable ones to initialize the edge.
        """
        pars = self.make_params()

        pars["%sfd_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.mean().item() * 2)
        pars["%soffset" % self.prefix].set(value=data.min().item())

        pars["%sfd_width" % self.prefix].set(0.005)  # TODO we can do better than this
        pars["%sconv_width" % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiLorentzianModel(XModelMixin):
    """A Lorentzian multiplied by a gstepb background."""

    @staticmethod
    def gstepb_mult_lorentzian(
        x, center=0, width=1, erf_amp=1, lin_bkg=0, const_bkg=0, gamma=1, lorcenter=0
    ):
        """A Lorentzian multiplied by a gstepb background."""
        return gstepb(x, center, width, erf_amp, lin_bkg, const_bkg) * lorentzian(
            x, gamma, lorcenter, 1
        )

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.gstepb_mult_lorentzian, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)
        self.set_param_hint("gamma", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slorcenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiDiracModel(XModelMixin):
    """A model for the Fermi Dirac function."""

    def __init__(self, independent_vars=("x",), prefix="", missing="drop", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(fermi_dirac, **kwargs)

        self.set_param_hint("width", min=0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["{}center".format(self.prefix)].set(value=0)
        pars["{}width".format(self.prefix)].set(value=0.05)
        pars["{}scale".format(self.prefix)].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(gstepb, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoBandEdgeBModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution.

    TODO, actually implement two_band_edge_bkg (find original author and their intent).
    """

    @staticmethod
    def two_band_edge_bkg():
        """Some missing model referenced in old Igor code retained for visibility here."""
        raise NotImplementedError

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(self.two_band_edge_bkg, **kwargs)

        self.set_param_hint("amplitude_1", min=0.0)
        self.set_param_hint("gamma_1", min=0.0)
        self.set_param_hint("amplitude_2", min=0.0)
        self.set_param_hint("gamma_2", min=0.0)

        self.set_param_hint("offset", min=-10)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%scenter" % self.prefix].set(value=0)
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution."""

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(band_edge_bkg, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%scenter" % self.prefix].set(value=0)
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBGModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution."""

    @staticmethod
    def band_edge_bkg_gauss(
        x,
        center=0,
        width=0.05,
        amplitude=1,
        gamma=0.1,
        lor_center=0,
        offset=0,
        lin_bkg=0,
        const_bkg=0,
    ):
        """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution."""
        return np.convolve(
            band_edge_bkg(x, 0, width, amplitude, gamma, lor_center, offset, lin_bkg, const_bkg),
            g(np.linspace(-6, 6, 800), 0, 0.01),
            mode="same",
        )

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(self.band_edge_bkg_gauss, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)
        self.set_param_hint("center", vary=False)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        # pars['%scenter' % self.prefix].set(value=0)
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class FermiDiracAffGaussModel(XModelMixin):
    """Fermi Dirac function with an affine background multiplied, then all convolved with a Gaussian."""

    @staticmethod
    def fermi_dirac_bkg_gauss(x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1, sigma=0.01):
        """Fermi Dirac function with an affine background multiplied, then all convolved with a Gaussian."""
        return np.convolve(
            fermi_dirac_affine(x, center, width, lin_bkg, const_bkg, scale),
            g(x, (min(x) + max(x)) / 2, sigma),
            mode="same",
        )

    def __init__(self, independent_vars=("x",), prefix="", missing="drop", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.fermi_dirac_bkg_gauss, **kwargs)

        # self.set_param_hint('width', min=0)
        self.set_param_hint("width", vary=False)
        # self.set_param_hint('lin_bkg', max=10)
        # self.set_param_hint('scale', max=50000)
        self.set_param_hint("scale", min=0)
        self.set_param_hint("sigma", min=0, vary=True)
        self.set_param_hint("lin_bkg", vary=False)
        self.set_param_hint("const_bkg", vary=False)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["{}center".format(self.prefix)].set(value=0)
        # pars['{}width'.format(self.prefix)].set(value=0.05)
        pars["{}width".format(self.prefix)].set(value=0.0009264)
        pars["{}scale".format(self.prefix)].set(value=data.mean() - data.min())
        pars["{}lin_bkg".format(self.prefix)].set(value=0)
        pars["{}const_bkg".format(self.prefix)].set(value=0)
        pars["{}sigma".format(self.prefix)].set(value=0.023)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStdevModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_stdev(x, center=0, sigma=1, erf_amp=1, lin_bkg=0, const_bkg=0):
        """Fermi function convolved with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            center: center of the step
            sigma: width of the step
            erf_amp: height of the step
            lin_bkg: linear background slope
            const_bkg: constant background
        """
        dx = x - center
        return const_bkg + lin_bkg * np.min(dx, 0) + gstep_stdev(x, center, sigma, erf_amp)

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.gstepb_stdev, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%ssigma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStandardModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_standard(x, center=0, sigma=1, amplitude=1, **kwargs):
        """Specializes paramters in gstepb."""
        return gstepb(x, center, width=sigma, erf_amp=amplitude, **kwargs)

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.gstepb_standard, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%ssigma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%samplitude" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorEdgeModel(XModelMixin):
    """A model for (two lorentzians with an affine background) multiplied by a gstepb."""

    def twolorentzian_gstep(
        x,
        gamma,
        t_gamma,
        center,
        t_center,
        amp,
        t_amp,
        lin_bkg,
        const_bkg,
        g_center,
        sigma,
        erf_amp,
    ):
        """Two Lorentzians, an affine background, and a gstepb edge."""
        TL = twolorentzian(x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg)
        GS = gstep(x, g_center, sigma, erf_amp)
        return TL * GS

    def __init__(self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs):
        """Defer to lmfit for initialization."""
        kwargs.update({"prefix": prefix, "missing": missing, "independent_vars": independent_vars})
        super().__init__(self.twolorentzian_gstep, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_gamma", min=0)
        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%sg_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%sgamma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%st_gamma" % self.prefix].set(0.02)
        pars["%ssigma" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC