# Python library for COntinuous IMprovement of SETs

PyCoimset aims to provide a framework for the iterative solution of nonlinear optimal control problems with distributed binary controls without the need for standard binary optimization methods such as Branch and Bound. In the long term, it is meant to relieve these solvers from the extreme computational stresses that come with spatially distributed control variables so that their efforts can be directed more towards their primary purpose, i.e., making decisions that are actually discrete in nature.

Mathematically, PyCoimset approximately solves infimization problems of the form
```math
\begin{alignedat}{4}
  \inf_U\ && F(U) \\
  \text{s.t.}\ %
  && G_j(U) &\leq 0 \quad && \forall j \in \lbrack n \rbrack, \\
  && U &\in \Sigma \quad
\end{alignedat}
```
where $\Sigma$ is a $\sigma$-algebra and $F \colon \Sigma \to \mathbb{R}$ as well as $G_j \colon \Sigma \to \mathbb{R}$ are appropriately differentiable functionals defined on the quotient space resulting from equating sets in $\Sigma$ if their symmetric difference is a nullset.

The primary intended field of application is one where the functionals are evaluated by approximately solving ordinary or partial differential equations. The algorithms are designed to be resilient to evaluation errors and agnostic with respect to the formatting of the underlying evaluation data structure. This is intended to give the user a lot of flexibility in how they want to implement differential equation solvers.

PyCoimset follows semantic versioning rules. It is currently in a pre-`1.0.0` version. Therefore, the API is still in a lot of flux and breaking changes may occur on each version.


## Installing

PyCoimset is a pure Python library with a minimal dependency footprint. To use it, you need Python 3.11 or newer. The main library depends only on NumPy. You can install PyCoimset using `pip`:
```bash
pip install git+https://kosinus.math.uni-magdeburg.de/mirhahn/pycoimset.git@[version]
```
The `@[version]` part at the end is optional and can be used to install a specific branch or tag. You can also install a pre-packaged Python wheel:
```bash
pip install --index-url https://kosinus.math.uni-magdeburg.de/api/v4/projects/576/packages/pypi pycoimset
```
Future upload to PyPI is planned.

Additional dependencies must be installed to run the examples located in the `examples` subfolder. Each example comes with its own `requirements.txt` file so that you can install those dependencies using
```bash
pip install -r requirements.txt
```

## Usage

Once you have installed PyCoimset, you can import the main package using
```python
import pycoimset
```
Currently, implementing your own problem requires that you create implementations of the `SimilaritySpace`, `SimilarityClass`, and `SignedMeasure` protocols for your particular measure space discretization as well as one `Functional` per functional in your problem. These protocols are defined and documented in `pycoimset.typing.space` and `pycoimset.typing.functional`. You can then pass your functionals to either the `pycoimset.UnconstrainedSolver` or the `pycoimset.PenaltySolver` and use them to solve the infimization problem.

You can refer to the examples to see how these protocols can be implemented. There are currently no standard implementations, though such standard implementations are intended to be added in the future.

## Contributing

PyCoimset is currently in a very immature state and one of the primary sources of this immaturity is API instability. The API cannot stabilize until a sufficiently large pool of applications exists to accurately assess what the practical demands on the API are. Therefore, the best way to help develop PyCoimset is to use it. If you have problems using PyCoimset, feel free to open an issue and we will try to help you. Keep in mind that this is not a full-time project, so the response cycle may feel slow sometimes.

## Roadmap

Currently, the following changes to PyCoimset's API are planned prior to version `1.0.0`:

- [ ] Replacing the `Functional` protocol with a function-based interface similar to the "simplified evaluator" API internally used by the solvers;
- [ ] Standard implementation of `SimilarityClass` for time intervals using variable-length switching time lists;
- [ ] Standard mesh-based implementations of `SimilarityClass` based on FEniCS and/or scikit-fem;
- [ ] Facilities to pull parallelization from the implementation layer to the solver layer with an abstract IPC layer;
- [ ] IPC facility implementation for MPI.

The following changes are planned for an unspecified major release past `1.0.0`:

- [ ] Reimplementation of time-critical code in a compiled language;
- [ ] Reimplementation of the example code in a compiled language.

## License

PyCoimset is released under the Apache License, Version 2.0, which is an OSI-approved open source license that permits commercial use, but limits developer liability to the greatest extent possible. See the `LICENSE.md` file for more details.