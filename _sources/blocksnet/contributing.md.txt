# Contributing

We welcome you to
[check the existing issues](https://github.com/aimclub/blocksnet/issues)
for bugs or enhancements to work on. If you have an idea for an
extension to BlocksNet, please
[file a new issue](https://github.com/aimclub/blocksnet/issues/new)
so we can discuss it.

Make sure to familiarize yourself with the project layout before making
any major contributions.

At first please see the [Get started](../get_started/index.md) section of the documentation to setup the environment.

## Before submitting your pull request

Before you submit a pull request for your contribution, please work
through this checklist to make sure that you have done everything
necessary so we can efficiently review and accept your changes.

If your contribution changes BlocksNet in any way:

- Update the [`examples`](https://github.com/aimclub/blocksnet/blob/main/examples) directory with an examples of your features or fixes.
- Update the [README.md](https://github.com/aimclub/blocksnet/blob/main/README.md) file if anything there has changed.

If your contribution involves any code changes:

- Update the project unit [`tests`](https://github.com/aimclub/blocksnet/blob/main/examples) to test your code changes.
- Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.

If your contribution requires a new library dependency:

- Double-check that the new dependency is easy to install via `pip` and supports Python 3.10. If the dependency requires a complicated installation, then we most likely won't merge your changes because we want to keep BlocksNet easy to install.
- Add the requirement to [`pyproject.toml`](https://github.com/aimclub/blocksnet/blob/main/pyproject.toml) dependencies.

## Contribute to the documentation

The documentation is generated automatically with all the examples. So please make sure that the `Documentation` check is passed durring commits.

## After submitting your pull request

Check back shortly after submitting your pull request to make sure that
your code passes all checks (except for mirroring). If any of the checks come back with a red X, then do your best to address the errors.
