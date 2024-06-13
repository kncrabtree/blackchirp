# Contributing to Blackchirp

Discussions and help with installing/using Blackchirp or about program bugs/features can be found in both the [Discord Server](https://discord.gg/88CkbAKUZY) and the [Blackchirp Issues](https://github.com/kncrabtree/blackchirp/issues) page.

## Installation Questions

The Discord server is the best place to request help with installation issues. Include information about your operating system and provide a copy of your ``config.pri`` file as well as any compiler output.

## Reporting a Bug

If you find a Blackchirp bug, please open an issue. Be sure to include your:
- Operating system
- Compiler version
- Blackchirp version
- Expected behavior
- Observed behavior
- Relvant excerpts from log files with error messages

## Submitting Patches/Features

Before starting work on a bugfix or new feature yourself, it is strongly recommended to discuss your plan in the Discord server and to create a Github issue describing the scope of your contribution. This can help make sure you are aware of any potentially overlapping or conflicting development being done elsewhere in the code, or help give guidance on the best way to proceed.

To contribute to Blackchirp, create a fork of the development branch to your own github repository, and start a new branch for your work. When complete, ensure your branch is up-to-date with the development branch and [submit a pull request](https://github.com/kncrabtree/blackchirp/pulls). To ensure that your code can be readily merged, please adhure to the guidelines below:

### C++ Code Guidelines

- C++ classes, structs, and enums have upper-case names, variables and functions have lower-case names.
- C++ member variables are prefixed with ``d_`` for value types, ``p_`` for pointer types, ``pu_`` for ``std::unique_ptr`` types, and ``ps_`` for ``std::shared_ptr`` types.
- C++ Indentation is with spaces only, 4 spaces per indent. Use of the Qt code style available in Qt Creator is strongly recommended.
- Prefer the use of C++ data structures from the standard library over Qt data structures, except where interaction with the Qt API is required.
- Keys used in C++ HeaderStorage and SettingsStorage contexts should be statically declared and appropriately namespaced, not entered as string literals in the code at the point of usage.

### Python Code Guidelines

- Python code is formatted with [``black``](https://github.com/psf/black) and must pass [``pylint -E``](https://pylint.org/).
- Classes and functions must include docstrings.
- Docstrings should be written in [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Example Jupyer notebooks must be fully executed in order to render properly in the documentation.
