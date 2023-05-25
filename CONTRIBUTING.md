# Code conventions

* use black to lint the code
* default values for Part/Assembly config:
  * defaults for common hyper-parameters (e.g. number of layers): no
  * defaults for rarely/never tuned hyper-parameters (e.g. optimizer constants): maybe
  * defaults for structural elements (parts / assemblies / PyTorch modules/functions): yes
  * if a PR contains a default (not covered by the third point) there should be a reason given for it in the PR / code
* prefer to use torch.nn.functional over equivalent module unless it’s required for torch.nn.Sequential
* prefer to do checks in the config class if feasible, if checks are not in the config there should be a reason given in the PR / code
