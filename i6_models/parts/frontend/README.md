# Different front-ends for acoustic encoders

### Contributing

If you want to add your own front-end:

- Normally two classes are required. A config class and a model class
- `Config` class inherits from `ModelConfiguration`
- `Model` class inherits from `nn.Module` from `torch`
- `forward(tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
- `sequence_mask` is a boolean tensor where `True` means is inside the sequence and `False` is masked.
- Please add tests
