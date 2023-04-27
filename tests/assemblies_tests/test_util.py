from dataclasses import dataclass

from i6_models.assemblies.util import ModelConfiguration

def test_simple_configuration():
    @dataclass
    class TestConfiguration(ModelConfiguration):
        num_layers: int = 5
        hidden_dim: int = 256
        name: str = "Cool Model Configuration"

    test_cfg = TestConfiguration(num_layers=12, name="Even Cooler Model Configuration")
    assert test_cfg.hidden_dim == 256
    assert test_cfg.name == "Even Cooler Model Configuration"
    assert test_cfg.num_layers == 12
    test_cfg.num_layers = 7
    assert test_cfg.num_layers == 7

def test_nested_configuration():

    @dataclass
    class TestConfiguration(ModelConfiguration):
        num_layers: int = 5
        hidden_dim: int = 256
        name: str = "Cool Model Configuration"

    @dataclass
    class TestNestedConfiguration(ModelConfiguration):
        encoder_config: TestConfiguration = TestConfiguration(num_layers=4, hidden_dim=3, name="encoder_config")
        decoder_config: TestConfiguration = TestConfiguration(num_layers=6, hidden_dim=5, name="decoder_config")

    dec_cfg = TestConfiguration()
    test_cfg = TestNestedConfiguration(decoder_config=dec_cfg)

    assert test_cfg.encoder_config.num_layers == 4
    assert test_cfg.encoder_config.hidden_dim == 3
    assert test_cfg.encoder_config.name == "encoder_config"
    assert test_cfg.decoder_config.num_layers == 5
    assert test_cfg.decoder_config.hidden_dim == 256
    assert test_cfg.decoder_config.name == "Cool Model Configuration"
    test_cfg.encoder_config = TestConfiguration(num_layers=1, hidden_dim=2, name="better_encoder_config")
    assert test_cfg.encoder_config.num_layers == 1
    assert test_cfg.encoder_config.hidden_dim == 2
    assert test_cfg.encoder_config.name == "better_encoder_config"
