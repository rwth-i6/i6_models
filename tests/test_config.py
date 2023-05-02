from dataclasses import dataclass

from i6_models.config import ModelConfiguration


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


def test_config_from_namespace():
    @dataclass
    class TestConfiguration(ModelConfiguration):
        num_layers: int = 4
        hidden_dim: int = 13
        name: str = "Cool Model Configuration"

    @dataclass
    class TestConfiguration2(ModelConfiguration):
        num_layers: int = 3
        hidden_dim: int = 12
        name: str = "Cool Model Configuration2"

    cfg1 = TestConfiguration(num_layers=1, hidden_dim=2)
    cfg2 = TestConfiguration.from_namespace(cfg1)
    assert cfg1 == cfg2

    cfg3 = TestConfiguration2.from_namespace(cfg1)
    assert cfg3.num_layers == 1
    assert cfg3.hidden_dim == 2
    assert cfg3.name == "Cool Model Configuration"


def test_config_typing():
    @dataclass
    class TestConfiguration(ModelConfiguration):
        num_layers: int = 4
        hidden_dim: int = 13
        name: str = "Cool Model Configuration"

    from typeguard import TypeCheckError

    TestConfiguration(num_layers=2, hidden_dim=1)
    with pytest.raises(TypeCheckError):
        TestConfiguration(num_layers=2.0, hidden_dim="One")
