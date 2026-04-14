from tape import constants as C


def test_feature_names_length_17():
    assert len(C.FEATURE_NAMES) == 17
    assert len(C.TRADE_FEATURES) == 9
    assert len(C.OB_FEATURES) == 8


def test_avax_excluded_from_pretraining_and_contrastive():
    assert "AVAX" not in C.PRETRAINING_SYMBOLS
    assert "AVAX" not in C.LIQUID_CONTRASTIVE_SYMBOLS
    # LTC is the substitute — gotcha #25
    assert "LTC" in C.LIQUID_CONTRASTIVE_SYMBOLS


def test_spring_sigma_mult_recalibrated():
    # falsifiability prereq #4 — original 2.0 fired >8% on BTC/ETH/SOL/HYPE
    assert C.SPRING_SIGMA_MULT == 3.0


def test_mem_excludes_trivially_copyable():
    # gotcha #22
    for feat in ("delta_imbalance_L1", "kyle_lambda", "cum_ofi_5"):
        assert feat in C.MEM_EXCLUDED_FEATURES


def test_mem_random_mask_features_have_high_autocorr():
    # falsifiability prereq #5 — lag-5 r>0.8 → random position masking
    assert set(C.MEM_RANDOM_MASK_FEATURES) == {"prev_seq_time_span", "kyle_lambda"}
