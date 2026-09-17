"""Every feature an analyst can see has a name in words."""

from rga.explain.glossary import GROUP_TITLES, describe_feature, feature_titles
from rga.features.build import CANDIDATE_BLOCK
from rga.features.spec import FeatureGroup


def test_nothing_in_the_feature_space_is_left_unexplained() -> None:
    """The table shows whatever the model used; none of it may be a bare English word."""
    missing = [name for name in CANDIDATE_BLOCK.names if name not in feature_titles()]

    assert missing == []


def test_endpoint_features_say_which_endpoint_they_describe() -> None:
    assert "субъект" in describe_feature("subj_max_level").lower()
    assert "ресурс" in describe_feature("obj_max_level").lower()


def test_every_group_has_a_title_and_a_meaning() -> None:
    for group in FeatureGroup:
        title, meaning = GROUP_TITLES[group]
        assert title and meaning


def test_an_unknown_feature_falls_back_to_its_own_name() -> None:
    assert describe_feature("something_new") == "something_new"
