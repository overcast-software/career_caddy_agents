from pollers.hold_poller import format_scrape_label


def test_no_swap_renders_single_id():
    """No fork — entry id is the row that did the work."""
    assert format_scrape_label(273, 273) == "273"


def test_swap_renders_parent_arrow_child():
    """ResolveFinalUrl forked a child scrape — log must show both ids.

    Regression: the hold-poller used to log only the entry id, so the
    child's outcome was misattributed to the parent (e.g. scrape 302
    reported "graph done outcome=success" twice in logfire even though
    the work actually happened on child 303).
    """
    assert format_scrape_label(302, 303) == "302→303"


def test_final_none_falls_back_to_entry():
    assert format_scrape_label(150, None) == "150"


def test_final_zero_falls_back_to_entry():
    assert format_scrape_label(150, 0) == "150"
