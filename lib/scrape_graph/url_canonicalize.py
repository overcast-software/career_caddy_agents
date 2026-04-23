"""Strip tracker query params from a URL, yielding the canonical form.

Used by ResolveFinalUrl to detect redirect-to-different-job (LinkedIn
comm/ → Greenhouse) and by the dedup short-circuit to compare URLs.

Lives here (not in api) because ai/ runs the scrape-graph. When cc_auto
adopts the same canonicalization rules on its side, share the module
via a small PyPI-style extraction or a checked-in copy.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


# Exact param names to strip (case-insensitive)
_STRIP_EXACT = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "yclid",
    "mc_cid", "mc_eid",
    "_ga", "_gl",
    "igshid",
    "ref", "refid", "ref_id", "refsrc", "ref_src",
    "trackingid", "tracking_id",
    "lipi", "lgcadv", "lgcit",
    "eid", "midtoken", "midsig", "otptoken",
    "sharedid", "sharesource",
    "src",
}
# Glob-style prefixes to strip
_STRIP_PREFIX = ("utm_", "trk", "vq_", "mc_", "hsa_")


def canonicalize_url(url: str) -> str:
    """Strip known tracker params. Case-insensitive param-name match.

    Also strips the fragment (LinkedIn's #... and similar — carries no
    identity). Preserves path, remaining query, order.
    """
    if not url:
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return url

    kept = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        k_low = key.lower()
        if k_low in _STRIP_EXACT:
            continue
        if any(k_low.startswith(p) for p in _STRIP_PREFIX):
            continue
        kept.append((key, value))

    cleaned_query = urlencode(kept, doseq=True)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            # collapse trailing slash duplicates only — don't alter path
            # semantics in any other way
            re.sub(r"/+", "/", parsed.path),
            parsed.params,
            cleaned_query,
            "",  # drop fragment
        )
    )


def urls_differ(submitted: str, landed: str) -> bool:
    """True when the canonicalized forms point at different jobs.

    Handles the LinkedIn comm/ → Greenhouse case: submitted is the
    tracker URL, landed is the ATS destination. Both canonicalized
    and compared after the tracker-param strip.
    """
    return canonicalize_url(submitted) != canonicalize_url(landed)
