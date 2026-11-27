"""drumroles.py - which 'other drums' tracks are toms or cymbals, by track name (Satyr's 6 drum stems).

A session's roles.json puts toms, cymbals and percussion together under 'other_drums'. Satyr learns toms and
cymbals as their own stems, so make_testset.py writes them out separately and prep.py reads them.
The name has to say it: 'Lo Tom', 'MOONBOY - Tom 7', 'PML_MH4_Perc_003_Tom_Alone', 'Crash 2', 'Ride'.
Not a tom: 'LilGeniuz_Perc_TomZay' (a producer's name), 'custom', 'atomic'. Not a cymbal: 'override', 'pride'.
"""
import re

TOMS = re.compile(r'(^|[^a-z])(toms?|tt\d*|floor ?toms?|rack ?toms?|tom ?rolls?|tom ?fills?)([^a-z]|$)', re.I)
CYMBALS = re.compile(r'(^|[^a-z])(crash(es)?|rides?|cymbals?|china|splash(es)?|cym)([^a-z\d]|\d|$)', re.I)


def kind(name):
    """'toms', 'cymbals' or 'perc' for one 'other_drums' track name (toms win if both match)."""
    if TOMS.search(name):
        return 'toms'
    if CYMBALS.search(name):
        return 'cymbals'
    return 'perc'


def split_other(names):
    """{'toms': [...], 'cymbals': [...], 'perc': [...]} for a roles.json 'other_drums' list."""
    out = {'toms': [], 'cymbals': [], 'perc': []}
    for n in names:
        out[kind(n)].append(n)
    return out


def split_key(role, *names):
    """For Satyr's split of an 'other_drums' track: 'toms', 'cymbals' or 'perc_only'; None for other roles.
    The first name that says toms or cymbals decides (track/channel name, then the sample's file name)."""
    if role != 'other_drums':
        return None
    for n in names:
        k = kind(n or '')
        if k != 'perc':
            return k
    return 'perc_only'


def keep_split(tracks):
    """Drop the split stems when the song has no toms or cymbals (prep.py then uses other_drums as perc)."""
    if not any(k in tracks and abs(tracks[k]).max() > 0 for k in ('toms', 'cymbals')):
        for k in ('toms', 'cymbals', 'perc_only'):
            tracks.pop(k, None)
    return tracks
