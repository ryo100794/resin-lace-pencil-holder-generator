#!/usr/bin/env python3
"""Search direct actual-stock routes and all cut partitions before short paths.

This is a scheduling overlay on the existing complete-scene verifier. Every
remaining operation receives exact-enabled direct whole-body and declared or
generic body-cut routes. It never searches a short-departure family in this
process. The normal continuation closes a complete residual proof.
"""
import json
import numpy as np

import continue_external_removal_cached_bounds_v6 as D
import external_removal_extended_v6 as E
import external_removal_holder_endpoints_v6 as V
import plan_external_removal_v6 as P

_BASE_VERIFY = D.verify_prior
_BASE_INSTALL = D.S.install
_BASE_FINISH = D.S.finish_report
_PRIOR = None


def verify_prior(prior):
    global _PRIOR
    plans = _BASE_VERIFY(prior)
    V.verify_prior_definition(prior)
    _PRIOR = prior
    return plans


def route_fragment(item, fragment, pieces, original, obstacles, preferred=(), broad_search=True):
    pieces, proxy_proof = E._fragment_bounds(item, fragment, pieces)
    if item['kind'] == 'plate_perimeter_tack':
        # No finishing pads remain in the adopted prefix. Keep the ordinary
        # exact downward route if a future fully bound source still has one.
        return E._plate_routes(item, fragment, pieces, original, obstacles, False)
    bounds = np.asarray(fragment.bounding_box(), float)
    center = (bounds[:3] + bounds[3:]) / 2
    directions = E.candidate_directions(item, center, preferred)
    if not broad_search:
        directions = directions[:4]
    return E._direct_routes(item, fragment, pieces, directions, original, obstacles,
                            allow_exact=True, proxy_proof=proxy_proof)


def install(phase):
    if phase != 'direct':
        raise ValueError('This controller requires --search-phase direct; exact-enabled direct routes are its sole family')
    if _PRIOR is None:
        raise RuntimeError('Prior proof must be verified before the exact-direct overlay')
    _BASE_INSTALL(phase)
    P.frozen_bytes(__file__)
    V.install(_PRIOR)
    P.route_fragment = route_fragment


def finish(path):
    _BASE_FINISH(path)
    V.finish_report(path)
    report = json.loads(path.read_text())
    report['global_search_phase'] = 'exact_direct_whole_and_all_body_partitions'
    report['exact_direct_scheduler'] = 'source/continue_external_removal_exact_direct_v6.py'
    report['candidate_family_order'] = ('Saved instructions, then exact-enabled direct whole-body routes and every '
        'declared/generic body-cut partition from the unchanged planner. No short-departure family is searched.')
    report['phase_scope'] = ('The prior complete proof is byte-verified and its exact residual scene is recreated. '
        'All new cuts, sibling states, continuous sweeps and full exits use unchanged gates. '
        'This process changes candidate scheduling only; unresolved IDs remain explicit.')
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)


if __name__ == '__main__':
    D.verify_prior = verify_prior
    D.S.install, D.S.finish_report = install, finish
    D.main()
