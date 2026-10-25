#!/usr/bin/env python3
"""Continue explicit removal candidates with verified finite holder endpoints."""
import continue_external_removal_candidates_v6 as C
import external_removal_holder_endpoints_v6 as V
import plan_external_removal_v6 as P

_PRIOR = None
_BASE_VERIFY = C.D.verify_prior


def verify_prior(prior):
    global _PRIOR
    plans = _BASE_VERIFY(prior)
    V.verify_prior_definition(prior)
    _PRIOR = prior
    return plans


def install(phase):
    if _PRIOR is None:
        raise RuntimeError('Prior complete proof must be verified before installing endpoint definitions')
    C.install(phase)
    P.frozen_bytes(__file__)
    V.install(_PRIOR)


def finish(path):
    C.finish(path)
    V.finish_report(path)


if __name__ == '__main__':
    C.D.verify_prior = verify_prior
    C.D.S.install, C.D.S.finish_report = install, finish
    C.D.main()
