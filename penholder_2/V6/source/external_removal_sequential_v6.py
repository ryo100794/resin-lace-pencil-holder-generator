#!/usr/bin/env python3
"""Replay external removal with the actual sequential fragment state.

The original planner remains byte-identical. All partition, cutter, sweep and
exit gates are reused. A sibling disappears only after its verified exit.
"""
import json,time
import plan_external_removal_v6 as P

_ORIGINAL_SAVE=P.save_plan
_INSTALLED=False

def try_body_cuts(item,rem,cuts,original,obstacles,saved_routes=None,preferred=None,broad_search=True):
    partition=P.body_partition(rem,cuts)
    if partition is None:return None
    fragments,kerf,checks=partition
    if not fragments:return None
    overlap=obstacles.obstruction(kerf,original,convex=False)
    if overlap>P.TOL:return None
    remaining=list(range(len(fragments)));routes=[];order=[]
    if saved_routes is not None:
        if len(saved_routes)!=len(fragments):return None
        saved_order=[r.get('body_section_index',i) for i,r in enumerate(saved_routes)]
        if any(not isinstance(k,int) for k in saved_order) or sorted(saved_order)!=remaining:return None
        for k,record in zip(saved_order,saved_routes):
            fragment=fragments[k]
            state=obstacles.with_extra([fragments[j] for j in remaining if j!=k])
            legs=P.verify_saved_legs(item,fragment,[fragment],record['translation_legs'],original,state)
            if legs is None:return None
            routes.append((fragment,legs));order.append(k);remaining.remove(k)
    else:
        while remaining:
            progress=False
            for k in remaining:
                fragment=fragments[k]
                state=obstacles.with_extra([fragments[j] for j in remaining if j!=k])
                directions=preferred[k:k+1] if preferred else []
                legs=P.route_fragment(item,fragment,[fragment],original,state,directions,broad_search)
                if legs is None:continue
                routes.append((fragment,legs));order.append(k);remaining.remove(k);progress=True
                break
            if not progress:return None
    return dict(mode='explicit_body_cuts',cuts=cuts,fragments=routes,fragment_source_indices=order,
        kerf_volume_mm3=P.H.volume(kerf),kerf_original_overlap_mm3=P.H.volume(kerf^original),
        kerf_complete_obstacle_state_overlap_mm3=overlap,body_cut_partition_checks=checks,
        actual_body_kerf=kerf,sequential_sibling_state=True)

def save_plan(item,plan,sequence):
    row=_ORIGINAL_SAVE(item,plan,sequence)
    order=plan.get('fragment_source_indices',list(range(len(row['fragments']))))
    for fragment,index in zip(row['fragments'],order):fragment['body_section_index']=index
    row['fragment_extraction_order']=order
    row['sequential_sibling_state']=bool(plan.get('sequential_sibling_state',False))
    if row['sequential_sibling_state']:
        row['sibling_state_proof']='Every earlier listed fragment has a certified complete exit before being omitted; all not-yet-extracted siblings remain obstacles.'
    return row

def install():
    global _INSTALLED
    P.frozen_bytes(__file__)
    if not _INSTALLED:
        P.try_body_cuts=try_body_cuts;P.save_plan=save_plan;_INSTALLED=True

def finish_report(path):
    report=json.loads(path.read_text())
    report['method']='Continuous conservative swept-solid bounds against the complete original, all actual nominal retained stock, all fixed additions and every not-yet-removed support. For additional body cuts, each extracted sibling is omitted only after its certified exit; all remaining siblings stay present. Final exit from the entire initial scene is checked.'
    report['sequential_sibling_verifier']='source/external_removal_sequential_v6.py'
    report['partition_index_field']='Each fragment body_section_index identifies its original ordered nonempty partition fragment; fragment list order is the actual extraction order.'
    temporary=path.with_suffix('.json.part');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)

def main():
    install();started_ns=time.time_ns();path=P.ROOT/'reports/v6_external_removal_validation.json'
    try:P.main()
    except SystemExit as error:
        if error.code==2 and path.exists() and path.stat().st_mtime_ns>=started_ns:finish_report(path)
        raise
    finish_report(path)

if __name__=='__main__':main()
