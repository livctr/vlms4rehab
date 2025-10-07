#!/usr/bin/env python3
"""
Primitives utilities for stroke rehabilitation analysis

This module provides functions for converting motion and contact predictions
to action primitives and computing evaluation metrics.
"""

import numpy as np
from Levenshtein import distance as levenshtein_distance


def dedupe_list(seq):
    """Deduplicate adjacent elements in a list while preserving order."""
    dedup, cnt = [], []
    for item in seq:
        if dedup and item == dedup[-1]:
            cnt[-1] += 1
            continue
        dedup.append(item)
        cnt.append(1)
    return dedup, cnt


def convert_motions_and_contacts_to_prims(motions, contacts, times, future_window=2.0):
    """
    Convert motion and contact predictions to action primitives.
    
    Args:
        motions: binary list or tuple of length n; motions[i] is True/1 if motion
        contacts: binary list or tuple of length n; contacts[i] is True/1 if contact
        times: list or tuple of floats of length n+1; times[i] is the start
            of segment i, times[i+1] its end.
        future_window: how many seconds ahead to scan for contact to label 'reach'
    
    Returns:
        primitives: list of str of length n, one of
            ["reach","reposition","transport","stabilize","idle"]
        times: the exact same list/tuple you passed in (length n+1)
    """
    primitives = []
    start_times = times[:-1]  # length n
    n = min(len(motions), len(contacts), len(start_times))

    for i in range(n):
        t0 = start_times[i]
        m = motions[i]
        c = contacts[i]

        if m and not c:
            # scan ahead up to future_window
            reach = False
            j = i + 1
            while j < n and (start_times[j] - t0) <= future_window:
                if contacts[j]:
                    reach = True
                    break
                j += 1
            prim = "reach" if reach else "reposition"

        elif m and c:
            prim = "transport"

        elif not m and c:
            prim = "stabilize"

        else:  # not m and not c
            prim = "idle"

        primitives.append(prim)

    # return the new primitives list, and the original times unchanged
    return primitives, times


def get_primitives_score(pred, ref):
    """
    Compute Edit Score and Action Error Rate for primitive sequences.
    
    Args:
        pred: predicted primitive sequence
        ref: reference primitive sequence
    
    Returns:
        dict: Contains 'edit_score' and 'action_error_rate'
    """
    # Normalize to lowercase and deduplicate
    pred_dedup = _dedup([x.lower() for x in pred])
    ref_dedup = _dedup([x.lower() for x in ref])

    max_len = max(len(pred_dedup), len(ref_dedup))
    if max_len == 0:
        edit_score = 100.0
        action_error_rate = 0.0
    else:
        edit_dist = levenshtein_distance(pred_dedup, ref_dedup)
        edit_score = (1 - (edit_dist / max_len)) * 100.0
        
        if len(ref_dedup) == 0:
            action_error_rate = 0.0
        else:
            action_error_rate = edit_dist / len(ref_dedup)
    
    return {
        'edit_score': edit_score,
        'action_error_rate': action_error_rate
    }


def _dedup(lst):
    """Deduplicate the adjacent elements in a list while preserving order."""
    deduped = []
    for item in lst:
        if not deduped or item != deduped[-1]:
            deduped.append(item)
    return deduped


def convert_motion_contact_to_primitives(motion_and_contact, times, future_window=2.0):
    """
    Convert motion and contact predictions to primitives (alternative format).
    
    Args:
        motion_and_contact: list of length n, each either
            - "Yes <SEP> Yes" strings, or
            - 2-tuples ("Yes"/"No", "Yes"/"No")
        times: list or tuple of floats of length n+1; times[i] is the start
            of segment i, times[i+1] its end.
        future_window: how many seconds ahead to scan for contact to label 'reach'
    
    Returns:
        primitives: list of str of length n, one of
            ["reach","reposition","transport","stabilize","idle"]
        times: the exact same list/tuple you passed in (length n+1)
    """
    n = len(motion_and_contact)
    assert len(times) == n + 1, "times must be one longer than motion_and_contact"

    # parse Yes/No into booleans
    motion_flags = []
    contact_flags = []
    for mc in motion_and_contact:
        if isinstance(mc, str):
            mot_str, con_str = mc.split("<SEP>")
            motion = "yes" in mot_str.strip().lower()
            contact = "yes" in con_str.strip().lower()
        else:
            motion = ("yes" in mc[0].strip().lower())
            contact = ("yes" in mc[1].strip().lower())
        motion_flags.append(motion)
        contact_flags.append(contact)

    primitives = []
    start_times = times[:-1]  # length n

    for i in range(n):
        t0 = start_times[i]
        m = motion_flags[i]
        c = contact_flags[i]

        if m and not c:
            # scan ahead up to future_window
            reach = False
            j = i + 1
            while j < n and (start_times[j] - t0) <= future_window:
                if contact_flags[j]:
                    reach = True
                    break
                j += 1
            prim = "reach" if reach else "reposition"

        elif m and c:
            prim = "transport"

        elif not m and c:
            prim = "stabilize"

        else:  # not m and not c
            prim = "idle"

        primitives.append(prim)

    # return the new primitives list, and the original times unchanged
    return primitives, times


# Define the primitive types
PRIMITIVES = ["reach", "reposition", "transport", "stabilize", "idle"]


def validate_primitives(primitives):
    """Validate that all primitives are valid."""
    valid_primitives = set(PRIMITIVES)
    invalid = [p for p in primitives if p.lower() not in valid_primitives]
    if invalid:
        raise ValueError(f"Invalid primitives found: {invalid}. Valid primitives: {PRIMITIVES}")
    return True


def primitive_statistics(primitives):
    """Compute statistics for a primitive sequence."""
    stats = {}
    for primitive in PRIMITIVES:
        count = primitives.count(primitive)
        stats[f"count_{primitive}"] = count
        stats[f"pct_{primitive}"] = count / len(primitives) * 100 if len(primitives) > 0 else 0
    
    stats["total_segments"] = len(primitives)
    stats["unique_primitives"] = len(set(primitives))
    
    return stats


if __name__ == "__main__":
    # Test the functions
    print("Testing primitives utilities...")
    
    # Test deduplication
    test_seq = ['reach', 'reach', 'transport', 'transport', 'idle']
    deduped, counts = dedupe_list(test_seq)
    print(f"Deduplication test: {test_seq} -> {deduped}, {counts}")
    
    # Test primitive conversion
    motions = [1, 1, 0, 0, 1, 1]
    contacts = [0, 1, 1, 0, 0, 1]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    primitives, _ = convert_motions_and_contacts_to_prims(motions, contacts, times)
    print(f"Primitive conversion test: {primitives}")
    
    # Test metrics
    pred = ['reach', 'transport', 'idle']
    gt = ['reach', 'transport', 'stabilize']
    metrics = get_primitives_score(pred, gt)
    print(f"Metrics test: {metrics}")
    
    print("✅ All tests passed!")
