from typing import List, Optional, Tuple, Union

import numpy as np

from iris.io.dataclasses import IrisTemplate
from iris.io.errors import MatcherError


def normalized_HD(irisbitcount: int, maskbitcount: float, nm_dist: float) -> float:
    """Perform normalized HD calculation.

    Args:
        irisbitcount (int): nonmatched iriscode bit count.
        maskbitcount (int): common maskcode bit count.
        nm_dist (float): nonmatch distance used for normalized HD.

    Returns:
        float: normalized Hamming distance.
    """
    norm_HD = max(0, nm_dist - (nm_dist - irisbitcount / maskbitcount) * (0.00005 * maskbitcount + 0.5))
    return norm_HD


def get_bitcounts(template_probe: IrisTemplate, template_gallery: IrisTemplate, shift: int) -> np.ndarray:
    """Get bitcounts in iris and mask codes.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        shift (int): _description_

    Returns:
        np.ndarray: bitcounts in iris and mask codes.
    """
    irisbits = [
        np.roll(probe_code, shift, axis=1) != gallery_code
        for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes)
    ]
    maskbits = [
        np.roll(probe_code, shift, axis=1) & gallery_code
        for probe_code, gallery_code in zip(template_probe.mask_codes, template_gallery.mask_codes)
    ]
    return irisbits, maskbits


def count_nonmatchbits(
    irisbits: np.ndarray,
    maskbits: np.ndarray,
    half_width: Optional[List[int]] = None,
    weights: Optional[List[np.ndarray]] = None,
) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
    """Count nonmatch bits for Hammming distance.

    Args:
        irisbits (np.ndarray): nonmatch irisbits.
        maskbits (np.ndarray): common maskbits.
        half_width (Optional[np.ndarray] = None): list of half of code width. Optional paremeter for scoring the upper and lower halves separately. Defaults to None.
        weights (Optional[np.ndarray] = None): list of weights table. Optional paremeter for weighted HD. Defaults to None.

    Returns:
        Tuple[int, int]: total nonmatch iriscode bit count and common maskcode bit count, could be a list for top and bottom iris separately.
    """
    if weights:
        irisbitcount = [np.sum((x & y) * z, axis=0) / z.sum() for x, y, z in zip(irisbits, maskbits, weights)]
        maskbitcount = [np.sum(y * z, axis=0) / z.sum() for y, z in zip(maskbits, weights)]
    else:
        irisbitcount = [np.sum(x & y, axis=0) for x, y in zip(irisbits, maskbits)]
        maskbitcount = [np.sum(y, axis=0) for y in maskbits]

    if half_width:
        totalirisbitcount = np.sum(
            [[np.sum(x[hw:, ...]), np.sum(x[:hw, ...])] for x, hw in zip(irisbitcount, half_width)], axis=0
        )
        totalmaskbitcount = np.sum(
            [[np.sum(y[hw:, ...]), np.sum(y[:hw, ...])] for y, hw in zip(maskbitcount, half_width)], axis=0
        )
    else:
        totalirisbitcount = np.sum(irisbitcount)
        totalmaskbitcount = np.sum(maskbitcount)

    return totalirisbitcount, totalmaskbitcount


def simple_hamming_distance(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int = 15,
    normalise: bool = False,
    norm_mean: float = 0.45,
) -> Tuple[float, int]:
    """Compute Hamming distance, without bells and whistles.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        rotation_shift (int): Rotations allowed in matching, in columns. Defaults to 15.
        normalise (bool): Flag to normalize HD. Defaults to False.
        norm_mean (float): Peak of the non-match distribution. Defaults to 0.45.

    Returns:
        Tuple[float, int]: miminum Hamming distance and corresonding rotation shift.
    """
    for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes):
        if probe_code.shape != gallery_code.shape:
            raise MatcherError("prove and gallery iriscode are of different sizes")

    best_dist = 1
    rot_shift = 0
    for current_shift in [0] + [y for x in range(1, rotation_shift + 1) for y in (-x, x)]:
        irisbits, maskbits = get_bitcounts(template_probe, template_gallery, current_shift)
        totalirisbitcount, totalmaskbitcount = count_nonmatchbits(irisbits, maskbits)

        if totalmaskbitcount == 0:
            continue

        current_dist = (
            normalized_HD(totalirisbitcount, totalmaskbitcount, norm_mean)
            if normalise
            else totalirisbitcount / totalmaskbitcount
        )

        if current_dist < best_dist:
            best_dist = current_dist
            rot_shift = current_shift

    return best_dist, rot_shift


def hamming_distance(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    normalise: bool = False,
    nm_dist: float = 0.45,
    separate_half_matching: bool = False,
    weights: Optional[List[np.ndarray]] = None,
) -> Tuple[float, int]:
    """Compute Hamming distance.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        rotation_shift (int): rotation allowed in matching, converted to columns.
        normalise (bool): Flag to normalize HD. Defaults to False.
        nm_dist (float): nonmatch mean distance for normalized HD. Defaults to 0.45.
        separate_half_matching (bool): separate the upper and lower halves for matching. Defaults to False.
        weights (Optional[List[np.ndarray]]= None): list of weights table. Optional paremeter for weighted HD. Defaults to None.

    Returns:
        Tuple[float, int]: miminum Hamming distance and corresonding rotation shift.
    """
    half_codewidth = []

    for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes):
        if probe_code.shape != gallery_code.shape:
            raise MatcherError("probe and gallery iris codes are of different sizes")
        if (probe_code.shape[1] % 2) != 0:
            raise MatcherError("number of columns of iris codes need to be even")
        if separate_half_matching:
            half_codewidth.append(int(probe_code.shape[1] / 2))

    if weights:
        for probe_code, w in zip(template_probe.iris_codes, weights):
            if probe_code.shape != w.shape:
                raise MatcherError("weights table and iris codes are of different sizes")

    # Calculate the Hamming distance between probe and gallery template.
    match_dist = 1
    match_rot = 0
    for current_shift in [0] + [y for x in range(1, rotation_shift + 1) for y in (-x, x)]:
        irisbits, maskbits = get_bitcounts(template_probe, template_gallery, current_shift)
        totalirisbitcount, totalmaskbitcount = count_nonmatchbits(irisbits, maskbits, half_codewidth, weights)
        totalmaskbitcountsum = totalmaskbitcount.sum()
        if totalmaskbitcountsum == 0:
            continue

        if normalise:
            normdist = normalized_HD(totalirisbitcount.sum(), totalmaskbitcountsum, nm_dist)
            if separate_half_matching:
                normdist0 = (
                    normalized_HD(totalirisbitcount[0], totalmaskbitcount[0], nm_dist)
                    if totalmaskbitcount[0] > 0
                    else nm_dist
                )
                normdist1 = (
                    normalized_HD(totalirisbitcount[1], totalmaskbitcount[1], nm_dist)
                    if totalmaskbitcount[0] > 0
                    else nm_dist
                )
                Hdist = np.mean(
                    [
                        normdist,
                        (normdist0 * totalmaskbitcount[0] + normdist1 * totalmaskbitcount[1]) / totalmaskbitcountsum,
                    ]
                )
            else:
                Hdist = normdist
        else:
            Hdist = totalirisbitcount.sum() / totalmaskbitcountsum

        if Hdist < match_dist:
            match_dist = Hdist
            match_rot = current_shift

    return match_dist, match_rot
