from typing import List, Optional, Tuple, Union

import numpy as np

from iris.io.dataclasses import IrisTemplate
from iris.io.errors import MatcherError


def normalized_HD(irisbitcount: int, maskbitcount: int, norm_mean: float, norm_gradient: float) -> float:
    """Perform normalized HD calculation.

    Args:
        irisbitcount (int): Nonmatched iriscode bit count.
        maskbitcount (int): Common maskcode bit count.
        norm_mean (float): Nonmatch distance used for normalized HD.
        norm_gradient (float): Gradient for linear approximation of normalization term.

    Returns:
        float: Normalized Hamming distance.
    """

    # Linear approximation to replace the previous sqrt-based normalization term.
    norm_HD = max(0, norm_mean - (norm_mean - irisbitcount / maskbitcount) * (norm_gradient * maskbitcount + 0.5))
    return norm_HD


def get_bitcounts(template_probe: IrisTemplate, template_gallery: IrisTemplate, shift: int) -> np.ndarray:
    """Get bitcounts in iris and mask codes.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        shift (int): Rotation shift (in columns)

    Returns:
        np.ndarray: Bitcounts in iris and mask codes.
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
        irisbits (np.ndarray): Nonmatch irisbits.
        maskbits (np.ndarray): Common maskbits.
        half_width (Optional[np.ndarray] = None): List of half of code width. Optional paremeter for scoring the upper and lower halves separately. Defaults to None.
        weights (Optional[np.ndarray] = None): List of weights table. Optional paremeter for weighted HD. Defaults to None.

    Returns:
        Tuple[int, int]: Total nonmatch iriscode bit count and common maskcode bit count, could be a list for top and bottom iris separately.
    """
    if weights:
        irisbitcount = [np.sum((x & y) * z, axis=0) / z.sum() * z.size for x, y, z in zip(irisbits, maskbits, weights)]
        maskbitcount = [np.sum(y * z, axis=0) / z.sum() * z.size for y, z in zip(maskbits, weights)]
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
    norm_gradient: float = 0.00005,
) -> Tuple[float, int]:
    """Compute Hamming distance, without bells and whistles.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        rotation_shift (int): Rotations allowed in matching, in columns. Defaults to 15.
        normalise (bool): Flag to normalize HD. Defaults to False.
        norm_mean (float): Peak of the non-match distribution. Defaults to 0.45.
        norm_gradient (float): Gradient for linear approximation of normalization term. Defaults to 0.00005.

    Returns:
        Tuple[float, int]: Miminum Hamming distance and corresonding rotation shift.
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
            normalized_HD(totalirisbitcount, totalmaskbitcount, norm_mean, norm_gradient)
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
    norm_mean: float = 0.45,
    norm_gradient: float = 0.00005,
    separate_half_matching: bool = False,
    weights: Optional[List[np.ndarray]] = None,
) -> Tuple[float, int]:
    """Compute Hamming distance.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        rotation_shift (int): Rotation allowed in matching, converted to columns.
        normalise (bool, optional): Flag to normalize HD. Defaults to False.
        norm_mean (float, optional): Nonmatch mean distance for normalized HD. Defaults to 0.45.
        norm_gradient (float): Gradient for linear approximation of normalization term. Defaults to 0.00005.
        separate_half_matching (bool, optional): Separate the upper and lower halves for matching. Defaults to False.
        weights (Optional[List[np.ndarray]], optional): List of weights table. Optional paremeter for weighted HD. Defaults to None.

    Raises:
        MatcherError: If probe and gallery iris codes are of different sizes or number of columns of iris codes is not even or If weights (when defined) and iris codes are of different sizes.

    Returns:
        Tuple[float, int]: Miminum Hamming distance and corresonding rotation shift.
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
            normdist = normalized_HD(totalirisbitcount.sum(), totalmaskbitcountsum, norm_mean, norm_gradient)
            if separate_half_matching:
                normdist0 = (
                    normalized_HD(totalirisbitcount[0], totalmaskbitcount[0], norm_mean, norm_gradient)
                    if totalmaskbitcount[0] > 0
                    else norm_mean
                )
                normdist1 = (
                    normalized_HD(totalirisbitcount[1], totalmaskbitcount[1], norm_mean, norm_gradient)
                    if totalmaskbitcount[0] > 0
                    else norm_mean
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
