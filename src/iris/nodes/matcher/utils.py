from typing import List, Literal, Optional, Tuple

import numpy as np

from iris.io.dataclasses import IrisTemplate
from iris.io.errors import MatcherError


def simple_hamming_distance(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int = 15,
    normalise: bool = False,
    norm_mean: float = 0.45,
    norm_nb_bits: float = 12288,
) -> Tuple[float, int]:
    """Compute Hamming distance, without bells and whistles.
    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        rotation_shift (int): Rotations allowed in matching, in columns. Defaults to 15.
        normalise (bool): Flag to normalize HD. Defaults to False.
        norm_mean (float): Peak of the non-match distribution. Defaults to 0.45.
        norm_nb_bits (float): Average number of bits visible in 2 randomly sampled iris codes. Defaults to 12288 (3/4 * total_bits_number for the iris code format v0.1).

    Returns:
        Tuple[float, int]: miminum Hamming distance and corresonding rotation shift.
    """
    for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes):
        if probe_code.shape != gallery_code.shape:
            raise MatcherError("prove and gallery iriscode are of different sizes")

    best_dist = 1
    rot_shift = 0
    for current_shift in range(-rotation_shift, rotation_shift + 1):
        irisbits = [
            np.roll(probe_code, current_shift, axis=1) != gallery_code
            for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes)
        ]
        maskbits = [
            np.roll(probe_code, current_shift, axis=1) & gallery_code
            for probe_code, gallery_code in zip(template_probe.mask_codes, template_gallery.mask_codes)
        ]

        irisbitcount = sum([np.sum(x & y) for x, y in zip(irisbits, maskbits)])
        maskbitcount = sum([maskbit.sum() for maskbit in maskbits])

        if maskbitcount == 0:
            continue

        current_dist = irisbitcount / maskbitcount
        if normalise:
            current_dist = max(0, norm_mean - (norm_mean - current_dist) * np.sqrt(maskbitcount / norm_nb_bits))

        if (current_dist < best_dist) or (current_dist == best_dist and current_shift == 0):
            best_dist = current_dist
            rot_shift = current_shift

    return best_dist, rot_shift


def normalized_HD(irisbitcount: int, maskbitcount: int, sqrt_totalbitcount: float, nm_dist: float) -> float:
    """Perform normalized HD calculation.

    Args:
        irisbitcount (int): nonmatched iriscode bit count.
        maskbitcount (int): common maskcode bit count.
        sqrt_totalbitcount (float): square root of bit counts.
        nm_dist (float): nonmatch distance used for normalized HD.

    Returns:
        float: normalized Hamming distance.
    """
    norm_HD = max(0, nm_dist - (nm_dist - irisbitcount / maskbitcount) * np.sqrt(maskbitcount) / sqrt_totalbitcount)
    return norm_HD


def count_sqrt_totalbits(
    toal_codesize: int,
    half_width: List[int],
    weights: Optional[List[np.ndarray]] = None,
) -> Tuple[float, float, float]:
    """Count total amount of sqrt bits.

    Args:
        toal_codesizes (int): total size of iriscodes.
        half_width (List[int]): half width of iriscodes.
        weights (Optional[List[np.ndarray]] = None): list of weights table. Optional paremeter for weighted HD. Defaults to None.

    Returns:
        Tuple[float, float, float]: square root of bit counts from whole iris, top iris and bottom iris.
    """
    sqrt_totalbitcount = np.sqrt(np.sum([np.sum(w) for w in weights])) if weights else np.sqrt(toal_codesize * 3 / 4)

    sqrt_totalbitcount_bot = (
        np.sqrt(np.sum([np.sum(w[:, :hw, ...]) for w, hw in zip(weights, half_width)]))
        if weights
        else sqrt_totalbitcount / np.sqrt(2)
    )

    sqrt_totalbitcount_top = (
        np.sqrt(np.sum([np.sum(w[:, hw:, ...]) for w, hw in zip(weights, half_width)]))
        if weights
        else sqrt_totalbitcount / np.sqrt(2)
    )

    return sqrt_totalbitcount, sqrt_totalbitcount_top, sqrt_totalbitcount_bot


def count_nonmatchbits(
    irisbits: np.ndarray,
    maskbits: np.ndarray,
    half_width: List[int],
    weights: Optional[List[np.ndarray]] = None,
) -> Tuple[int, int, int, int]:
    """Count nonmatch bits for Hammming distance.

    Args:
        irisbits (np.ndarray): nonmatch irisbits.
        maskbits (np.ndarray): common maskbits.
        half_width (List[int]): list of half of code width.
        weights (Optional[np.ndarray] = None): list of weights table. Optional paremeter for weighted HD. Defaults to None.

    Returns:
        Tuple[int, int, int, int]: nonmatch iriscode bit count and common maskcode bit count from top iris and bottom iris.
    """
    if weights:
        irisbitcount_top = np.sum(
            [
                np.sum(np.multiply(x[:, hw:, ...] & y[:, hw:, ...], z[:, hw:, ...]))
                for x, y, hw, z in zip(irisbits, maskbits, half_width, weights)
            ]
        )
        maskbitcount_top = np.sum(
            [np.sum(np.multiply(x[:, hw:, ...], z[:, hw:, ...])) for x, hw, z in zip(maskbits, half_width, weights)]
        )
        irisbitcount_bot = np.sum(
            [
                np.sum(np.multiply(x[:, :hw, ...] & y[:, :hw, ...], z[:, :hw, ...]))
                for x, y, hw, z in zip(irisbits, maskbits, half_width, weights)
            ]
        )
        maskbitcount_bot = np.sum(
            [np.sum(np.multiply(x[:, :hw, ...], z[:, :hw, ...])) for x, hw, z in zip(maskbits, half_width, weights)]
        )
    else:
        irisbitcount_top = np.sum(
            [np.sum(x[:, hw:, ...] & y[:, hw:, ...]) for x, y, hw in zip(irisbits, maskbits, half_width)]
        )
        maskbitcount_top = np.sum([np.sum(x[:, hw:, ...]) for x, hw in zip(maskbits, half_width)])
        irisbitcount_bot = np.sum(
            [np.sum(x[:, :hw, ...] & y[:, :hw, ...]) for x, y, hw in zip(irisbits, maskbits, half_width)]
        )
        maskbitcount_bot = np.sum([np.sum(x[:, :hw, ...]) for x, hw in zip(maskbits, half_width)])

    return irisbitcount_top, maskbitcount_top, irisbitcount_bot, maskbitcount_bot


def hamming_distance(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    normalise: bool = False,
    nm_dist: float = 0.45,
    nm_type: Literal["linear", "sqrt"] = "sqrt",
    weights: Optional[List[np.ndarray]] = None,
) -> Tuple[float, int]:
    """Compute Hamming distance.

    Args:
        template_probe (IrisTemplate): Iris template from probe.
        template_gallery (IrisTemplate): Iris template from gallery.
        rotation_shift (int): rotation allowed in matching, converted to columns.
        normalise (bool): Flag to normalize HD. Defaults to False.
        nm_dist (float): nonmatch mean distance for normalized HD. Defaults to 0.45.
        nm_type (Literal["linear", "sqrt"]): type of normalized HD. Defaults to "sqrt".
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
        half_codewidth.append(int(probe_code.shape[1] / 2))

    if weights:
        for probe_code, w in zip(template_probe.iris_codes, weights):
            if probe_code.shape != w.shape:
                raise MatcherError("weights table and iris codes are of different sizes")

    if normalise:
        if weights:
            sqrt_totalbitcount, sqrt_totalbitcount_top, sqrt_totalbitcount_bot = count_sqrt_totalbits(
                np.sum([np.size(a) for a in template_probe.iris_codes]), half_codewidth, weights
            )
        else:
            sqrt_totalbitcount, sqrt_totalbitcount_top, sqrt_totalbitcount_bot = count_sqrt_totalbits(
                np.sum([np.size(a) for a in template_probe.iris_codes]), half_codewidth
            )

    # Calculate the Hamming distance between probe and gallery template.
    match_dist = 1
    match_rot = 0
    for shiftby in range(-rotation_shift, rotation_shift + 1):
        irisbits = [
            np.roll(probe_code, shiftby, axis=1) != gallery_code
            for probe_code, gallery_code in zip(template_probe.iris_codes, template_gallery.iris_codes)
        ]
        maskbits = [
            np.roll(probe_code, shiftby, axis=1) & gallery_code
            for probe_code, gallery_code in zip(template_probe.mask_codes, template_gallery.mask_codes)
        ]

        if weights:
            irisbitcount_top, maskbitcount_top, irisbitcount_bot, maskbitcount_bot = count_nonmatchbits(
                irisbits, maskbits, half_codewidth, weights
            )
        else:
            irisbitcount_top, maskbitcount_top, irisbitcount_bot, maskbitcount_bot = count_nonmatchbits(
                irisbits, maskbits, half_codewidth
            )
        maskbitcount = maskbitcount_top + maskbitcount_bot

        if maskbitcount == 0:
            continue

        if normalise:
            normdist_top = (
                normalized_HD(irisbitcount_top, maskbitcount_top, sqrt_totalbitcount_top, nm_dist)
                if maskbitcount_top > 0
                else 1
            )
            normdist_bot = (
                normalized_HD(irisbitcount_bot, maskbitcount_bot, sqrt_totalbitcount_bot, nm_dist)
                if maskbitcount_bot > 0
                else 1
            )
            if nm_type == "linear":
                Hdist = (
                    normalized_HD((irisbitcount_top + irisbitcount_bot), maskbitcount, sqrt_totalbitcount, nm_dist) / 2
                    + (normdist_top * maskbitcount_top + normdist_bot * maskbitcount_bot) / maskbitcount / 2
                )
            elif nm_type == "sqrt":
                w_top = np.sqrt(maskbitcount_top)
                w_bot = np.sqrt(maskbitcount_bot)
                Hdist = (
                    normalized_HD((irisbitcount_top + irisbitcount_bot), maskbitcount, sqrt_totalbitcount, nm_dist) / 2
                    + (normdist_top * w_top + normdist_bot * w_bot) / (w_top + w_bot) / 2
                )
            else:
                raise NotImplementedError(
                    "Given `nm_type` not supported. Expected: Literal[\"linear\", \"sqrt\"]. Received {nm_type}."
                )
        else:
            Hdist = (irisbitcount_top + irisbitcount_bot) / maskbitcount

        if (Hdist < match_dist) or (Hdist == match_dist and shiftby == 0):
            match_dist = Hdist
            match_rot = shiftby

    return match_dist, match_rot
