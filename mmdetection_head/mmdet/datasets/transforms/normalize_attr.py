from __future__ import annotations

from typing import Dict

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class NormalizeAttr:
    """Normalize pH and VBN attributes and store them back into results.

    Args:
        ph_div (float): Divisor to scale pH value. Default 14.0.
        vbn_div (float): Divisor to scale VBN value. Default 50.0.
    """

    def __init__(self, ph_div: float = 14.0, vbn_div: float = 50.0):
        self.ph_div = ph_div
        self.vbn_div = vbn_div

    def transform(self, results: Dict) -> Dict:
        # Meta keys may reside directly in results dict (from dataset) or in
        # results['img_meta']; handle both.
        if 'ph_value' in results and results['ph_value'] is not None:
            results['ph_value'] = float(results['ph_value']) / self.ph_div
        if 'vbn_value' in results and results['vbn_value'] is not None:
            results['vbn_value'] = float(results['vbn_value']) / self.vbn_div
        return results

    # mmengine transform entrypoint
    __call__ = transform

    def __repr__(self) -> str:  # noqa: D401
        return (f'{self.__class__.__name__}(ph_div={self.ph_div}, '
                f'vbn_div={self.vbn_div})') 