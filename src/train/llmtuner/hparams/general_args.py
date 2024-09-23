from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class GeneralArguments:
    r"""
    Arguments pertaining to which stage we are going to perform.
    """
    stage: Optional[Literal["pt", "sft", "rm", "ppo", "dpo", "emb"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )

    task: Optional[Literal["amazon_pro", "amazon_cat", "danawa_pro", "danawa_cat", "task3_pro", "zodal_cat", "zodal_cat_lv1", "zodal_cat_lv2", "unspsc_lv1", "unspsc_lv2"]] = field(
        default='amazon_product',
        metadata={"help": "KISTI TASK [amazon_product]"}
    )

    input_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether masking zero-padding"}
    )

    use_eos: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether using only EOS as embedding"}
    )

    old_checkpoint: Optional[bool] = field(
        default=False
    )
