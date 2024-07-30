# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs

from ..registry import DATASETS


@DATASETS.register_module()
class BaseDetDatasetTest(BaseDataset):
    """Base dataset for detection.

    Args:
        proposal_file (str, optional): Proposals file path. Defaults to None.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        return_classes (bool): Whether to return class information
            for open vocabulary-based algorithms. Defaults to False.
        caption_prompt (dict, optional): Prompt for captioning.
            Defaults to None.
    """
    METAINFO = {
        "classes": ("circle", "square", "triangle"),
    }
    def __init__(self,
                 *args,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False,
                 caption_prompt: Optional[dict] = None,
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        self.caption_prompt = caption_prompt
        self.category_name_to_id = {
            "circle": 0,
            "square": 1,
            "triangle": 2,
        }
        if self.caption_prompt is not None:
            assert self.return_classes, \
                'return_classes must be True when using caption_prompt'
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(*args, **kwargs)

    def full_init(self) -> None:
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - load_proposals: Load proposals from proposal file, if
              `self.proposal_file` is not None.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # get proposals from file
        if self.proposal_file is not None:
            self.load_proposals()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        # import pudb;pudb.set_trace()
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def load_data_list(self) -> List[dict]:        
        import os
        import json

        # import pudb;pudb.set_trace()
        data_list = []
        for annofile_name in os.listdir(self.ann_file):
            anno_path = os.path.join(self.ann_file, annofile_name)
            with open(anno_path, 'r') as f:
                data_info = json.load(f)
                data_info['img_path'] = os.path.join(self.data_prefix["img"], data_info['img_name'])
                data_info = self.format_instance(data_info)
                data_list.append(data_info)
        
        return data_list

    def load_proposals(self) -> None:
        raise NotImplementedError

    def format_instance(self, data_info: dict) -> List[int]:
        for instance in data_info['instances']:
            instance['bbox_label'] = self.category_name_to_id[instance['category']]
            instance['ignore_flag'] = False
        return data_info

