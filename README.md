## Pointly-Supervised Panoptic Segmentation (ECCV 2022 Oral)

---

Panoptic segmentation with a single point label per target!

Here are all the codes to reproduce the results. We are still cleaning them and will update later.

The code is based on the implementation of [Panoptic SegFormer](https://github.com/zhiqi-li/Panoptic-SegFormer).

The point labels used in the paper can be downloaded [here](https://drive.google.com/drive/folders/19qBN_da_icbXvMFjlFtz4y24CN2MAvXB?usp=sharing).

## Install

0. Install common prerequisties: Python, PyTorch, CUDA, etc.
1. Install [mmcv](https://github.com/open-mmlab/mmcv). This work is tested with mmcv up to version 1.7.1.
2. Install other requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. Install the custom `pydijkstra` package:
    ```bash
    cd PSPS/py-dijkstra/pydijkstra
    python setup.py install --user
    ```

Note that the `panopticapi` has risk of memory leakage, as been discussed in this [issue](https://github.com/cocodataset/panopticapi/issues/27). The solution is to add a `workers.close()` in `panopticapi/evaluation.py/pq_compute_multi_core` before the function return.


## Prepare Dataset

- Pascal VOC

    1. Download the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
    2. Download the point labels [here](https://drive.google.com/drive/folders/19qBN_da_icbXvMFjlFtz4y24CN2MAvXB?usp=sharing).
    3. Organize the dataset by the following structure:
        ```
        - PSPS/data/voc
            - JPEGImages
                20xx_xxxxxx.jpg
                ...

            - Panoptic        
                voc_panoptic_train_aug.json
                voc_panoptic_val.json
                - voc_panoptic_train_aug_1pnt_uniform
                    20xx_xxxxxx.png
                    ...

                - voc_panoptic_val
                    20xx_xxxxxx.png
                    ...
        ``` 

- MS COCO

    ...

### Train & Test

Please find the `example_run.sh` for details. 


**TODO**

- [ ] clean code
- [x] update README.md
