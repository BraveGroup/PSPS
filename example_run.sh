#./tools/dist_train.sh ./configs/panformer/panformer_coco_wsup.py 8 --work-dir ./work_dir/coco_wsps_1x --cfg-options lr_config.step="[8]" runner.max_epochs=12

# ./tools/dist_train.sh ./configs/panformer/panformer_coco_wsup.py 8 --work-dir ./work_dir/coco_wsps_2x --cfg-options lr_config.step="[18]" runner.max_epochs=24

# ./tools/dist_train.sh ./configs/panformer/panformer_coco_wsup.py 8 --work-dir ./work_dir/coco_wsps_3x --cfg-options lr_config.step="[27,34]" runner.max_epochs=36

#./tools/dist_train.sh ./configs/panformer/panformer_voc_wsup.py 8 --work-dir ./work_dir/voc_20ep

#./tools/dist_train.sh ./configs/panformer/pointsup_r50_voc.py 8 --work-dir ./work_dir/voc_repeat

#PYTHONPATH='./' python ./tools/train.py ./configs/models/r50_voc_wsup.py --work-dir ./work_dir/voc_wsup_new

#./tools/dist_train.sh ./configs/models/r50_voc.py 8 --work-dir ./work_dir/voc_new
./tools/dist_train.sh ./configs/models/r50_voc_wsup.py 8 --work-dir ./work_dir/voc_new