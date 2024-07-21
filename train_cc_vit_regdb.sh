for trial in 1 2 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=0,1 python cluster_contrast_vit_regdb.py -b 256 -a agw -d  regdb_rgb --epochs 30 --iters 50 --momentum 0.99 --eps 0.6 --num-instances 16 --trial $trial
done
echo 'Done!'
