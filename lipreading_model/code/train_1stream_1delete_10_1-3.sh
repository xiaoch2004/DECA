#!/bin/sh

CUDA_VISIBLE_DEVICES=1

#for i in 2 3 4 5
#do
#echo "=================training $i stream...====================\n"
#echo "COMMAND: python ${i}stream.py"
#python ${i}stream.py >> train.log 2>&1
#done

for view in 1 2 3
do
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo "view:$view"
echo "iter:$iter"
python 1stream_multiview.py --one_stream_view $view --iteration $iter --save_path ../results/1stream_1delete_10_pad --data_pickle_path ../data/oulu_processed_pad.pkl >> trainlogs/1stream_1delete_10_view123_pad.log 2>&1
done
done
