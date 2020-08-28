#!/bin/sh

CUDA_VISIBLE_DEVICES=1

#for i in 2 3 4 5
#do
#echo "=================training $i stream...====================\n"
#echo "COMMAND: python ${i}stream.py"
#python ${i}stream.py >> train.log 2>&1
#done

for view in 4 5
do
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo "view:$view"
echo "iter:$iter"
python 1stream.py --one_stream_view $view --iteration $iter --save_path ../results/1stream_2delete_910 >> trainlogs/1stream_2delete_910_view45.log 2>&1
done
done
