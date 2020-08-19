
for iter in 1 2 3 4 5
do
python 5stream.py --model_1stream_path ../results/5stream_1delete_10 --iteration $iter >> trainlogs/5stream-1.log 2>&1
done
