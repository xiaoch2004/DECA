
for iter in 6 7 8 9 10
do
python 5stream.py --model_1stream_path ../results/1stream_1delete_10_pad --save_path ../results/5stream_1delete_10_pad --data_pickle_path ../data/oulu_processed_pad.pkl --iteration $iter >> trainlogs/5stream_1delete_10_pad.log 2>&1
done
