
for iter in 6 7 8 9 10
do
python 1stream_multiview_pad.py --one_stream_view -1 --data_pickle_path ../data/oulu_processed_pad.pkl --save_path ../results/5views_1stream_1delete_pad --iteration $iter >> trainlogs/5views_1stream_1delete_pad.log
done
