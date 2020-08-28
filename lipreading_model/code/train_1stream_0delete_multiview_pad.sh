
for iter in 1 2 3 4 5 6 7 8 9 10
do
python 1stream_multiview_pad.py --one_stream_view -1 --data_pickle_path ../data/oulu_processed_pad.pkl --save_path ../results/3views_1,2,3_1stream_0delete_pad --iteration $iter >> trainlogs/3views_1,2,3_1stream_0delete_pad.log
done
