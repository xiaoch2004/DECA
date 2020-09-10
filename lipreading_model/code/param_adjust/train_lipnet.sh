
for iter in 1 2 3 4 5 6 7 8 9 10
do
python lipnet_multiview_pad.py --iteration $iter >> trainlogs/lipnet_allviews.log 2>&1
done
