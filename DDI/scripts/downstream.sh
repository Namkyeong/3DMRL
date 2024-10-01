cd ../
for embedder in CIGIN
do
    for lr in 5e-4
    do
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset ChChMiner --split molecule --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path 3DMRL.pth
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset ChChMiner --split scaffold --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path 3DMRL.pth
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset ZhangDDI --split molecule --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path 3DMRL.pth
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset ZhangDDI --split scaffold --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path 3DMRL.pth
    done
done