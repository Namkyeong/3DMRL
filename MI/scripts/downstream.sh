# For Chromophore Tasks
cd ../
for embedder in CIGIN
do
    for lr in 5e-3
    do
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset "Absorption max (nm)" --batch_size 256 --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path "3DMRL.pth"
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset "Emission max (nm)" --batch_size 256 --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path "3DMRL.pth"
        python evaluate.py --data_path "./data_eval" --embedder $embedder --dataset "Lifetime (ns)" --batch_size 256 --lr $lr --message_passing 3 --device 0 --log_target True --pretrained True --pretrained_path "3DMRL.pth"
    done
done

# For Solvation Free Energy Tasks
cd ../
for embedder in CIGIN
do
    for lr in 1e-3
    do
        python evaluate.py --embedder $embedder --dataset "Abraham" --batch_size 256 --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path "3DMRL.pth"
        python evaluate.py --embedder $embedder --dataset "CombiSolv" --batch_size 256 --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path "3DMRL.pth"
        python evaluate.py --embedder $embedder --dataset "CompSol" --batch_size 256 --lr $lr --message_passing 3 --device 0 ---pretrained True --pretrained_path "3DMRL.pth"
        python evaluate.py --embedder $embedder --dataset "FreeSol" --batch_size 32 --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path "3DMRL.pth"
        python evaluate.py --embedder $embedder --dataset "MNSol" --batch_size 32 --lr $lr --message_passing 3 --device 0 --pretrained True --pretrained_path "3DMRL.pth"
    done
done