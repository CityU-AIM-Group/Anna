
python train.py --source A --target C  --exp-name A_to_C  --save-path './rerun/officehome/' --warm_up_epoch 20
python train.py --source A --target P  --exp-name A_to_P  --save-path './rerun/officehome/' --warm_up_epoch 20
python train.py --source A --target R  --exp-name A_to_R  --save-path './rerun/officehome/' --warm_up_epoch 20

python train.py --source C --target A  --exp-name C_to_A  --save-path './rerun/officehome/' --warm_up_epoch 10
python train.py --source C --target P  --exp-name C_to_P  --save-path './rerun/officehome/' --warm_up_epoch 10
python train.py --source C --target R  --exp-name C_to_R  --save-path './rerun/officehome/' --warm_up_epoch 10

python train.py --source P --target A  --exp-name P_to_A  --save-path './rerun/officehome/' --warm_up_epoch 30
python train.py --source P --target C  --exp-name P_to_C  --save-path './rerun/officehome/' --warm_up_epoch 30
python train.py --source P --target R  --exp-name P_to_R  --save-path './rerun/officehome/' --warm_up_epoch 25

python train.py --source R --target A  --exp-name R_to_A  --save-path './rerun/officehome/' --warm_up_epoch 30
python train.py --source R --target C  --exp-name R_to_C  --save-path './rerun/officehome/' --warm_up_epoch 30
python train.py --source R --target P  --exp-name R_to_P  --save-path './rerun/officehome/' --warm_up_epoch 30

python generate_results.py