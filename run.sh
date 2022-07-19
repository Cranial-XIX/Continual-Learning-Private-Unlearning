seed=$1

dataset=perm_mnist
alpha=0.5
beta=1.0
python main.py --dataset $dataset --method sequential  --seed $seed                
python main.py --dataset $dataset --method independent --seed $seed                
python main.py --dataset $dataset --method ewc         --seed $seed                
python main.py --dataset $dataset --method er          --seed $seed                
python main.py --dataset $dataset --method derpp       --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha --use_pretrain 


dataset=rot_mnist
alpha=0.5
beta=1.0
python main.py --dataset $dataset --method sequential  --seed $seed                
python main.py --dataset $dataset --method independent --seed $seed                
python main.py --dataset $dataset --method ewc         --seed $seed                
python main.py --dataset $dataset --method er          --seed $seed                
python main.py --dataset $dataset --method derpp       --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha --use_pretrain 

dataset=cifar10
alpha=0.5
beta=0.5
python main.py --dataset $dataset --method sequential  --seed $seed                
python main.py --dataset $dataset --method independent --seed $seed                
python main.py --dataset $dataset --method ewc         --seed $seed                
python main.py --dataset $dataset --method er          --seed $seed                
python main.py --dataset $dataset --method derpp       --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha --use_pretrain 

dataset=cifar10
alpha=0.5
beta=1.0
python main.py --dataset $dataset --method sequential  --seed $seed                
python main.py --dataset $dataset --method independent --seed $seed                
python main.py --dataset $dataset --method ewc         --seed $seed                
python main.py --dataset $dataset --method er          --seed $seed                
python main.py --dataset $dataset --method derpp       --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha                
python main.py --dataset $dataset --method clpu_derpp  --seed $seed --beta $beta --alpha $alpha --use_pretrain 
