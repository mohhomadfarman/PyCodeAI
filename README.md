# PyCodeAI
 ./venv/Scripts/python.exe 


python cli.py train --epochs 2 --batch-size 8 --grad-accum 4 --vocab-size 5000


day 2

python cli.py train --epochs 15 --device gpu --embed-dim 256 --num-heads 8 --num-layers 6 --seq-len 128 --vocab-size 5000 --batch-size 16 --learning-rate 3e-4


day 3 
python cli.py train --epochs 10 --device gpu \
  --embed-dim 256 --num-heads 8 --num-layers 6 --seq-len 128 \
  --load-model model.npz \
  --learning-rate 1e-4
