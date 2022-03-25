import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simsiam_trans2_Nknn_add_soft')
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--N', type=int, default=2)
parser.add_argument('--M', type=int, default=9)
parser.add_argument('--epoch', type=int, default=800)
parser.add_argument('--warmup_epoch', type=int, default=10)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--soft_loss_rate', type=float, default=0.1, help='soft_loss_rate')

args = parser.parse_args()

exp_name = f'{args.exp_name}_backbone-{args.backbone}_epoch{args.epoch}_warmup-{args.warmup_epoch}_base_lr{args.base_lr}_N{args.N}_M{args.M}_soft_loss_rate-{args.soft_loss_rate}'

cmd_test = f"nohup python main_lincls.py " \
           f"--arch {args.backbone} --num_cls 10 --batch_size 256 --lr 30.0 --weight_decay 0.0 " \
           f"--pretrained ./outputs/{exp_name}/{args.trial}/{args.trial}_best.pth " \
           f"../Data @ linear_test_{exp_name}-{args.trial}.out"

cmd_test = cmd_test.replace(' ', '~blank~')
cmd = f"nohup python main.py " \
      f"--data_root ../Data " \
      f"--exp_dir ./outputs/{exp_name} " \
      f"--arch {args.backbone} " \
      f"--trial {args.trial} " \
      f"--base_lr {args.base_lr} " \
      f"--eval_freq 1 " \
      f"--epochs {args.epoch} " \
      f"--weight_decay 5e-4 " \
      f"--momentum 0.9 " \
      f"--N {args.N} " \
      f"--M {args.M} " \
      f"--soft_loss_rate {args.soft_loss_rate} " \
      f"--batch_size 512 " \
      f"--warmup_epoch {args.warmup_epoch} " \
      f"--gpu 0 " \
      f"--test_cmd {cmd_test} " \
      f"> {exp_name}-{args.trial}.out &"

print('cmd:', cmd)
print('test_cmd:', cmd_test.replace('~blank~', ' ').replace('@', '>'))
os.system(cmd)