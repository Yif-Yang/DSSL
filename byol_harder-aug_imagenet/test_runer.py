import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='byol_trans2_twice')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--N', type=int, default=2)
parser.add_argument('--M', type=int, default=9)
parser.add_argument('--hard_N', type=int, default=8)
parser.add_argument('--hard_M', type=int, default=16)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--linear_epoch', type=int, default=90)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--linear_lr', type=float, default=1.6)
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--linear_batch_size', type=int, default=4096)
parser.add_argument('--dist_url', default='224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--world_size', default=2, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--grid', type=int, default=4)
parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--pretrained', type=int, default=99)
args = parser.parse_args()

exp_name = f'{args.exp_name}_backbone-{args.backbone}_' \
           f'worldsize{args.world_size}_rank{args.rank}_bs{args.batch_size}_epoch{args.epoch}_lr{args.lr}_N{args.N}_M{args.M}_hardN{args.hard_N}_hardM{args.hard_M}_grid{args.grid}_trial{args.trial}'
saved_path = f"./work_dir/{exp_name}"
resume_file = os.path.join(saved_path, 'linear_checkpoint_{:04d}.pth.tar'.format(args.resume))
pretrain_resume_file = os.path.join(saved_path, 'checkpoint_{:04d}.pth.tar'.format(args.pretrained))
assert os.path.isfile(pretrain_resume_file), f'pretrain_resume_file:{pretrain_resume_file} not exists'
cmd = f"nohup python main_lincls.py --cos -a {args.backbone} " \
      f"-p 100 --lr {args.linear_lr} " \
      f"--pretrained {pretrain_resume_file} " \
      f"--batch-size {args.linear_batch_size} " \
      f"--epochs {args.linear_epoch} " \
      f"--multiprocessing-distributed " \
      f"--dist-url tcp://{args.dist_url} " \
      f"--world-size {args.world_size} " \
      f"--rank {args.rank} " \
      f"{f'--resume {resume_file} ' if args.resume >= 0 else ''}" \
      f"/imagenet --print-freq 100 " \
      f"--work_dir ./work_dir/{exp_name} " \
      f"> linear_test_pretrained-{args.pretrained}_{(exp_name+'_resume-'+str(args.resume)) if args.resume >=0 else exp_name}.out &"

print('cmd:', cmd)
os.system(cmd)