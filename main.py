import os, argparse

from vdsr import VDSR


# Arguments
parser = argparse.ArgumentParser(description='TensorFlow implementation of VDSR')

# Select GPU 
parser.add_argument('--gpu-id', type=int, default=0)

parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--model-path', type=str, default='checkpoint/VDSR_pretrained')

# Training settings
parser.add_argument('--epoch', type=int, default=60, help='Number of epoch, default: 60')
parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size, default: 128')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate, default: 0.0001')

# Network setting
parser.add_argument('--layer-depth', type=int, default=20, help='Depth of the network, default: 20')

# Test setttings
parser.add_argument('--scale', type=int, default=3, help='Up-scale factor, only for test. default: 3')

parser.add_argument('--print-interval', type=int, default=100)
parser.add_argument('--eval-interval', type=int, default=200)

# Directory path
parser.add_argument('--train-dataset', type=str, default='291')
parser.add_argument('--train-dataset-path', type=str, default='Train')
parser.add_argument('--valid-dataset', type=str, default='Set5')
parser.add_argument('--test-dataset', type=str, default='Set5')
parser.add_argument('--test-dataset-path', type=str, default='Test')

parser.add_argument('--checkpoint-path', type=str, default='checkpoint/VDSR')
parser.add_argument('--result-dir', type=str, default='result')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    if not args.test:
        checkpoint_path = os.path.join(args.checkpoint_path)
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

    vdsr = VDSR(args)
    
    if args.test:
        vdsr.test()
    else:
        vdsr.train()

if __name__ == '__main__':
    main()
