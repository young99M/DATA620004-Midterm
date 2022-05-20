import torch
from utils import *
from model import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-type', type=str, required=True, default='baseline', help='baseline or not')
    args = parser.parse_args()

    cifar100_mean = [0.5071, 0.4867, 0.4408]
    cifar100_std = [0.2675, 0.2565, 0.2761]
    cifar100_test_loader = get_test_dataloader(cifar100_mean, cifar100_std, num_workers=4, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = get_network(args)
    if args.type == 'baseline':
        net.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])
    elif args.type == 'cutout':
        net.load_state_dict(torch.load('./checkpoint/ckpt_cutout.pth')['net'])
    elif args.type == 'cutmix':
        net.load_state_dict(torch.load('./checkpoint/ckpt_cutmix.pth')['net'])
    else:
        net.load_state_dict(torch.load('./checkpoint/ckpt_mixup.pth')['net'])       
    print(net)
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
           
            image, label = image.to(device), label.to(device)

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    print()
    print('The method is ' + args.type)
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
