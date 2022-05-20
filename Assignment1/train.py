import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from utils import * 
from model import *

def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # iteration number
        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))        

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    with torch.no_grad():
        for (images, labels) in cifar100_test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        finish = time.time()

        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            finish - start
        ))
        # print()

        # add informations to tensorboard
        if tb:
            writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
            writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

        return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', required=True, help='net type')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-type', type=str, default='baseline', help='baseline or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    args = parser.parse_args()


    # some parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    milestones = [60, 120, 160]
    epochnum = 200
    log_dir = 'runs'  # tensorboard log dir

    net = get_network(args)

    # data processing
    cifar100_mean = [0.5071, 0.4867, 0.4408]
    cifar100_std = [0.2675, 0.2565, 0.2761]

    cifar100_training_loader = get_train_val_dataloader(cifar100_mean, cifar100_std, batch_size=64, num_workers=4, shuffle=True)
    cifar100_test_loader = get_test_dataloader(cifar100_mean, cifar100_std, num_workers=4, batch_size=64, shuffle=True)
     
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # learning rate decay
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) 
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # use tensorboard
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, args.net, args.type, datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')))
    # image in cifar100: 32 * 32
    input_tensor = torch.Tensor(1, 3, 32, 32).to(device)
    writer.add_graph(net, input_tensor)

    best_acc = 0.0
    
    for epoch in range(1, epochnum + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
    
    if epoch > milestones[0] and best_acc < acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    writer.close()