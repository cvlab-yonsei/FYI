import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import wandb
from OT import SinkhornDistance


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--device', type=str, default='0', help='device number')
    parser.add_argument('--run_name', type=str, default='MTT', help='name of the run')
    parser.add_argument('--run_tags', type=str, default=None, help='name of the run')
    parser.add_argument('--batch_aug_syn', type=str, default='Standard', help='type of the batch augmentation for synthesizing images')
    parser.add_argument('--batch_aug', type=str, default='Standard', help='type of the batch augmentation for training networks')
    parser.add_argument('--matching', type=str, default='Baseline', help='How to match features')
    parser.add_argument('--eval_method', type=str, default='Standard_Flip_FlipBatchBT', help='evaluation method')

    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    eval_method = args.eval_method.split('_')

    # Reduce batch size if batch augmentation is used
    if args.batch_aug == 'FlipBatch':
        args.batch_train = args.batch_train//2

    # Downloading should be already done
    if not os.path.exists(args.data_path):
        raise Exception('Wrong data directory')

    args.save_path = './logs/' + args.run_name

    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    eval_it_pool = [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for metric in eval_method:
        accs_all_exps[metric] = dict()
        for key in model_eval_pool:
            accs_all_exps[metric][key] = []

    data_save = []

    dsa_params = args.dsa_param

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               tags=args.run_tags.split('_')
               )
    if args.run_name is not None:
        wandb.run.name = args.run_name

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params

    class FlipBatchMaxGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
            return x
        @staticmethod
        def backward(ctx, g):
            g_original = g[:g.size(0)//2]
            g_flipped = torch.flip(g[g.size(0)//2:], dims=[-1])
            # take the max
            g = torch.where(torch.abs(g_original) > torch.abs(g_flipped), g_original, g_flipped)
            return g, None
        
    class FlipBatchMaxGradRescale(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
            return x
        @staticmethod
        def backward(ctx, g):
            g_original = g[:g.size(0)//2]
            g_flipped = torch.flip(g[g.size(0)//2:], dims=[-1])
            # take the larger absolute value
            g = torch.where(torch.abs(g_original) > torch.abs(g_flipped), g_original, g_flipped)
            # Rescale by norm
            g = g / torch.norm(g) * torch.norm(g_original)
            return g, None
        
    class FlipBatchMinGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
            return x
        @staticmethod
        def backward(ctx, g):
            g_original = g[:g.size(0)//2]
            g_flipped = torch.flip(g[g.size(0)//2:], dims=[-1])
            # take the min
            g = torch.where(torch.abs(g_original) < torch.abs(g_flipped), g_original, g_flipped)
            return g, None
        
    class FlipBatchZeroConflict(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
            return x
        @staticmethod
        def backward(ctx, g):
            g_original = g[:g.size(0)//2]
            g_flipped = torch.flip(g[g.size(0)//2:], dims=[-1])
            # Ensure that the gradients of the two flipped images do not conflict
            # If they conflict, set the gradient to the one with larger magnitude
            g = torch.where(g_original * g_flipped < 0, torch.where(torch.abs(g_original) > torch.abs(g_flipped), g_original, g_flipped), g_original + g_flipped)
            return g, None

    def BatchAug(img, batch_aug=None):
        # img: (N, C, H, W)
        if batch_aug == 'Standard':
            return img
        # Best
        elif batch_aug in ['FlipBatch', 'FlipBatchBT']:
            img = torch.cat([img, torch.flip(img, dims=[-1])], dim=0)
            return img
        elif batch_aug == 'Flip':
            randf = torch.rand(img.size(0), 1, 1, 1, device=img.device)
            return img
        elif batch_aug == 'FlipBatchMaxGrad':
            img = FlipBatchMaxGrad.apply(img)
            return img
        elif batch_aug == 'FlipBatchMaxGradRescale':
            img = FlipBatchMaxGradRescale.apply(img)
            return img
        elif batch_aug == 'FlipBatchZeroConflict':
            img = FlipBatchZeroConflict.apply(img)
            return img
        else:
            raise NotImplementedError('batch augmentation %s is not implemented'%batch_aug)


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)



        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle].to(args.device)

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            wandb.log({"Progress": exp}, step=exp)

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    for metric in eval_method:
                        print(f'Evaluate by {metric} method')
                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, batch_aug=metric)
                            accs.append(acc_test)
                        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                        wandb.log({f'Accuracy_{metric}_default/{model_eval}': np.mean(accs)}, step=exp)
                        wandb.log({f'Std_{metric}_default/{model_eval}': np.std(accs)}, step=exp)
                        if it == args.Iteration: # record the final results
                            accs_all_exps[metric][model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=10) # Trying normalize = True/False may get better visual effects.
                wandb.log({"Synthetic_Images": wandb.Image(make_grid(image_syn_vis, nrow=10))}, step=exp)



            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                # loss = torch.tensor(0.0).to(args.device)
                optimizer_img.zero_grad()
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    img_syn= BatchAug(img_syn, args.batch_aug_syn)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    if args.matching == 'Baseline':
                        output_real = embed(img_real)
                        output_syn = embed(img_syn)
                        loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                    elif args.matching == 'FlipFeat':
                        output_real = embed(img_real).detach()
                        output_syn = net.features(img_syn)
                        output_syn = torch.cat([output_syn, torch.flip(output_syn, dims=[-1])], dim=0)
                        output_syn = output_syn.view(output_syn.size(0), -1)
                        loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                    elif args.matching == 'AlignMatch':
                        output_real = net.features(img_real).detach()
                        output_syn = torch.mean(net.features(img_syn), dim=0)

                        with torch.no_grad():
                            feat1 = output_real.view(output_real.shape[0], output_real.shape[1], -1)
                            feat2 = torch.repeat_interleave(output_syn.unsqueeze(0), args.batch_real, dim=0)
                            feat2 = feat2.view(output_real.shape[0], output_real.shape[1], -1)

                            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
                            P = sinkhorn(feat1.permute(0,2,1), feat2.permute(0,2,1)).detach()  # optimal plan batch x 4 x 4
                            P = P*(output_real.size(2)*output_real.size(3)) # assignment matrix
                            f2 = torch.matmul(feat1, P.cuda()).view(output_real.shape).to(args.device)

                        f2 = torch.mean(f2, dim=0)
                        loss = torch.sum((output_syn.view(output_syn.size(0), -1) - f2.view(f2.size(0), -1))**2)
                    else:
                        raise NotImplementedError
                    
                    loss.backward()
                    loss_avg += loss.item()
                optimizer_img.step()

            else: # for ConvNetBN
                raise NotImplementedError
                # images_real_all = []
                # images_syn_all = []
                # loss = torch.tensor(0.0).to(args.device)
                # for c in range(num_classes):
                #     img_real = get_images(c, args.batch_real)
                #     img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                #     img_syn= BatchAug(img_syn, args.batch_aug_syn)

                #     if args.dsa:
                #         seed = int(time.time() * 1000) % 100000
                #         img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                #         img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                #     images_real_all.append(img_real)
                #     images_syn_all.append(img_syn)

                # images_real_all = torch.cat(images_real_all, dim=0)
                # images_syn_all = torch.cat(images_syn_all, dim=0)

                # output_real = embed(images_real_all).detach()
                # if args.flip_feat:
                #     output_syn = net.features(images_syn_all)
                #     output_syn = torch.cat([output_syn, torch.flip(output_syn, dims=[3])], dim=0)
                #     output_syn = output_syn.view(output_syn.size(0), -1)
                # else:
                #     output_syn = embed(images_syn_all)

                # loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

                # optimizer_img.zero_grad()
                # loss.backward()
                # optimizer_img.step()
                # loss_avg += loss.item()
            


            loss_avg /= (num_classes)

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        for metric in eval_method:
            accs = accs_all_exps[metric][key]
            print(f"Accuracy_{metric}")
            print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

        # log the final accuracy
        data = [[f"{metric}_{key}", '%.2f (%.2f)'%(np.mean(accs_all_exps[metric][key])*100, np.std(accs_all_exps[metric][key])*100)] for metric in eval_method]
        table = wandb.Table(data=data, columns = ["Evaluation", "Accuracy"])
        wandb.log({f"Final Results {key}" : table})
        
    wandb.finish()


if __name__ == '__main__':
    main()


