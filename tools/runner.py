from socket import PF_CAN
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

str2int = {"Crayon": 1, "Doraemon": 2, "Elephant": 3, "MashiMaro": 4, "Mermaid": 5, "Minions": 6, "PeppaPig": 7, "Pikachu": 8, "Smurf": 9, "SpongeBob": 10, "bicycle": 11, "butterfly": 12, "car": 13, "cup": 14, "dinosaur": 15, "dolphin": 16, "house": 17, "kartell": 18, "mickey": 19, "pigeon": 20, "plane": 21, "tree": 22, "umbrella": 23, "watch": 24, "Snowman":25, "donald":26, "Garfield":27, "Twilight":28}
DisplayDebug = True
cls_backward_weight = 10
cls_lossRecord_weight = 1
AddPointcloudMatch = True

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    ChamferDisL1 = ChamferDistanceL1() # These following two metrics function are defined in _init_.py
    ChamferDisL2 = ChamferDistanceL2()
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['ClsLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

            ret=base_model(partial)

            rt=(ret[0],ret[1]) #coarse_point_cloud, inp_sparse
            
            # added
            # 32x28
            cls_pred = ret[2]

            y_cls_true = []
            for t in taxonomy_ids:
                y_cls_true.append(str2int[t])
            y_cls_true = torch.FloatTensor(y_cls_true).cuda()

            # 32x1 -> 32x28
            y_cls_true = torch.nn.functional.one_hot(y_cls_true.to(torch.int64), num_classes=28).float()
            # print("="*50)
            # print(y_cls_true.shape)
            # print(cls_pred.shape)


            # print('Coarse point Shape:')
            # print(ret[0].shape)
            # print('Dense point Shape:')
            # print(ret[1].shape)
            # # print('y_cls_true:')
            # # print(y_cls_true)
            # print('cls_pred.shape & cls_pred:')
            # print(cls_pred.shape)
            # print(torch.argmax(cls_pred))

            # print("="*50)
            # print(cls_pred)
            # print(y_cls_true)

            cls_loss_Function = nn.CrossEntropyLoss()
            cls_loss = cls_loss_Function(cls_pred, y_cls_true)
          

            # cls_loss=0

            _loss = cls_loss 
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([cls_loss * cls_lossRecord_weight])
            else:
                losses.update([cls_loss* cls_lossRecord_weight])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
               train_writer.add_scalar('Loss/Epoch/Cls', cls_loss.item(), epoch)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f, ZHOU loss: %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr'], cls_loss), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Cls',losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    train_writer.close()
    val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['ClsLoss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 32

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            inp_sparse = ret[1]
            cls_pred = ret[2]

            # added

            # y_cls_true = torch.Tensor(str2int[taxonomy_id] - 1)
            # cls_loss = F.binary_cross_entropy_with_logits(torch.argmax(cls_pred), y_cls_true)

            y_cls_true = []
            for t in taxonomy_ids:
                y_cls_true.append(str2int[t])
            y_cls_true = torch.FloatTensor(y_cls_true).cuda()
            # if DisplayDebug:
                # print('='*15, y_cls_true.shape)

            # 32x1
            y_cls_true = torch.nn.functional.one_hot(y_cls_true.to(torch.int64), num_classes=28).float()
            
            cls_loss = (cls_pred, y_cls_true)
            
            if AddPointcloudMatch:
                 sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                 sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                 test_losses.update([cls_loss * cls_lossRecord_weight])
            else:
                 test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,cls_loss * cls_lossRecord_weight])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(coarse_points, gt, cls_pred, y_cls_true)
            # _metrics = [dist_utils.reduce_tensor(item, args) for item in _metrics]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            # if val_writer is not None and idx % 1000 == 0:
            if False:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

                dense = inp_sparse.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % 1000 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/cartoon_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Cls', test_losses.avg(0), epoch)

        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
   
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    # The test_losses here is just a recorder
    if AddPointcloudMatch:
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2','ClsLoss'])
    else:
        test_losses = AverageMeter(['ClsLoss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 32
    cls_pred_record = []
    cls_true_record = []
    category_pred_record = []
    category_true_record = []


    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                inp_points = ret[1]

                # added
                cls_pred = ret[2]
                # if DisplayDebug:
                #     print('='*15, taxonomy_ids)
                #     print(cls_pred)

                y_cls_true = []
                
                for t in taxonomy_ids:
                    y_cls_true.append(str2int[t])
                y_cls_true = torch.FloatTensor(y_cls_true).cuda()

                # 32x28
                y_cls_true = torch.nn.functional.one_hot(y_cls_true.to(torch.int64), num_classes=28).float()
                
                cls_loss_Function = nn.CrossEntropyLoss()
                cls_loss = cls_loss_Function(cls_pred, y_cls_true)
                if AddPointcloudMatch:
                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,cls_loss * cls_lossRecord_weight])
                else:
                   test_losses.update([cls_loss * cls_lossRecord_weight])
                _metrics = Metrics.get(coarse_points,gt,cls_pred,y_cls_true)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        # if dataset_name == 'KITTI':
        #     return

        # Record the cls_pred and cls_true
        for i in range(cls_pred.shape[0]):
            cls_pred_record.append(cls_pred[i].tolist() )
            cls_true_record.append(y_cls_true[i].tolist())

        # record the predicted cls result and true cls result as a category number according to index
        for i in range(cls_pred.shape[0]):
            # find the index of the max value and index is the category number
            category_pred_record.append(cls_pred[i].argmax().tolist())
            category_true_record.append(y_cls_true[i].argmax().tolist())

        
        for _,v in category_metrics.items():
            print('Test_metrics Updated!')
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)
    
    print_log('============================ CLS CON Matrix ============================',logger=logger)
    print(cls_true_record.shape)
    confusion_matrix = multilabel_confusion_matrix(np.array(cls_true_record), np.array(cls_pred_record))

    # Print testing results
    shapenet_dict = json.load(open('./data/cartoon_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Save the cls_pred and cls_true into a txt file
    with open('cls_pred.txt', 'w') as f:
        for item in cls_pred_record:
            f.write("%s\n" % item)
    with open('cls_true.txt', 'w') as f:
        for item in cls_true_record:
            f.write("%s\n" % item)
    with open('confusion_matrix.txt', 'w') as f:
        for item in confusion_matrix:
            f.write("%s\n" % item)
    with open('category_pred.txt', 'w') as f:
        for item in category_pred_record:
            f.write("%s\n" % item)
    with open('category_true.txt', 'w') as f:
        for item in category_true_record:
            f.write("%s\n" % item)
    return
