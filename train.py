import os
import torch
import datetime
from dataset import VMVPH5
from models import network
from models.utils import fps_subsample
from utils.train_utils import calc_cd, setup_logger


def train_step1(model_dir, N=8192, num_offset=4, load_model_dir=None, epoches=25, initial_learning_rate=1e-3):
    os.makedirs('./log/model/%s/'%model_dir, exist_ok=True)
    logger = setup_logger('train_step1', './log/model/%s/train.log'%model_dir)
    dataset = VMVPH5(train=True, npoints=N)
    dataset_test = VMVPH5(train=False, npoints=N)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32,
                                                  shuffle=False, num_workers=8)
    model = network.Offset_Model(num_offset).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    if load_model_dir != None:
        model.load_state_dict(torch.load(os.path.join('./log', 'model', load_model_dir, 'best_params.pth')))
    
    min_loss = 1e7
    best_epoch = -1
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    for epoch in range(1, 1+epoches):
        # ---------------------- Train ----------------------
        cd0=[]
        cd1=[]
        model = model.train()
        print(model.training)
        for i, data in enumerate(dataloader, 0):
            _, view_points, inputs, gt_pc = data
            inputs = inputs.float().cuda()
            view_points = view_points.float().cuda()
            gt_pc = gt_pc.float().cuda()

            pc_list, dist_list = model(inputs, view_points)

            _, cd_loss0 = calc_cd(pc_list[0], gt_pc, train=True)
            _, cd_loss1 = calc_cd(pc_list[1], gt_pc, train=True)

            total_loss = cd_loss0 + cd_loss1
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            cd0_item = cd_loss0.item()*1e4
            cd1_item = cd_loss1.item()*1e4

            cd0.append(cd0_item)
            cd1.append(cd1_item)

            if i%100==0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, '[%03d/%03d] iter%04d: cd0: %.2f cd1: %.2f'%(epoch, epoches, i, cd0_item, cd1_item))
        cd0 = sum(cd0)/len(cd0)
        cd1 = sum(cd1)/len(cd1)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, '[%03d/%03d] train cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
        logger.info('[%03d/%03d] train cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))

        # ---------------------- Valid ----------------------
        model = model.eval()
        print(model.training)
        cd0=[]
        cd1=[]
        with torch.no_grad():
            if epoch%1==0:
                for i, data in enumerate(dataloader_test, 0):
                    _, view_points, inputs, gt_pc = data
                    inputs = inputs.float().cuda()
                    view_points = view_points.float().cuda()
                    gt_pc = gt_pc.float().cuda()
                    
                    pc_list, dist_list = model(inputs, view_points)

                    _, cd_loss0 = calc_cd(pc_list[0], gt_pc, train=True)
                    _, cd_loss1 = calc_cd(pc_list[1], gt_pc, train=True)
                    
                    cd0_item = cd_loss0.item()*1e4
                    cd1_item = cd_loss1.item()*1e4

                    cd0.append(cd0_item)
                    cd1.append(cd1_item)
                    
                cd0 = sum(cd0)/len(cd0)
                cd1 = sum(cd1)/len(cd1)
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, '[%03d/%03d] valid cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
                if cd1<min_loss:
                    min_loss = cd1
                    best_epoch = epoch
                    torch.save(model.state_dict(),os.path.join('./log', 'model', model_dir, 'best_params.pth'))
                    print("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
                logger.info('[%03d/%03d] valid cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
                logger.info("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
        
        # ---------------------- scheduler ----------------------
        scheduler.step()


def train_step2(model_dir, N=16384, load_model_dir=None, epoches=15, initial_learning_rate=1e-3):
    os.makedirs('./log/model/%s/'%model_dir, exist_ok=True)
    logger = setup_logger('train_step2', './log/model/%s/train.log'%model_dir)
    dataset = VMVPH5(train=True, npoints=N)
    dataset_test = VMVPH5(train=False, npoints=N)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32,
                                                  shuffle=False, num_workers=8)
    model = network.SCPS_Model(up_factors=[8]).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])

    state_dict = torch.load(os.path.join('./log', 'model', 'Offset_Model', 'best_params.pth'))
    state_dict = {k[7:]:v for k,v in state_dict.items()}
    model.module.offset_model.load_state_dict(state_dict)
    for p in model.module.offset_model.parameters():
        p.requires_grad = False

    if load_model_dir != None:
        model.load_state_dict(torch.load(os.path.join('./log', 'model', load_model_dir, 'best_params.pth')))
    
    min_loss = 1e7
    best_epoch = -1
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    for epoch in range(1, 1+epoches):
        # ---------------------- Train ----------------------
        cd0=[]
        cd1=[]
        model = model.train()
        print(model.training)
        for i, data in enumerate(dataloader, 0):
            _, view_points, inputs, gt_pc = data
            inputs = inputs.float().cuda()
            view_points = view_points.float().cuda()
            gt_pc = gt_pc.float().cuda()

            pc_list = model(inputs, view_points)
            
            subsampled_gt_pc = fps_subsample(gt_pc, pc_list[1].shape[1])
            _, cd_loss0 = calc_cd(pc_list[1], subsampled_gt_pc, train=True)
            _, cd_loss1 = calc_cd(pc_list[2], gt_pc, train=True)

            total_loss = cd_loss0 + cd_loss1
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            cd0_item = cd_loss0.item()*1e4
            cd1_item = cd_loss1.item()*1e4
            cd0.append(cd0_item)
            cd1.append(cd1_item)
            if i%100==0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, '[%03d/%03d] iter%04d: cd0: %.2f cd1: %.2f'%(epoch, epoches, i, cd0_item, cd1_item))
        cd0 = sum(cd0)/len(cd0)
        cd1 = sum(cd1)/len(cd1)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, '[%03d/%03d] train cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
        logger.info('[%03d/%03d] train cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))

        # ---------------------- Valid ----------------------
        model = model.eval()
        print(model.training)
        cd0=[]
        cd1=[]
        with torch.no_grad():
            if epoch%1==0:
                for i, data in enumerate(dataloader_test, 0):
                    _, view_points, inputs, gt_pc = data
                    inputs = inputs.float().cuda()
                    view_points = view_points.float().cuda()
                    gt_pc = gt_pc.float().cuda()
                    
                    pc_list = model(inputs, view_points)

                    subsampled_gt_pc = fps_subsample(gt_pc, pc_list[1].shape[1])
                    _, cd_loss0 = calc_cd(pc_list[1], subsampled_gt_pc, train=True)
                    _, cd_loss1 = calc_cd(pc_list[2], gt_pc, train=True)
                    
                    cd0_item = cd_loss0.item()*1e4
                    cd1_item = cd_loss1.item()*1e4

                    cd0.append(cd0_item)
                    cd1.append(cd1_item)
                    
                cd0 = sum(cd0)/len(cd0)
                cd1 = sum(cd1)/len(cd1)
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, '[%03d/%03d] valid cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
                if cd1<min_loss:
                    min_loss = cd1
                    best_epoch = epoch
                    torch.save(model.state_dict(),os.path.join('./log', 'model', model_dir, 'best_params.pth'))
                    print("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
                logger.info('[%03d/%03d] valid cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
                logger.info("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
        
        # ---------------------- scheduler ----------------------
        scheduler.step()


def train_step3(model_dir, N=16384, load_model_dir=None, epoches=5, initial_learning_rate=1e-4):
    os.makedirs('./log/model/%s/'%model_dir, exist_ok=True)
    logger = setup_logger('train_step3', './log/model/%s/train.log'%model_dir)
    dataset = VMVPH5(train=True, npoints=N)
    dataset_test = VMVPH5(train=False, npoints=N)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32,
                                                  shuffle=False, num_workers=8)
    model = network.SCPS_Model(up_factors=[8]).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])

    model.load_state_dict(torch.load(os.path.join('./log', 'model', 'SCPS_tmp', 'best_params.pth')))
    if load_model_dir != None:
        model.load_state_dict(torch.load(os.path.join('./log', 'model', load_model_dir, 'best_params.pth')))
    
    min_loss = 1e7
    best_epoch = -1
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    for epoch in range(1, 1+epoches):
        # ---------------------- Train ----------------------
        cd0=[]
        cd1=[]
        model = model.train()
        print(model.training)
        for i, data in enumerate(dataloader, 0):
            _, view_points, inputs, gt_pc = data
            inputs = inputs.float().cuda()
            view_points = view_points.float().cuda()
            gt_pc = gt_pc.float().cuda()

            pc_list = model(inputs, view_points)
            
            subsampled_gt_pc = fps_subsample(gt_pc, pc_list[1].shape[1])
            _, cd_loss0 = calc_cd(pc_list[1], subsampled_gt_pc, train=True)
            _, cd_loss1 = calc_cd(pc_list[2], gt_pc, train=True)

            total_loss = cd_loss0 + cd_loss1
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            cd0_item = cd_loss0.item()*1e4
            cd1_item = cd_loss1.item()*1e4
            cd0.append(cd0_item)
            cd1.append(cd1_item)
            if i%100==0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, '[%03d/%03d] iter%04d: cd0: %.2f cd1: %.2f'%(epoch, epoches, i, cd0_item, cd1_item))
        cd0 = sum(cd0)/len(cd0)
        cd1 = sum(cd1)/len(cd1)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, '[%03d/%03d] train cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
        logger.info('[%03d/%03d] train cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))

        # ---------------------- Valid ----------------------
        model = model.eval()
        print(model.training)
        cd0=[]
        cd1=[]
        with torch.no_grad():
            if epoch%1==0:
                for i, data in enumerate(dataloader_test, 0):
                    _, view_points, inputs, gt_pc = data
                    inputs = inputs.float().cuda()
                    view_points = view_points.float().cuda()
                    gt_pc = gt_pc.float().cuda()
                    
                    pc_list = model(inputs, view_points)

                    subsampled_gt_pc = fps_subsample(gt_pc, pc_list[1].shape[1])
                    _, cd_loss0 = calc_cd(pc_list[1], subsampled_gt_pc, train=True)
                    _, cd_loss1 = calc_cd(pc_list[2], gt_pc, train=True)
                    cd0_item = cd_loss0.item()*1e4
                    cd1_item = cd_loss1.item()*1e4
                    cd0.append(cd0_item)
                    cd1.append(cd1_item)
                    
                cd0 = sum(cd0)/len(cd0)
                cd1 = sum(cd1)/len(cd1)
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(time, '[%03d/%03d] valid cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
                if cd1<min_loss:
                    min_loss = cd1
                    best_epoch = epoch
                    torch.save(model.state_dict(),os.path.join('./log', 'model', model_dir, 'best_params.pth'))
                    print("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
                logger.info('[%03d/%03d] valid cd0: %.2f cd1: %.2f'%(epoch, epoches, cd0, cd1))
                logger.info("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
        
        # ---------------------- scheduler ----------------------
        scheduler.step()



if __name__ == '__main__':
    train_step1(model_dir='Offset_Model')
    train_step2(model_dir='SCPS_tmp')
    train_step3(model_dir='SCPS')





































