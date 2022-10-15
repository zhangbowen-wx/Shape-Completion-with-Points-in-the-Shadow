import sys
import os
import torch
import logging
import numpy as np
sys.path.append(os.path.join("./utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore


label_map ={
            0: "airplane",
            1: "cabinet",
            2: "car",
            3: "chair",
            4: "lamp",
            5: "sofa",
            6: "table",
            7: "watercraft",

            8: "bed",
            9: "bench",
            10: "bookself",
            11: "bus",
            12: "guitar",
            13: "motorbike",
            14: "pistol",
            15: "skateboard",
        }


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s')  
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class DistributionCounter():
    def __init__(self, string, value_list, interval_list=(1,3,5,10,15)):
        self.string = string
        self.value_list = value_list
        self.interval_list = interval_list

    def add(self, value):
        if isinstance(value, list):
            self.value_list += value
        else:
            self.value_list.append(value)

    def cal_distribution(self):
        print("-"*30)
        print(self.string)
        print("-"*30)
        total_number = len(self.value_list)
        total_value = sum(self.value_list)
        print("Average value: %.3f / %d = %.3f"%(total_value, total_number, (total_value/total_number)))
        if self.interval_list != None:
            numbers = [0]*(len(self.interval_list)+1)
            values = [0]*(len(self.interval_list)+1)
            for item in self.value_list:
                ok = 0
                for i, interval in enumerate(self.interval_list):
                    if item<interval:
                        numbers[i]+=1
                        values[i]+=item
                        ok=1
                        break
                if ok==0:
                    numbers[-1]+=1
                    values[-1]+=item
            print("Numbers")
            for i in range(len(numbers)):
                if i==0:
                    print('[%.1f, %.1f): %d/%d=%d%%'%(0,self.interval_list[0],numbers[0],total_number,numbers[0]/total_number*100))
                elif i==(len(numbers)-1):
                    print('[%.1f, %s): %d/%d=%d%%'%(self.interval_list[-1],'INF',numbers[-1],total_number,numbers[-1]/total_number*100))
                else:
                    print('[%.1f, %.1f): %d/%d=%d%%'%(self.interval_list[i-1],self.interval_list[i],numbers[i],total_number,numbers[i]/total_number*100))
            print("-"*20)
            print("Values")
            for i in range(len(values)):
                if i==0:
                    print('[%.1f, %.1f): %.3f/%.3f=%d%%'%(0,self.interval_list[0],values[0],total_value,values[0]/total_value*100))
                elif i==(len(values)-1):
                    print('[%.1f, %s): %.3f/%.3f=%d%%'%(self.interval_list[-1],'INF',values[-1],total_value,values[-1]/total_value*100))
                else:
                    print('[%.1f, %.1f): %.3f/%.3f=%d%%'%(self.interval_list[i-1],self.interval_list[i],values[i],total_value,values[i]/total_value*100))
            print("-"*20)
        print()


class CategoryAverageLossCalculator():
    def __init__(self, category_num=16):
        """
        1st column is value
        2nd column is number
        """
        self.array = np.zeros((category_num,2))
        self.category_num = category_num
    
    def add(self, label_list, value_list, number=1):
        if isinstance(label_list, list) and isinstance(value_list, list):
            if len(label_list) == len(value_list):
                for i in range(len(label_list)):
                    self.array[int(label_list[i]), 0] += value_list[i]
                    self.array[int(label_list[i]), 1] += number
            else:
                raise RuntimeError("Expect the length of label_list and value_list to be the same, but got %d and %d."(len(label_list),len(value_list)))
        else:
            raise ValueError("Expect label_list and value_list to be list, but got "+type(label_list)+" and "+type(value_list)+'.')
    
    def cal_average(self):
        for i in range(self.category_num):
            print(("%s"%(label_map[i])).ljust(10), ": %.3f/%d = %.3f"%(self.array[i,0], self.array[i,1], self.array[i,0]/self.array[i,1]))


def calc_cd(output, gt, train=False, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if train:
        cd_p = cd_p.mean()
        cd_t = cd_t.mean()
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        if train:
            f1 = f1.mean()
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t


def calc_dcd(output, gt, alpha=1000, n_lambda=1, non_reg=False, train=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2,idx1, idx2 = cham_loss(gt, output)
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    batch_size, n_x, _ = output.shape
    batch_size, n_gt, _ = gt.shape
    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x
    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2
    if train:
        return loss.mean()
    else:
        return loss


def resample_pcd(pcd, n):
    """
    Drop or duplicate points so that pcd has exactly n points
    Args:
        pcd: Tensor, (N, 3)
        n: Int
    return:
        pcd: Tensor, (n, 3)
    """
    device = pcd.device
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    idx = torch.from_numpy(idx).to(device)
    return pcd[idx[:n], :]


def calc_average_neighboring_points_number(query_xyz, xyz, d=0.01):
    """
    Args:
        query_xyz: Tensor, (b, n, 3)
        xyz: Tensor, (b, m, 3)
        d: Float
    Return:
        avg: Float
    """
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    _, dist, _, _ = cham_loss(query_xyz, xyz) 
    batch_size, n_q, _ = query_xyz.shape
    device = dist.device
    mask = dist<d**2
    avg = []
    for b in range(batch_size):
        idx = mask[b].nonzero().squeeze(1)
        number = idx.shape[0]
        avg.append(torch.tensor(number/n_q).to(device))
    avg = torch.stack(avg)
    return avg


def calc_vdd(pre_vp, gt_vp, train=True):
    """
    Calculate the view direction discrepency.
    Args:
        pre_vp: (b, 3)
        gt_vp: (b, 3)
    return:
        vdd: (b, 1)
    """
    vdd = []
    batchsize, _ = pre_vp.shape
    for b in range(batchsize):
        vdd.append(torch.acos(torch.dot(pre_vp[b], gt_vp[b]))/3.1415926535*180)
    vdd = torch.stack(vdd)
    if train:
        return vdd.mean()
    else:
        return vdd


def seperate_pc_by_distance(inputs, gt, d=0.01):
    """
    Args:
        inputs: (1, m, 3)
        gt: (1, n, 3)
        d: float
    Returns:
        pc_part1: (1, n1, 3) near points
        pc_part2: (1, n2, 3) far points
        n1+n2=n
    """
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    _, dist, _, _ = cham_loss(inputs, gt)
    mask1 = dist[0]<(d**2)
    mask2 = dist[0]>=(d**2)
    idx_part1 = mask1.nonzero().squeeze(1)
    idx_part2 = mask2.nonzero().squeeze(1)
    pc_part1 = gt[:,idx_part1,:]
    pc_part2 = gt[:,idx_part2,:]
    return pc_part1, pc_part2


def get_2part_pc(inputs, gt, outputs, d=0.01):
    """
    Args:
        inputs: Tensor, (1, n1, 3)
        gt: Tensor, (1, n2, 3)
        outputs: Tensor, (1, n3, 3)
    Return:
        visible_part: (1, n4, 3)
        missing_part: (1, n5, 3)
        output_part1: (1, n6, 3)
        output_part2: (1, n7, 3)
    """
    device = inputs.device
    visible_part, missing_part = seperate_pc_by_distance(inputs, gt, d)
    if missing_part.shape[1]==0:
        output_part1 = outputs
        output_part2 = torch.zeros(1,0,3).to(device)
        return visible_part, missing_part, output_part1, output_part2
    else:
        cham_loss = dist_chamfer_3D.chamfer_3DDist()
        _, dist1, _, _ = cham_loss(inputs, outputs)
        _, dist2, _, _ = cham_loss(missing_part, outputs)
        mask1 = dist1[0]<dist2[0]
        mask2 = dist2[0]<dist1[0]
        idx1 = mask1.nonzero().squeeze(1)
        idx2 = mask2.nonzero().squeeze(1)
        output_part1 = outputs[:, idx1, :]
        output_part2 = outputs[:, idx2, :]
        return visible_part, missing_part, output_part1, output_part2


def calc_split_cd(inputs, gt, outputs, train=True, d=0.01):
    """
    Args:
        inputs: Tensor, (b, n1, 3)
        gt: Tensor, (b, n2, 3)
        outputs: Tensor, (b, n3, 3)
    Return:
        loss_part1
        loss_part2
    """
    batch_size = inputs.shape[0]
    device = inputs.device
    loss_part1=[]
    loss_part2=[]
    for b in range(batch_size):
        visible_part, missing_part, output_part1, output_part2 = get_2part_pc(inputs[b].unsqueeze(0), gt[b].unsqueeze(0), outputs[b].unsqueeze(0), d)
        loss_part1.append(calc_cd(visible_part, output_part1, train=True)[1])
        if (missing_part.shape[1]==0):
            loss_part2.append(torch.tensor(0).to(device))
        else:
            loss_part2.append(calc_cd(missing_part, output_part2, train=True)[1])
    loss_part1 = torch.stack(loss_part1)
    loss_part2 = torch.stack(loss_part2)
    if train:
        return loss_part1.mean(), loss_part2.mean()
    else:
        return loss_part1, loss_part2








