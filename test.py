from symbol import import_from
import torch
from dataset import VMVPH5
import os
from models import network
from utils.train_utils import *


def test_model(model_dir, N=16384, bs=32):
    dataset_test = VMVPH5(train=False, npoints=N)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs,
                                                  shuffle=False, num_workers=8)
    model = network.SCPS_Model(up_factors=[8]).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.load_state_dict(torch.load(os.path.join('./log', 'model', model_dir, 'best_params.pth')))
    model = model.eval()
    cd_t_counter = DistributionCounter(string="cd_t", value_list=[])
    f1_counter   = DistributionCounter(string="f1 score", value_list=[], interval_list=None)
    dcd_counter   = DistributionCounter(string="dcd", value_list=[], interval_list=[0.1, 0.3, 0.5, 0.7])
    cd_part1_counter = DistributionCounter(string="cd part1", value_list=[])
    cd_part2_counter = DistributionCounter(string="cd part2", value_list=[])
    gt_avg_num_counter = DistributionCounter(string="gt_avg_num", value_list=[])
    output_avg_num_counter = DistributionCounter(string="output_avg_num", value_list=[])
    category_loss_calculator = CategoryAverageLossCalculator()
    category_f1_calculator = CategoryAverageLossCalculator()
    category_dcd_calculator = CategoryAverageLossCalculator()
    category_cd_part1_calculator = CategoryAverageLossCalculator()
    category_cd_part2_calculator = CategoryAverageLossCalculator()
    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            label, view_points, inputs, gt = data
            view_points = view_points.float().cuda()
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            outputs = model(inputs, view_points)[2]

            _, cd_t, f1 = calc_cd(outputs, gt, train=False, calc_f1=True)
            dcd = calc_dcd(outputs, gt, train=False)
            cd_part1, cd_part2 = calc_split_cd(inputs, gt, outputs, train=False)
            num1 = calc_average_neighboring_points_number(inputs, gt)
            num2 = calc_average_neighboring_points_number(inputs, outputs)
            cd_t = cd_t*1e4
            cd_part1 = cd_part1*1e4
            cd_part2 = cd_part2*1e4
            cd_t_counter.add(cd_t.tolist())
            f1_counter.add(f1.tolist())
            dcd_counter.add(dcd.tolist())
            cd_part1_counter.add(cd_part1.tolist())
            cd_part2_counter.add(cd_part2.tolist())
            gt_avg_num_counter.add(num1.tolist())
            output_avg_num_counter.add(num2.tolist())
            category_loss_calculator.add(label.tolist(), cd_t.tolist())
            category_f1_calculator.add(label.tolist(), f1.tolist())
            category_dcd_calculator.add(label.tolist(), dcd.tolist())
            category_cd_part1_calculator.add(label.tolist(), cd_part1.tolist())
            category_cd_part2_calculator.add(label.tolist(), cd_part2.tolist())
        cd_t_counter.cal_distribution()
        category_loss_calculator.cal_average()
        f1_counter.cal_distribution()
        category_f1_calculator.cal_average()
        dcd_counter.cal_distribution()
        category_dcd_calculator.cal_average()
        cd_part1_counter.cal_distribution()
        category_cd_part1_calculator.cal_average()
        cd_part2_counter.cal_distribution()
        category_cd_part2_calculator.cal_average()
        gt_avg_num_counter.cal_distribution()
        output_avg_num_counter.cal_distribution()
    


if __name__ == '__main__':
    test_model(model_dir="SCPS")




