import torch

def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2



if __name__=="__main__":
    import chamfer3D.dist_chamfer_3D

    cham3D = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    points1 = torch.rand(4, 100, 3).cuda()
    points2 = torch.rand(4, 200, 3, requires_grad=True).cuda()
    dist1, dist2, idx1, idx2= cham3D(points1, points2)
    f = fscore(dist1, dist2)
    print(f)