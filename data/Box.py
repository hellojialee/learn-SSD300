# coding:utf-8
from __future__ import print_function, division
import torch

import math, itertools


class PriorBox(object):
    """
    每个输入神经网络的图片都是固定大小的300X300，这里的代码 只是针对　一张　图片  !!
    这里针对每个图片产生prior box，并与label进行match
    """

    def __init__(self, opt):
        """
        opt: Config 记录参数的类
        """
        scale = opt.img_size  # 300
        # 产生相对位置的cell大小 size 0-1之间
        # [0.02666666666666667, 0.05333333333333334, 0.10666666666666667, 0.21333333333333335, 0.3333333333333333, 1.0]
        # cell size是每个feature map中每个cell对应原图的大小，steps是每个cell占据原始图像的比例
        steps = [s / scale for s in opt.cell_size]  # cell_size = (8, 16, 32, 64, 100, 300)
        # 产生相对的prior box 的大小 0-1之间
        # [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
        hw_sizes = [s / scale for s in opt.prior_box_hw]
        # 预先设计好的h w之比
        aspect_ratios = opt.aspect_ratios
        # MultiBoxLayer输出的各级feature map大小
        feature_map_sizes = opt.feature_map_sizes

        num_layers = len(feature_map_sizes)

        # prior box
        boxes = []
        # 对每个feature map都要产生prior box，下面的循环是按SSD的几个阶段layers依次进行处理的
        for i in range(num_layers):
            fmsize = feature_map_sizes[i]
            # 产生从(0, 0)到(fmsize-1, fmsize-1)
            for h, w in itertools.product(range(fmsize), repeat=2):
                # 得到centerX centerY 这里是相对位置，百分数. 这里产生的h,w是按照行和列的顺序排列的
                cx = (w + 0.5) * steps[i]  # notice: 考虑量化误差，方格中心位置其实要　+0.5
                cy = (h + 0.5) * steps[i]

                # 原始尺寸的prior box
                s = hw_sizes[i]  # 每一个阶段的feature map上都设置不同尺寸基准的box
                boxes.append((cx, cy, s, s))  # 相对值，在0~1
                # 每个特征图都设置了两个长宽比为1但大小不同的正方形先验框，这一个，还有下面那一个

                s = math.sqrt(hw_sizes[i] * hw_sizes[i + 1])  # 非矩阵运算，计算单个值math比numpy的快
                # 默认情况下，每个特征图会有一个 a_r=1 且尺度为 s_k 的先验框，除此之外，还会设置一个尺度为 sk'=sqrt{sk * s_(k+1)} 且 a_r=1 的先验框
                boxes.append((cx, cy, s, s))

                s = hw_sizes[i]
                # 不同比例的box大小 总的来说即使使用了不同的ratio box所占的面积仍是一样的
                for ar in aspect_ratios[i]:  # 循环到某一层，添加sqrt(2)或者sqrt(3). debug一下就明白了
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))  # 这些都是归一化0~1的值

        # (num, 4)
        # 这里实现的和原论文中一样
        # num = 38*38*(2+2) + 19*19*(2+4) + 10*10*(2+4) + 5*5*(2+4) + 3*3*4 + 1*1*4 = 8723
        # 网络结构和超参数确定后，default box就固定了
        self.default_boxes = torch.Tensor(boxes)  # 把list类型的boxes转换成 Tensor.

    def iou(self, box1, box2):
        """
        使用pytorch实现的

        box1: tensor, (N, 4) (xmin,ymin,xmax,ymax)
        box2: tensor, (M, 4) (xmin,ymin,xmax,ymax)
        return: tensor, iou (N, M) 一个矩阵
        计算两个box之间的IOU iou[i, j]第i个box1中的box和第j个box2中的box之间的iou 0-1
        """
        N = box1.size(0)
        M = box2.size(0)

        # left top 左上角 由于是从左上角开始计坐标的所以右下角的坐标大
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # print('in Box iou, lt[0, 0, :]', lt[0, 0, :])
        # right behind 右下角
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # print('in Box iou, rb[0, 0, :]', rb[0, 0, :])

        # 右下角减去左上角得到w h
        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        # print('in Box iou, inter[0, 0]', inter[0, 0])

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        # print('in Box iou, area1[0, 0]', area1[0, 0])
        # print('in Box iou, area2[0, 0]', area2[0, 0])

        iou = inter / (area1 + area2 - inter)
        return iou

    def match(self, boxes, classes, threshold=0.5):
        """
        这里的匹配原则与知乎上说的有点不一样，但是应该效果差异很小
        offset的编码如下，具体自己查
         .. math::
          l^{cx} = (b^{cx} - d^{cx})/d^w, l^{cy} = (b^{cy} - d^{cy})/d^h

          l^{w} = \log(b^{w}/d^w), l^{h} = \log(b^{h}/d^h)
        Args:
            boxes: tensor, (num, 4) 图片里标注的bounding box，数值已经在0-1之间！！每个box格式为(xmin,ymin,xmax,ymax)
            (xmin,ymin,xmax,ymax)是相对于300的归一化之后的坐标值. num是
            classes: tensor, (num,)
            threshold: double, Jaccard阈值

        return:
        boxes: tensor, (num, 8732, 4)
        classes: tensor, (8732,)

        计算出每个default box和每个label box之间的iou
        将Jaccard值大于0.5的标为这个label box代表的类别
        Jaccard(A, B) = AB / (A+B-AB)
        小于0.5的就标记为背景类别0

        对于位置，计算完iou之后将每个default box的最匹配的label box的值赋给这个default box   !!! 并且用这个值监督训练选中的default box

        这里没有通过去除而是标记来达到匹配的效果
        """
        default_boxes = self.default_boxes.clone()  # 之前已经将self.default_boxes转换成了tensor  torch.Size([8732, 4])
        # clone: Unlike copy_(), this function is recorded in the computation graph. Gradients propagating to the
        # cloned tensor will propagate to the original tensor.
        # num_default_boxes = default_boxes.size(0)
        # num_objs = boxes.size(0)
        # print('in match() default_box[0:10, :]: ', default_boxes[0:10, :])

        iou = self.iou(  # [num,8732] num是label的个数，　iou计算函数中返回iou是 (N, M) 一个矩阵，这里每个label都与8732个default box计算iou
            boxes,
            # 从 (centerX, centerY, h, w)转换到(xmin,ymin,xmax,ymax)
            # cscy - wh/2 = xminymin
            # cxcy + wh/2 = xmaxymax
            torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,
                       default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1)  # 按照行的方向进行拼接堆叠
        )
        # 将每个label box和default box进行计算iou
        # 计算出每个default box和哪个label box之间的iou最大
        iou, max_idx = iou.max(0)  # (1, 8732)
        max_idx.squeeze_(0)  # (8732,)
        iou.squeeze_(0)  # (8732,)
        # print('in match() iou[:10]: ', iou[:10])

        # boxes的长度可能只有几个，也就是一张图片里可能只有几个物体
        # 在pytorch中用long类型并且这个Tensor的长度更大的作为索引
        # 会自动将原Tensor扩大
        # boxes现在存的是仍是原boxes内容，只不过长度变成了8732
        # boxes[i]就是与第i个default boxJ匹配accard最大的label box的值
        boxes = boxes[max_idx]  # 数值已经在0-1之间每个box格式为(xmin,ymin,xmax,ymax)
        variances = [0.1, 0.2]
        # 从 (xmin,ymin,xmax,ymax)转换到(centerX, centerY, h, w)
        # 这里好像是因为用SmoothL1Loss所以在这里就进行了处理
        # 下面的处理就是边界框的编码（encode），　offset
        # boxes[:, :2] + boxes[:, 2:])/2 得到的就是center，bounding box左上和右下的中点
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # (8732, 2)　boxes是标注数据，default_box与网络结构有关，是固定的。
        cxcy /= variances[0] * default_boxes[:, :2]  # default box 格式为0-1归一化之后的(cx, cy, s, s)
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # (8732, 2)
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([cxcy, wh], 1)  # (8732, 4)  # 注意拼接的axis方向　 (8732, 2) + (8732, 2) --> (8732, 4)
        # print('in Box classes:', classes)
        # 同样将classes进行扩充，conf[i]就是第i个default box最接近的那个类
        conf = 1 + classes[max_idx]  # (8732,) 背景被设置为0， 所以这里加一
        # 也就是说每个类别就与背景进行区分，一个值代表是否是背景还是这个类的概率
        conf[iou < threshold] = 0  # 背景类被设置为0，　与label box的iou小于0.5的default box视为背景
        # print('prior match', conf.sum())
        return loc, conf

    def nms(self, bboxes, scores, threshold=0.5, mode='union'):
        '''Non maximum suppression.

        Args:
          bboxes: (tensor) bounding boxes, sized [N,4]. N是检测出来的一张图片中的box的数目
          scores: (tensor) bbox scores, sized [N,]. 对应的N个box的类别概率
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        assert bboxes.size(0) == scores.size(0)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=float(x1[i]))
            yy1 = y1[order[1:]].clamp(min=float(y1[i]))
            xx2 = x2[order[1:]].clamp(max=float(x2[i]))
            yy2 = y2[order[1:]].clamp(max=float(y2[i]))

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)

    def convert_result(self, loc, conf):
        """
        对SSD预测结果输出解码
            .. math::
                b^{cx}=d^w (variance[0]*l^{cx}) + d^{cx}, b^{cy}=d^y (variance[1]*l^{cy}) + d^{cy} \n
                b^{w}=d^w \exp(variance[2]*l^{w}), b^{h}=d^h \exp(variance[3]*l^{h})
        loc: tensor, 预测的位置结果 (8732, 4) (cx, cy, h, w)
        conf: tensor, 预测的类别结果(8732, 21) 以上这两个都是MultiBoxLayer的输出
        return:
          boxes: tensor, 预测的有效的位置结果 (num, 4)，num是预测出的sample的数目
          labels: tensor, 预测的类别(num,1)
        进行nms，并将预测的结果重现变为(xmin,ymin,xmax,ymax)格式
        每次转换结果的是一张图片的
        """
        variances = [0.1, 0.2]
        wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [8732,4]

        assert boxes.size() == (8732, 4)
        assert conf.size() == (8732, 21)

        # print('in Box conver_result, boxes mean: ', boxes.mean())
        # 得到最大可能的类别，其中也包括了背景
        # 这个labels就是每个default box对应的21个类别中最大的那个的下标，当然也就代表了最大类别
        max_conf, labels = conf.max(1)  # [8732,1]
        # print('in Bos convert_result, max_conf mean: ', max_conf.mean())
        # print('in Box convert_result, labels.mean: ', labels.to(torch.float).mean())
        # 去掉背景的类别，得到了长为num,的一维tensor，每个位置的值就是类别，num就是检测数来的obj的个数
        ids = labels.squeeze().nonzero().squeeze()  # [#boxes,]
        # print('in Box convert_result, ids size: ', ids.size())
        # 非极大值抑制，选出保留下来的box label
        keep = self.nms(boxes[ids], max_conf[ids].squeeze())
        print('in Box convert_result, keep size: ', keep.size())
        # 这里的boxes labels max_conf都是8732维度的，所以先选出之前不是背景的那些，然后再选出nms的结果
        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]


if __name__ == '__main__':
    # for debug and understand the code
    from config import DefaultConfig

    opt = DefaultConfig()
    prior_box = PriorBox(opt)
    print(prior_box.default_boxes.size())
