import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
from torch.autograd.variable import Variable
from torch.nn.modules.loss import _Loss

class BasicConv2d_activation(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_activation, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=3, bias=False)
        self.conv3 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=5, bias=False)
        self.conv4 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=7, bias=False)       
        self.conv_cat = BasicConv2d(4*out_planes, out_planes, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.convd = RF(in_planes, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        #x = self.convd(x)
        #x2 = self.conv2(x)
        #x2 = self.bn(x2)
        #x3 = self.conv3(x)
        #x3 = self.bn(x3)
        #x4 = self.conv4(x)
        #x4 = self.bn(x4)  
        #x = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.sigmoid(x)
        return x
    
class BasicConv2d_activationRL(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_activationRL, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=3, bias=False)
        self.conv3 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=5, bias=False)
        self.conv4 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=7, bias=False)       
        self.conv_cat = BasicConv2d(4*out_planes, out_planes, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.convd = RF(in_planes, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        #x = self.convd(x)
        #x2 = self.conv2(x)
        #x2 = self.bn(x2)
        #x3 = self.conv3(x)
        #x3 = self.bn(x3)
        #x4 = self.conv4(x)
        #x4 = self.bn(x4)  
        #x = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.relu(x)
        return x

class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        #x = self.relu(x_cat + self.conv_res(x))
        return x
    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class BasicConv2d_pose(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_pose, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
 
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)   #[b, B*(16+1), 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out
    
class PrimaryCapsClass(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, h=2, w=2):
        super(PrimaryCapsClass, self).__init__()
        self.pose = nn.Conv2d(in_channels=A*h*w, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A*h*w, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, T, h, w = x.shape
        x = x.reshape(b, h * w * T, 1, 1)
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)   #[b, B*(16+1), 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out
    
#class PrimaryCapsClass(nn.Module):
    #r"""Creates a primary convolutional capsule layer
    #that outputs a pose matrix and an activation.

    #Note that for computation convenience, pose matrix
    #are stored in first part while the activations are
    #stored in the second part.

    #Args:
        #A: output of the normal conv layer
        #B: number of types of capsules
        #K: kernel size of convolution
        #P: size of pose matrix is P*P
        #stride: stride of convolution

    #Shape:
        #input:  (*, A, h, w)
        #output: (*, h', w', B*(P*P+1))
        #h', w' is computed the same way as convolution layer
        #parameter size is: K*K*A*B*P*P + B*P*P
    #"""
    #def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        #super(PrimaryCapsClass, self).__init__()
        #self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            #kernel_size=K, stride=stride, bias=True)
        #self.pose1 = nn.Conv2d(in_channels=B*P*P, out_channels=B*P*P,
                            #kernel_size=K, stride=stride, bias=True)        
        #self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            #kernel_size=K, stride=stride, bias=True)
        #self.a1 = nn.Conv2d(in_channels=B, out_channels=B,
                            #kernel_size=K, stride=stride, bias=True)        
        #self.sigmoid = nn.Sigmoid()
        
    #def forward(self, x):
        #p = self.pose(x)
        #p = self.pose1(p)
        #a = self.a(x)
        #a = self.a1(a)
        #a = self.sigmoid(a)
        #out = torch.cat([p, a], dim=1)   #[b, B*(16+1), 14, 14]
        #out = out.permute(0, 2, 3, 1)
        #return out


class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3, channel1 = 272,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        #self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        self.weights = nn.Parameter(torch.randn(1, K*K*B, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.conv1 = BasicConv2d_activation(B*K*K*16, B*K*K*1, 1)       
        self.conv2 = BasicConv2d_activationRL(B*K*K*17, B*17, 1)   
        #self.conv3 = BasicConv2d_activation()
    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        # keep the scale
        x_padding_h = torch.zeros(b, 1, w, c).cuda()
        x_padding_w = torch.zeros(b, h+2, 1, c).cuda()
        x = torch.cat([ x_padding_h, x, x_padding_h ], dim=1)
        x = torch.cat([ x_padding_w, x, x_padding_w ], dim=2)
        b, h, w, c = x.shape
        #assert h == w
        assert c == B*(psize+1)
        oh = ow = int(((h - K )/stride)+ 1) # moein - changed from: oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for k_idx in range(0, K)] \
                for h_idx in range(0, h - K + 1, stride)]
        
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def transform_view(self, x, w, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P
        x = x.view(b, B, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1)
        w = w + 1
        x = x.repeat(1, 1, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        #assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            #a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            #a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            p_out = self.transform_view(p_in, self.weights, self.P)
            #p_out = F.dropout(p_out, p = 0.5)
            p_out_R = p_out.reshape(p_out.size(0), -1)
            p_out_R = p_out_R.reshape(b, oh, ow, -1)
            p_out = p_out_R
            #p_out_R = p_out_R.reshape(b, oh, ow, self.B*self.K*self.K, self.P*self.P)
            p_out_1 = p_out_R.permute(0, 3, 1, 2)
            a_out = self.conv1(p_out_1)
            #a_out = self.sigmoid(torch.norm(p_out_R, dim=4))
            a_out = a_out.permute(0, 2, 3, 1)
            
            out = torch.cat([p_out, a_out], dim=3)
            out = out.permute(0, 3, 1, 2)
            out = self.conv2(out)
            #out = out.permute(0, 2, 3, 1)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K
            assert 1 == self.stride
            #x = x.reshape(b, 1, 1, h*w*c)
            #b, h, w, c = x.shape
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            #a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            #a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            
            p_out = self.transform_view(p_in, self.weights, self.P)
            #p_out = F.dropout(p_out, p = 0.5)
            p_out_R = p_out.reshape(p_out.size(0), -1)
            p_out_R = p_out_R.reshape(b, h, w, -1)
            p_out = p_out_R
            p_out_R = p_out_R.reshape(b, h, w, self.B*self.K*self.K, self.P*self.P)
            p_out_R = p_out_R.reshape(b, h, w, -1)
            p_in_T = p_out_R.permute(0, 3, 1, 2)
            
            a_out = self.conv1(p_in_T)
            #out = self.sigmoid(torch.norm(p_out_R, dim=4))
            out = a_out.reshape(b, -1)

        #return out, p_out, a_out
        return out


class CapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, FF=8, G=8, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=A,
                               kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.bn2 = nn.BatchNorm2d(num_features=B*17, eps=0.001,
                                 momentum=0.1, affine=True)
        self.bn3 = nn.BatchNorm2d(num_features=C*17, eps=0.001,
                                 momentum=0.1, affine=True)
        self.bn4 = nn.BatchNorm2d(num_features=D*17, eps=0.001,
                                 momentum=0.1, affine=True)      
        self.bn5 = nn.BatchNorm2d(num_features=FF*17, eps=0.001,
                                 momentum=0.1, affine=True)   
        self.bn6 = nn.BatchNorm2d(num_features=G*17, eps=0.001,
                                 momentum=0.1, affine=True)        
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=2)
        self.primary_caps2 = PrimaryCaps(B*17, C, 1, P, stride=2)
        self.primary_caps3 = PrimaryCaps(C*17, D, 1, P, stride=2)
        self.primary_caps4 = PrimaryCaps(D*17, FF, 1, P, stride=2)
        #self.primary_caps5 = PrimaryCaps(FF*17, E, 1, P, stride=1)
        self.primary_caps5 = PrimaryCapsClass(FF*17, E, 1, P, stride=1, h=2, w=2)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.conv_caps3_1 = ConvCaps(D, FF, K, P, stride=1, iters=iters)
        self.conv_caps3_2 = ConvCaps(D, FF, K, P, stride=1, iters=iters)
        self.conv_caps4_1 = ConvCaps(FF, G, K, P, stride=1, iters=iters)
        self.conv_caps4_2 = ConvCaps(FF, G, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(E, E, 1, P, stride=1, iters=iters,
                                        coor_add=True, w_shared=True) 
        self.convd = RF(A, A)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.convd(x)
        x = self.relu1(x)
        x = self.primary_caps1(x)
        x = self.conv_caps1_1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv_caps1_2(x)
        x = self.bn2(x)
        x = self.primary_caps2(x)
        x = self.conv_caps2_1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv_caps2_2(x)
        x = self.bn3(x)
        x = self.primary_caps3(x)
        x = self.conv_caps3_1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv_caps3_2(x)
        x = self.bn4(x)
        x = self.primary_caps4(x)
        x = self.conv_caps4_1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv_caps4_2(x)    
        x = self.bn5(x)
        x = self.primary_caps5(x)
        x = self.class_caps(x)
        return x


def capsules(**kwargs):
    """Constructs a CapsNet model.
    """
    model = CapsNet(**kwargs)
    return model


'''
TEST
Run this code with:
```
python -m capsules.py
```
'''
if __name__ == '__main__':
    model = capsules(E=10)
