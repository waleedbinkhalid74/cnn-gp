from turtle import forward
from typing import Tuple
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from zmq import device
from .kernel_patch import ConvKP, NonlinKP
import math
import numpy as np
# from numpy.fft import fft2, ifft2, fftn, ifftn
from torch import autograd

__all__ = ("NNGPKernel", "Conv2d", "ReLU", "Sequential", "Mixture",
           "MixtureModule", "Sum", "SumModule", "resnet_block", "ReLUCNNGP")

class ReLUCNNGP(autograd.Function):
    """Custom implementation of autograd function for ReLU Layer

    Args:
        autograd : torch.autograd.Function class

    """
    @staticmethod
    def forward(ctx, xy: t.Tensor, xx: t.Tensor, yy: t.Tensor) -> t.Tensor:
        """Forward evaluation of K(X,X') for the ReLU Layer in CNN GP
        We need to calculate (xy, xx, yy == c, v₁, v₂ == K(X,X'), K(X,X), K(X',X')):
        √(v₁v₂) / 2π ⎷1 - c²/v₁v₂ + (π - θ)c / √(v₁v₂)

        which is equivalent to:
        1/2π ( √(v₁v₂ - c²) + (π - θ)c )

        # NOTE we divide by 2 to avoid multiplying the ReLU by sqrt(2)
        Args:
            ctx : context object for storage purposes of backward pass
            xy (t.Tensor): K(X,X')
            xx (t.Tensor): K(X,X)
            yy (t.Tensor): K(X',X')

        Returns:
            t.Tensor: V(X,X')
        """
        ctx.save_for_backward(xy, xx, yy)
        f32_tiny = np.finfo(np.float32).tiny
        xx_yy = xx*yy + f32_tiny
        eps = 1e-6
        # NOTE: Replaced rsqrt with 1/t.sqrt()+eps. Check with Prof For accuracy
        inverse_sqrt_xx_yy = 1 / (t.sqrt(xx_yy) + eps)
        # inverse_sqrt_xx_yy = t.rsqrt(xx_yy)
        # Clamp these so the outputs are not NaN
        # Use small eps to avoid NaN during backpropagation
        cos_theta = (xy * inverse_sqrt_xx_yy).clamp(-1+eps, 1-eps)
        del inverse_sqrt_xx_yy
        sin_theta = t.sqrt((xx_yy - xy**2).clamp(min=eps))
        del xx_yy
        theta = t.acos(cos_theta)
        del cos_theta
        V_xy = (sin_theta + (math.pi - theta)*xy) / (2*math.pi)
        return V_xy

    @staticmethod
    def backward(ctx, grad_output: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """Evaluate the backward pass for ReLU Layer
        We need to differentiate (xy, xx, yy == c, v₁, v₂ == K(X,X'), K(X,X), K(X',X')):
        relu = 1/2π ( √(v₁v₂ - c²) + (π - θ)c )
        wrt c and a=v₁v₂ and subsequently apply the chain rule i.e.
        d(relu)/dv₁ = d(relu)/da * da/dv₁
        d(relu)/dv₂ = d(relu)/da * da/dv₂

        # NOTE: For differentiation wrt c=xy see https://www.wolframalpha.com/input?i=differentiate+%28sqrt%28a*b+-+c%5E2%29+%2B+%28pi+-+arccos%28c%2Fsqrt%28a*b%29%29%29*c%29%2F%282pi%29+wrt+c
        # NOTE: For differentiation wrt a=xx_yy see https://www.wolframalpha.com/input?i=differentiate+%28sqrt%28a+-+c%5E2%29+%2B+%28pi+-+arccos%28c%2Fsqrt%28a%29%29%29*c%29%2F%282pi%29+wrt+a

        where da/dv₂ = v₁ & da/dv₁ = v₂

        Args:
            ctx (): context object for retreiving forward pass variable
            grad_output (t.Tensor): Gradient from output layer

        Returns:
            Tuple[t.Tensor, t.Tensor, t.Tensor]: Gradients wrt K(X,X'), K(X,X) and K(X',X') respectively
        """
        xy, xx, yy = ctx.saved_tensors
        f32_tiny = np.finfo(np.float32).tiny
        eps = 0* 1e-10
        xx_yy = xx*yy + f32_tiny
        # NOTE: See https://www.wolframalpha.com/input?i=differentiate+%28sqrt%28a*b+-+c%5E2%29+%2B+%28pi+-+arccos%28c%2Fsqrt%28a*b%29%29%29*c%29%2F%282pi%29+wrt+c for differentiation wrt xy
        term_3 = t.acos((xy / (t.sqrt(xx_yy))).clamp(-1, 1))

        # Convert nans to 0 and inf to large numbers
        term_3 = t.nan_to_num(term_3)

        # NOTE: We can remove term_1 and term_2 as they cancel each other out in exact arthmetics
        diff_xy = (- term_3 + math.pi) / (2*math.pi)
        diff_xy_chained = grad_output * diff_xy

        # NOTE: See https://www.wolframalpha.com/input?i=differentiate+%28sqrt%28a+-+c%5E2%29+%2B+%28pi+-+arccos%28c%2Fsqrt%28a%29%29%29*c%29%2F%282pi%29+wrt+a for differentiation wrt xx_yy.
        sin_theta = (xx_yy - xy**2).clamp(min=eps)
        term_1 = 1 / (2 * t.sqrt(sin_theta))
        term_2 = xy**2 / (2 * xx_yy * t.sqrt(sin_theta))

        # Convert nans to 0 and inf to large numbers
        # NOTE: The two terms can be infinite and we wish the result to be zero if both terms are inifnity
        # In pytorch inf - inf is nan so we convert the nans to zero
        diff_xx_yy = (term_1 - term_2) / (2 * math.pi)
        diff_xx_yy = t.nan_to_num(diff_xx_yy)

        # diff_xx_yy = term_1 * (1 - term_2) / (2 * math.pi)
        diff_xx = yy * diff_xx_yy
        diff_xx_chained = grad_output * diff_xx
        diff_yy = xx * diff_xx_yy
        diff_yy_chained = grad_output * diff_yy

        return diff_xy_chained, diff_xx_chained, diff_yy_chained

class NNGPKernel(nn.Module):
    """
    Transforms one kernel matrix into another.
    [N1, N2, W, H] -> [N1, N2, W, H]
    """
    def forward(self, x, y=None, same=None, diag=False):
        """
        Either takes one minibatch (x), or takes two minibatches (x and y), and
        a boolean indicating whether they're the same.
        """
        if y is None:
            assert same is None
            y = x
            same = True

        assert not diag or len(x) == len(y), (
            "diagonal kernels must operate with data of equal length")

        assert 4==len(x.size())
        assert 4==len(y.size())
        assert x.size(1) == y.size(1)
        assert x.size(2) == y.size(2)
        assert x.size(3) == y.size(3)

        self.N1 = x.size(0)
        self.N2 = y.size(0)
        C = x.size(1)
        W = x.size(2)
        H = x.size(3)

        # [N1, C, W, H], [N2, C, W, H] -> [N1 N2, 1, W, H]
        if diag:
            xy = (x*y).mean(1, keepdim=True)
        else:
            xy = (x.unsqueeze(1)*y).mean(2).view(self.N1*self.N2, 1, W, H)
        xx = (x**2).mean(1, keepdim=True)
        yy = (y**2).mean(1, keepdim=True)

        initial_kp = ConvKP(same, diag, xy, xx, yy)
        final_kp = self.propagate(initial_kp)
        r = NonlinKP(final_kp).xy
        if diag:
            return r.view(self.N1)
        else:
            return r.view(self.N1, self.N2)


class Conv2d(NNGPKernel):
    def __init__(self, kernel_size, stride=1, padding="same", dilation=1,
                 var_weight=1., var_bias=0., in_channel_multiplier=1,
                 out_channel_multiplier=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.kp = None
        # Not needed as they will be registered separately as parameters
        # self.var_weight = var_weight
        # self.var_bias = var_bias
        self.kernel_has_row_of_zeros = False
        if padding == "same":
            self.padding = dilation*(kernel_size//2)
            if kernel_size % 2 == 0:
                self.kernel_has_row_of_zeros = True
        else:
            self.padding = padding

        self.var_weight = nn.Parameter(t.Tensor([var_weight]))
        self.var_bias = nn.Parameter(t.Tensor([var_bias]))
        if self.kernel_has_row_of_zeros:
            # We need to pad one side larger than the other. We just make a
            # kernel that is slightly too large and make its last column and
            # row zeros.
            kernel = t.ones(1, 1, self.kernel_size+1, self.kernel_size+1)
            kernel[:, :, 0, :] = 0.
            kernel[:, :, :, 0] = 0.
        else:
            kernel = t.ones(1, 1, self.kernel_size, self.kernel_size)

        self.register_buffer('kernel', kernel)
        self.kernel = self.kernel / self.kernel_size**2
        self.in_channel_multiplier, self.out_channel_multiplier = (
            in_channel_multiplier, out_channel_multiplier)


    def propagate(self, kp):
        kp = ConvKP(kp)
        # NOTE: Only collect data otherwise autograd computational graph will also be saved which is memory intensive
        # This will be used in our custom backward pass
        # self.kp = ConvKP(kp.same, kp.diag, kp.xy.data, kp.xx.data, kp.yy.data)
        ###########################ADDED CALCULATION OF KERNEL FROM TRAINABLE VARIANCES###########################
        kernel = self.kernel * self.var_weight
        ###########################ADDED CALCULATION OF KERNEL FROM TRAINABLE VARIANCES###########################
        def f(patch):
            return (F.conv2d(patch, kernel, stride=self.stride, # CHANGE self.kernel to kernel
                             padding=self.padding, dilation=self.dilation)
                    + self.var_bias)

        return ConvKP(kp.same, kp.diag, f(kp.xy), f(kp.xx), f(kp.yy))

    def nn(self, channels, in_channels=None, out_channels=None):
        if in_channels is None:
            in_channels = channels
        if out_channels is None:
            out_channels = channels
        conv2d = nn.Conv2d(
            in_channels=in_channels * self.in_channel_multiplier,
            out_channels=out_channels * self.out_channel_multiplier,
            kernel_size=self.kernel_size + (
                1 if self.kernel_has_row_of_zeros else 0),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=(self.var_bias > 0.),
        )
        conv2d.weight.data.normal_(0, math.sqrt(
            self.var_weight / conv2d.in_channels) / self.kernel_size)
        if self.kernel_has_row_of_zeros:
            conv2d.weight.data[:, :, 0, :] = 0
            conv2d.weight.data[:, :, :, 0] = 0
        if self.var_bias > 0.:
            conv2d.bias.data.normal_(0, math.sqrt(self.var_bias))
        return conv2d

    def layers(self):
        return 1



class ReLU(NNGPKernel):
    """
    A ReLU nonlinearity, the covariance is numerically stabilised by clamping
    values.
    """
    f32_tiny = np.finfo(np.float32).tiny

    def propagate(self, kp):
        kp = NonlinKP(kp)
        # NOTE: Only collect data otherwise autograd computational graph will also be saved which is memory intensive
        # This will be used in our custom backward pass
        # self.kp = NonlinKP(kp.same, kp.diag, kp.xy.data, kp.xx.data, kp.yy.data)
        """
        We need to calculate (xy, xx, yy == c, v₁, v₂):
                      ⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤
        √(v₁v₂) / 2π ⎷1 - c²/v₁v₂ + (π - θ)c / √(v₁v₂)

        which is equivalent to:
        1/2π ( √(v₁v₂ - c²) + (π - θ)c )

        # NOTE we divide by 2 to avoid multiplying the ReLU by sqrt(2)
        # NOTE this entire logic is encapsulated in a custom autograd function along with its differential
        """


        ###################SELF IMPLEMENTED BACKWARD PASS###########################
        # relu_cnngp = ReLUCNNGP.apply
        # xy = relu_cnngp(kp.xy, kp.xx, kp.yy)
        # xy = relu_cnngp(kp.xy, xx_yy)
        ###################SELF IMPLEMENTED BACKWARD PASS###########################

        xx_yy = kp.xx * kp.yy + self.f32_tiny
        # Clamp these so the outputs are not NaN
        # Use small eps to avoid NaN during backpropagation
        eps = 0*1e-6

        # NOTE: Replaced rsqrt with 1/t.sqrt()+eps. This is because diff of 1/sqrt(xx_yy) is 1 / (2*xx_yy^1.5) and this 1.5 power turns the small f32tiny into zero
        # Check with Prof For accuracy
        # inverse_sqrt_xx_yy = 1 / (t.sqrt(xx_yy) + eps)
        # cos_theta = (kp.xy * inverse_sqrt_xx_yy).clamp(-1+eps, 1-eps)
        cos_theta = (kp.xy * xx_yy.rsqrt()).clamp(-1+eps, 1-eps)

        sin_theta = t.sqrt((xx_yy - kp.xy**2).clamp(min=eps))
        # sin_theta = t.sqrt((xx_yy - kp.xy**2).clamp(min=0))
        theta = t.acos(cos_theta)
        xy = (sin_theta + (math.pi - theta)*kp.xy) / (2*math.pi)

        xx = kp.xx/2.
        if kp.same:
            yy = xx
            if kp.diag:
                xy = xx
            else:
                # Make sure the diagonal agrees with `xx`
                eye = t.eye(xy.size()[0]).unsqueeze(-1).unsqueeze(-1).to(kp.xy.device)
                xy = (1-eye)*xy + eye*xx
        else:
            yy = kp.yy/2.
        return NonlinKP(kp.same, kp.diag, xy, xx, yy)

    def nn(self, channels, in_channels=None, out_channels=None):
        assert in_channels is None
        assert out_channels is None
        return nn.ReLU()

    def layers(self):
        return 0


#### Combination classes

class Sequential(NNGPKernel):
    def __init__(self, *mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def propagate(self, kp):
        for mod in self.mods:
            kp = mod.propagate(kp)
        return kp
    def nn(self, channels, in_channels=None, out_channels=None):
        if len(self.mods) == 0:
            return nn.Sequential()
        elif len(self.mods) == 1:
            return self.mods[0].nn(channels, in_channels=in_channels, out_channels=out_channels)
        else:
            return nn.Sequential(
                self.mods[0].nn(channels, in_channels=in_channels),
                *[mod.nn(channels) for mod in self.mods[1:-1]],
                self.mods[-1].nn(channels, out_channels=out_channels)
            )
    def layers(self):
        return sum(mod.layers() for mod in self.mods)


class Mixture(NNGPKernel):
    """
    Applys multiple modules to the input, and sums the result
    (e.g. for the implementation of a ResNet).

    Parameterised by proportion of each module (proportions add
    up to one, such that, if each model has average variance 1,
    then the output will also have average variance 1.
    """
    def __init__(self, mods, logit_proportions=None):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
        if logit_proportions is None:
            logit_proportions = t.zeros(len(mods))
        self.logit = nn.Parameter(logit_proportions)
    def propagate(self, kp):
        proportions = F.softmax(self.logit, dim=0)
        total = self.mods[0].propagate(kp) * proportions[0]
        for i in range(1, len(self.mods)):
            total = total + (self.mods[i].propagate(kp) * proportions[i])
        return total
    def nn(self, channels, in_channels=None, out_channels=None):
        return MixtureModule([mod.nn(channels, in_channels=in_channels, out_channels=out_channels) for mod in self.mods], self.logit)
    def layers(self):
        return max(mod.layers() for mod in self.mods)

class MixtureModule(nn.Module):
    def __init__(self, mods, logit_parameter):
        super().__init__()
        self.mods = mods
        self.logit = t.tensor(logit_parameter)
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def forward(self, input):
        sqrt_proportions = F.softmax(self.logit, dim=0).sqrt()
        total = self.mods[0](input)*sqrt_proportions[0]
        for i in range(1, len(self.mods)):
            total = total + self.mods[i](input) # *sqrt_proportions[i]
        return total


class Sum(NNGPKernel):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def propagate(self, kp):
        # This adds 0 to the first kp, hopefully that's a noop
        return sum(m.propagate(kp) for m in self.mods)
    def nn(self, channels, in_channels=None, out_channels=None):
        return SumModule([
            mod.nn(channels, in_channels=in_channels, out_channels=out_channels)
            for mod in self.mods])
    def layers(self):
        return max(mod.layers() for mod in self.mods)


class SumModule(nn.Module):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def forward(self, input):
        # This adds 0 to the first value, hopefully that's a noop
        return sum(m(input) for m in self.mods)


def resnet_block(stride=1, projection_shortcut=False, multiplier=1):
    if stride == 1 and not projection_shortcut:
        return Sum([
            Sequential(),
            Sequential(
                ReLU(),
                Conv2d(3, stride=stride, in_channel_multiplier=multiplier, out_channel_multiplier=multiplier),
                ReLU(),
                Conv2d(3, in_channel_multiplier=multiplier, out_channel_multiplier=multiplier),
            )
        ])
    else:
        return Sequential(
            ReLU(),
            Sum([
                Conv2d(1, stride=stride, in_channel_multiplier=multiplier//stride, out_channel_multiplier=multiplier),
                Sequential(
                    Conv2d(3, stride=stride, in_channel_multiplier=multiplier//stride, out_channel_multiplier=multiplier),
                    ReLU(),
                    Conv2d(3, in_channel_multiplier=multiplier, out_channel_multiplier=multiplier),
                )
            ]),
        )
