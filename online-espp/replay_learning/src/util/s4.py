"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math

import torch
from einops import repeat
from torch import nn

import torch
import numpy as np
from numpy.linalg import eigh

# from src.models.nn import DropoutNd


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""
    def __init__(self, d_model:int, N:int=64, dt_min:float=0.001, dt_max:float=0.1, lr=None) -> None:
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L:int, rate: float = 1.0) -> torch.Tensor:
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt) * rate  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

        return K

    def register(self, name:str, tensor:torch.Tensor, lr:float=None) -> None:
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim


class S4D(nn.Module):
    def __init__(self, d_model: int, d_state:int=64, dropout:float=0.0, transposed:bool=True, **kernel_args) -> None:
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        dropout_fn = nn.Dropout2d  # NOTE: bugged in PyTorch 1.11
        # dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: torch.Tensor, rate: float = 1.0) -> torch.Tensor:  # absorbs return_output and transformer src mask
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L, rate=rate)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        # Return a dummy state None to satisfy this repo's interface is missing; likely won't work with s4 repo structure, i.e. s4.py
        return y



def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


class TraceLegS(nn.Module):
    """
    most naive concept implementation of a SSM with a state matrix of multiple diagonally placed A_LegS matrices.
    """
    def __init__(self, N:int, H:int) -> None:
        super().__init__()
        assert N % H == 0
        self.N = N
        self.H = H
        self.P = N // H
        A = make_HiPPO(self.H)
        self.A = nn.Linear(self.H, self.H, bias=False)
        self.A.weight = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        # self.A.requires_grad_(False)
        # L, _, _, V, _ = make_DPLR_HiPPO(self.P)
        # self.L = nn.Linear(self.P, self.P, bias=False, requires_grad=False)
        # self.L.weight = L * np.eye(self.P)
        # self.V = nn.Linear(self.P, self.P, bias=False, requires_grad=False)
        # self.V.weight = V
        # self.Vinv = nn.Linear(self.P, self.P, bias=False, requires_grad=False)
        # self.Vinv = V.conj().T

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        input: 
            x: (T, B, ...)
        output:
            x: (T, B, ...)
        """
        shape = x.shape
        x = x.view(shape[0], shape[1], self.P, self.H)
        ys = []
        y = torch.zeros_like(x[0,...])
        for t in range(shape[0]):
            y = self.A(y) + x[t,...]
            ys.append(y)
        ys = torch.stack(ys)
        ys = ys.view(shape)
        return ys





# L, P, B, V, B_orig = make_DPLR_HiPPO(512)


class S4DTrace(nn.Module):
    log_Lambda_real: torch.Tensor
    Lambda_imag: torch.Tensor
    V: torch.Tensor
    Vinv: torch.Tensor
    log_dt: torch.Tensor
    
    def __init__(self, N:int, H:int=64, dt_min:float=0.001, dt_max:float=0.1, lr=None) -> None:
        super().__init__()
        # Generate dt
        assert N % H == 0
        self.H = H
        self.N = N
        self.P = N // H
        Lambda, _, _, V, _ = make_DPLR_HiPPO(H)
        Lambda = torch.tensor(Lambda, dtype=torch.complex64)
        # log_Lambda_real = torch.log(Lambda.real)
        log_Lambda_real = torch.log(-Lambda.real)
        Lambda_imag = Lambda.imag
        V = torch.tensor(V, dtype=torch.complex64)
        Vinv = V.conj().T

        log_dt = torch.rand(H) * ((math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.register("log_dt", log_dt, lr)

        self.register("log_Lambda_real", log_Lambda_real, lr=0.0)
        self.register("Lambda_imag", Lambda_imag, lr=0.0)
        self.register("V", V, lr=0.0)
        self.register("Vinv", Vinv, lr=0.0)

    def forward(self, xs:torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (T, B, ...)
        output:
            y: (T, B, ...)
        """
        T = xs.shape[0]
        B = xs.shape[1]
        shape = xs.shape
        xs = xs.view(T, B, self.P, self.H)
        dt = torch.exp(self.log_dt)
        Lambda_real = -torch.exp(self.log_Lambda_real)
        A = torch.exp((Lambda_real + self.Lambda_imag * 1j) * dt)
        Vinv = self.Vinv * dt[:, None]
        ys = []
        y = torch.zeros_like(xs[0,...])
        for x in xs:
            y = A[None, None, :] * (y + 0j) + torch.vmap(torch.vmap(lambda x: Vinv @ x))(x + 0j)
            y = torch.vmap(torch.vmap(lambda x: self.V @ x))(y)
            ys.append(y)
        ys = torch.stack(ys)
        ys = ys.view(shape)
        ys = 2 * ys.real
        return ys

#     def calc_kernel(self, L:int, rate:float=1.0) -> torch.Tensor:
#         # Materialize parameters
#         dt = torch.exp(self.log_dt) * rate  # (H)
#         Lambda = -torch.exp(self.log_Lambda_real) + 1j * self.Lambda_imag  # (H N)

#         # Vandermonde multiplication
#         dtLambda = Lambda * dt.unsqueeze(-1)  # (H N)
#         K = dtLambda.unsqueeze(-1) * torch.arange(L, device=Lambda.device)  # (H N L)
        
#         # C = C * (torch.exp(dtLambda) - 1.0) / Lambda # zoh
#         C = self.V.conj().T * dt.unsqueeze(-1) # dirac-ssm
        
#         K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real
#         return K


    def register(self, name:str, tensor:torch.Tensor, lr:float=None) -> None:
        """Register a tensor with a configurable learning rate and 0 weight decay"""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim