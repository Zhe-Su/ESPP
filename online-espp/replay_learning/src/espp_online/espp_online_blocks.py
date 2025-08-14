from typing import Callable, Any
from src.util.blocks import FF_Block, CNN_Block, CNN_Block_Chipready
from src.espp_online import OnlineESPPLayer


def create_espp_ff_block(
        in_features:int, 
        out_features:int, 
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        rule:str="gated_mi",
        c_pos:float=2.0,
        c_neg:float=-1.0,
        input_thr:float=0.02,
        stream_data:bool=False,
        use_replay:bool=False,
        apply_weight:bool=False, 
        learn_temp:bool=False, 
        gamma:float=0.1,
        temp:float=0.5,
        K=1,
    ) -> OnlineESPPLayer:
        block = FF_Block(
            in_features=in_features, 
            out_features=out_features, 
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
        )
        return OnlineESPPLayer(
            layer=block,
            rule=rule, 
            c_pos=c_pos, 
            c_neg=c_neg, 
            input_thr=input_thr, 
            use_replay=use_replay, 
            stream_data=stream_data, 
            apply_weight=apply_weight, 
            learn_temp=learn_temp, 
            gamma=gamma,
            temp=temp,
            K=K,
        )


def create_espp_block(
        in_channels:int, 
        out_channels:int, 
        kernel_size:int, 
        pool_size:int,    
        beta:float, 
        spike_grad:Callable,
        init_hidden:bool,
        leaky_for_pooling:bool,
        rule:str="gated_mi",
        c_pos:float=2.0,
        c_neg:float=-1.0,
        input_thr:float=0.02,
        stream_data:bool=False,
        use_replay:bool=False,
        apply_weight:bool=False, 
        learn_temp:bool=False, 
        gamma:float=0.1,
        temp:float=0.5,
        K=1,
    ) -> OnlineESPPLayer:
        block = CNN_Block(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            pool_size=pool_size,    
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
            leaky_for_pooling=leaky_for_pooling
        )
        return OnlineESPPLayer(
            layer=block,
            rule=rule, 
            c_pos=c_pos, 
            c_neg=c_neg, 
            input_thr=input_thr, 
            use_replay=use_replay, 
            stream_data=stream_data, 
            apply_weight=apply_weight, 
            learn_temp=learn_temp, 
            gamma=gamma,
            temp=temp,
            K=K,
        )


def create_espp_block_chipready(
        in_channels:int, 
        out_channels:int, 
        kernel_size:int, 
        beta:float, 
        stride:int,
        spike_grad:Callable,
        init_hidden:bool,
        rule:str="gated_mi",
        c_pos:float=2.0,
        c_neg:float=-1.0,
        input_thr:float=0.02,
        stream_data:bool=False,
        use_replay:bool=False,
        apply_weight:bool=False, 
        learn_temp:bool=False, 
        gamma:float=0.1,
        temp:float=0.5,
        K=1,
    ) -> OnlineESPPLayer:
        block = CNN_Block_Chipready(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            beta=beta, 
            spike_grad=spike_grad,
            init_hidden=init_hidden,
        )
        return OnlineESPPLayer(
            layer=block,
            rule=rule, 
            c_pos=c_pos, 
            c_neg=c_neg, 
            input_thr=input_thr, 
            use_replay=use_replay, 
            stream_data=stream_data, 
            apply_weight=apply_weight, 
            learn_temp=learn_temp, 
            gamma=gamma,
            temp=temp,
            K=K,
        )