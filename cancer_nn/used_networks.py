"""Script for torch neural networks"""
import torch
import torch.nn as nn

from cancer_nn.nn_helpers import (
    attention_module,
    base_conv1_layer,
    base_linear_module,
    calculate_matching_padding,
    calculate_output_shape,
    transconv_matching_padding,
)

##################################################################
"""single regressor"""
##################################################################
class Regressor(nn.Module):
    def __init__(
        self,
        output_last_layer=True,
        n_inputs=[3, 16, 32, 64],
        n_outputs=[16, 32, 64, 128],
        kernel_size=[5, 4, 4, 3],
        stride=[1, 1, 2, 1],
        padding=[1, 2, 1, 2],
        dilation=[1, 2, 1, 2],
        series_len=20,
        with_att=True,
    ):
        super().__init__()
        self.n_layers = len(dilation)
        self.with_att = with_att
        self.output_last_layer = output_last_layer
        layers = [
            base_conv1_layer(
                in_f,
                out_f,
                k,
                stride=s,
                padding=calculate_matching_padding(series_len, series_len, k, d, s),
                dilation=d,
            )
            for in_f, out_f, k, s, p, d in zip(
                n_inputs, n_outputs, kernel_size, stride, padding, dilation
            )
        ]
        self.cnn_layers = nn.ModuleList(layers)
        self.att_layers = attention_module(embed_dim=n_outputs[-1])

        self.activation_layers = nn.GELU()
        w = 20
        for p, k, d, s in zip(padding, kernel_size, dilation, stride):
            w = calculate_output_shape(w, p, k, d, s)
        if self.output_last_layer:
            self.last_layer = nn.Linear(int(series_len * n_outputs[-1]), 1)
            # print(self.last_layer)
        else:
            self.last_layer = nn.Sequential(
                nn.Linear(series_len * n_outputs[-1], 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(),
            )

    def forward(self, batch):
        x, _ = batch
        for i in range(self.n_layers):
            x = x.permute(0, 2, 1)
            x = self.cnn_layers[i](x)
            x = x.permute(0, 2, 1)
            if i == self.n_layers - 1:
                if self.with_att:
                    x, _ = self.att_layers(x, x, x)
                    x = self.activation_layers(x)

        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        return x


##################################################################
"""Multi head regressor"""
##################################################################
class MultiTaskRegressor(nn.Module):
    def __init__(self, configs_list, mode="unet"):
        super().__init__()
        self.num_regr = len(configs_list)
        if mode == "unet":
            layers = [Unet(**config) for config in configs_list]
        elif mode == "fcnn":
            layers = [FCNN(**config) for config in configs_list]
        elif mode == "lstm":
            layers = [AttLSTM(**config) for config in configs_list]
        elif mode == "cnn_lstm":
            layers = [CNNAttLSTM(**config) for config in configs_list]
        elif mode == "cnn_lstm_att":
            layers = [CNNALSTMAtt(**config) for config in configs_list]

        else:

            layers = [
                Regressor(output_last_layer=True, **config) for config in configs_list
            ]
        self.layers = nn.ModuleList(layers)
        regr_heads = [nn.Sequential(nn.Linear(1, 1)) for _ in range(self.num_regr)]
        self.regr_heads = nn.ModuleList(regr_heads)
        self.last_layer = nn.Linear(self.num_regr * 1, 1)

    def forward(self, batch):
        outputs = []
        for i in range(self.num_regr):
            outputs.append(self.layers[i](batch))
        org_x = torch.cat(outputs, 1)
        x = torch.mean(org_x, 1)
        return torch.stack([x.unsqueeze(1)] + outputs, 1)


class MultiHeadTaskRegressor(nn.Module):
    def __init__(self, configs_list, mode="unet"):
        super().__init__()
        self.num_regr = len(configs_list)
        if mode == "unet":
            layers = [Unet(**config) for config in configs_list]
        elif mode == "fcnn":
            layers = [FCNN(**config) for config in configs_list]
        elif mode == "lstm":
            layers = [AttLSTM(**config) for config in configs_list]
        elif mode == "cnn_lstm":
            layers = [CNNAttLSTM(**config) for config in configs_list]
        elif mode == "cnn_lstm_att":
            layers = [CNNALSTMAtt(**config) for config in configs_list]

        else:

            layers = [
                Regressor(output_last_layer=True, **config) for config in configs_list
            ]

        self.layers = nn.ModuleList(layers)
        regr_heads = [nn.Sequential(nn.Identity()) for _ in range(self.num_regr)]
        self.regr_heads = nn.ModuleList(regr_heads)
        self.last_layer = nn.Linear(self.num_regr * 1, 1)

    def forward(self, batch):
        sample_batch = (batch[0], batch[1])
        other_sample_batch = (batch[2], batch[3])
        outputs = []
        other_outputs = []
        for i in range(self.num_regr):
            output = self.layers[i](sample_batch)
            outputs.append(output)

        org_x = torch.cat(outputs, 1)
        x = torch.mean(org_x, 1)

        for i in range(self.num_regr):
            other_output = self.layers[i](other_sample_batch)
            other_outputs.append(other_output)

        other_org_x = torch.cat(other_outputs, 1)
        other_x = torch.mean(other_org_x, 1)
        return torch.stack([x.unsqueeze(1)] + outputs + [other_x.unsqueeze(1)], 1)


##############################
"""UNET CNN 1D"""  #############
##############################


class BaseDecBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        maxpool_size=2,
        batchnorm="default",
        down="convolution1d",
        sh=128,
        dilation=1,
        *args,
        **kwargs
    ):
        modes = nn.ModuleDict(
            [
                ["maxpooling", nn.MaxPool1d(maxpool_size)],
                [
                    "convolution1d",
                    base_conv1_layer(
                        channels[-1],
                        channels[-1],
                        kernel_size=maxpool_size,
                        dilation=dilation,
                    ),
                ],
            ]
        )
        super(BaseDecBlock, self).__init__()
        dilations = [1 if i % 2 == 0 else 2 for i in range(len(channels[1:]))]
        conv_blocks = [
            base_conv1_layer(
                in_f,
                out_f,
                kernel_size=kernel_size,
                padding=calculate_matching_padding(sh, sh, kernel_size, dilation, 1),
                dilation=dilation,
            )
            for in_f, out_f, dilation in zip(channels, channels[1:], dilations)
        ]

        conv_blocks.append(modes[down])
        self.block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.block(x)


class BaseEncBlock(nn.Module):
    def __init__(
        self,
        channels,
        ch_out=3,
        kernel_size=3,
        upsl_size=2,
        batchnorm="default",
        sh=128,
        out_sh=256,
        trans_conv=False,
        *args,
        **kwargs
    ):
        super(BaseEncBlock, self).__init__()

        modes = nn.ModuleDict(
            [
                [
                    "convolution1d",
                    nn.Sequential(
                        nn.ConvTranspose1d(
                            channels[-1],
                            ch_out,
                            kernel_size,
                            stride=2,
                            padding=1,
                            output_padding=transconv_matching_padding(
                                out_sh, sh, 2, 1, kernel_size
                            ),
                        ),
                        nn.ReLU(),
                    ),
                ]
            ]
        )
        conv_blocks = [
            base_conv1_layer(
                in_f,
                out_f,
                kernel_size=kernel_size,
                padding=calculate_matching_padding(sh, sh, kernel_size, 1, 1),
            )
            for in_f, out_f in zip(channels, channels[1:])
        ]
        if trans_conv:
            conv_blocks.append(modes["convolution1d"])
        self.block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels,
        shapes=[20, 10, 5],
        channel_len=2,
        kernels=[3, 3, 3, 3],
        maxpool_size=2,
        batchnorm="default",
        down="maxpooling",
    ):
        super(Decoder, self).__init__()
        dec_blocks = [
            BaseDecBlock(
                [c_in if i == 0 else c_out for i in range(channel_len)],
                kernel_size=kernel,
                maxpool_size=maxpool_size,
                batchnorm=batchnorm,
                down=down,
                sh=sh,
            )
            for c_in, c_out, kernel, sh in zip(channels, channels[1:], kernels, shapes)
        ]

        self.dec = nn.ModuleList(dec_blocks)

    def forward(self, x):
        outputs_to_save = [x]
        for layer in self.dec:
            x = layer(x)
            outputs_to_save.append(x)
        return outputs_to_save


def middle_layer(in_c, out_c, kernel_size, sh, out_sh):
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_c,
            out_c,
            kernel_size,
            stride=2,
            padding=1,
            output_padding=transconv_matching_padding(sh, out_sh, 2, 1, kernel_size),
        ),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        channel_len=2,
        kernel_size=3,
        upsl_size=2,
        batchnorm="default",
        dosum="True",
        shapes=[20, 10, 5],
    ):
        super(Encoder, self).__init__()
        self.dosum = dosum
        if self.dosum:
            n = 1
        else:
            n = 2
        channels = channels[::-1][1:]
        sum_channels = channels[0]
        enc_channels = []
        for ch in channels:
            sum_channels = sum_channels + ch
            enc_channels.append(sum_channels)
        self.last_channels_num = enc_channels[-1]
        enc_blocks = [
            BaseEncBlock(
                [enc_channels[i] for j in range(channel_len)],
                ch_out=enc_channels[i],
                kernel_size=kernel_size,
                upsl_size=upsl_size,
                batchnorm=batchnorm,
                sh=shapes[i],
                out_sh=shapes[i + 1],
                trans_conv=True if i < len(enc_channels) - 1 else False,
            )
            for i in range(len(enc_channels))
        ]

        self.enc = nn.ModuleList(enc_blocks)

    def forward(self, x, saved_outputs):
        for layer, y in zip(self.enc, saved_outputs[::-1][1:]):
            if not self.dosum:
                # print(x.size(), y.size())
                x = torch.cat([y, x], dim=1)
            else:
                x = x + y
            x = layer(x)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        channels=[4, 32, 64],
        channel_len=4,
        kernels=[7, 5, 5, 5, 3],
        updown_size=2,
        batchnorm="default",
        dosum=False,
        down="maxpooling",
        shapes=[20, 10, 5],
        output_las_layer=True,
        with_att=True,
    ):
        super(Unet, self).__init__()

        self.dec = Decoder(
            channels,
            channel_len=channel_len,
            kernels=kernels,
            shapes=shapes,
            maxpool_size=updown_size,
            batchnorm=batchnorm,
            down=down,
        )
        self.middle_layer = middle_layer(
            channels[-1], channels[-2], 3, shapes[-1], shapes[-2]
        )
        self.enc = Encoder(
            channels,
            channel_len=channel_len,
            kernel_size=3,
            shapes=shapes,
            upsl_size=updown_size,
            batchnorm="default",
            dosum=dosum,
        )
        sum_channels = channels[0]
        enc_channels = []
        self.final_conv_layer = base_conv1_layer(
            self.enc.last_channels_num,
            self.enc.last_channels_num,
            5,
            padding=calculate_matching_padding(shapes[0], shapes[0], 5, 1, 1),
        )
        self.self_att_layer = attention_module(embed_dim=self.enc.last_channels_num)
        self.att_layer = attention_module(embed_dim=self.enc.last_channels_num)
        self.activation_layer = nn.GELU()
        if not output_las_layer:
            self.last_layer = nn.Linear(self.enc.last_channels_num, 32)
        else:
            self.last_layer = nn.Linear(self.enc.last_channels_num, 1)
        self.with_att = with_att

    def forward(self, x, output_att=False):
        x = x[0]
        x = x.permute(0, 2, 1)
        dec_outs = self.dec(x)
        x = dec_outs[-1]
        x = self.middle_layer(x)
        x = self.enc(x, dec_outs)
        x = self.final_conv_layer(x)
        #         print(x.size())
        x = x.permute(0, 2, 1)
        if self.with_att:
            query = torch.Tensor([1 if i % 5 == 0 else 0 for i in range(x.size(2))]).to(
                device="cuda" if torch.cuda.is_available() else "cpu"
            ) / (x.size(2) ** (0.5))
            query = torch.stack([query for _ in range(x.size(0))]).unsqueeze(1)
            x, att1 = self.self_att_layer(x, x, x)
            x = self.activation_layer(x)
            x, att2 = self.att_layer(query, x, x)
        x = self.activation_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        if output_att and self.with_att:
            return (x, att1, att2)
        else:
            return x


#################################################################################
"""FCNN"""
#################################################################################
class FCNN(nn.Module):
    def __init__(
        self,
        features=[3 * 20, 128, 256, 1],
    ):
        super(FCNN, self).__init__()
        first_layers = [
            base_linear_module(in_f, out_f)
            for in_f, out_f in zip(features[:-2], features[1:])
        ]
        self.first_layers = nn.Sequential(*first_layers)
        self.last_layer = nn.Linear(features[-2], features[-1])

    def forward(self, x, output_att=False):
        x = x[0]
        x = torch.flatten(x, start_dim=1)
        x = self.first_layers(x)
        x = self.last_layer(x)
        return x


#######################################################################################
"""Block with skip connections"""
#######################################################################################
class EncSkippCOnnectedBlock(nn.Module):
    def __init__(
        self,
        channels,
        ch_out=3,
        kernel_size=5,
        sh=128,
        out_sh=256,
        trans_conv=False,
        *args,
        **kwargs
    ):
        super(EncSkippCOnnectedBlock, self).__init__()

        modes = nn.ModuleDict(
            [
                [
                    "convolution1d",
                    nn.Sequential(
                        nn.ConvTranspose1d(
                            channels[-1],
                            ch_out,
                            kernel_size,
                            stride=2,
                            padding=1,
                            output_padding=transconv_matching_padding(
                                out_sh, sh, 2, 1, kernel_size
                            ),
                        ),
                        nn.LeakyReLU(),
                        nn.BatchNorm1d(num_features=int(ch_out)),
                    ),
                ]
            ]
        )
        conv_blocks = [
            base_conv1_layer(
                in_f,
                out_f,
                kernel_size=kernel_size,
                padding=calculate_matching_padding(sh, sh, kernel_size, 1, 1),
            )
            for in_f, out_f in zip(channels, channels[1:])
        ]

        if trans_conv:
            conv_blocks.append(modes["convolution1d"])
        self.block = nn.ModuleList(conv_blocks)

    def forward(self, x):
        outputs = []
        for i in range(len(self.block)):
            z = self.block[i](x)
            outputs.append(z)
            if i == len(self.block) - 1:
                x = z
            else:
                x = torch.sum(torch.stack(outputs + [z]), 0)
        return x


class DecSkippCOnnectedBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=5,
        maxpool_size=2,
        down="convolution1d",
        sh=128,
        dilation=1,
        *args,
        **kwargs
    ):
        modes = nn.ModuleDict(
            [
                ["maxpooling", nn.MaxPool1d(maxpool_size)],
                [
                    "convolution1d",
                    base_conv1_layer(
                        channels[-1],
                        channels[-1],
                        kernel_size=maxpool_size,
                        dilation=dilation,
                    ),
                ],
            ]
        )
        super(DecSkippCOnnectedBlock, self).__init__()
        dilations = [1 if i % 2 == 0 else 1 for i in range(len(channels[1:]))]
        conv_blocks = [
            base_conv1_layer(
                in_f,
                out_f,
                kernel_size=kernel_size,
                padding=calculate_matching_padding(sh, sh, kernel_size, dilation, 1),
                dilation=dilation,
            )
            for in_f, out_f, dilation in zip(channels, channels[1:], dilations)
        ]

        conv_blocks.append(modes[down])
        self.block = nn.ModuleList(conv_blocks)

    def forward(self, x):
        outputs = []
        for i in range(len(self.block)):
            z = self.block[i](x)
            # print(z.size())
            outputs.append(z)
            if i == len(self.block) - 1:
                x = z
            else:
                x = torch.sum(torch.stack(outputs + [z]), 0)
        return x


####################### LSTM ###############################
class AttLSTM(torch.nn.Module):
    def __init__(self, n_h=16, n_l=4, with_att=True):
        super(AttLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            3, n_h, n_l, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.with_att = with_att
        if self.with_att:
            self.att = attention_module(embed_dim=n_h * 2)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(20 * n_h * 2, 1)

    def forward(self, x, output_att=False):
        x = x[0]
        x, (c, h) = self.lstm(x)
        if self.with_att:
            x, att1 = self.att(x, x, x)
            x = self.activation_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        return x


#################### LSTM + CNN ##################


class CNNContext(nn.Module):
    def __init__(
        self,
        channels=[3, 8, 16],
        channel_len=4,
        kernels=[5, 3, 3, 3, 3],
        updown_size=2,
        batchnorm="default",
        dosum=False,
        down="maxpooling",
        shapes=[20, 10, 5],
        output_las_layer=True,
        with_att=True,
        output=64,
    ):
        super(CNNContext, self).__init__()

        self.dec = Decoder(
            channels,
            channel_len=channel_len,
            kernels=kernels,
            shapes=shapes,
            maxpool_size=updown_size,
            batchnorm=batchnorm,
            down=down,
        )
        self.middle_layer = middle_layer(
            channels[-1], channels[-2], 3, shapes[-1], shapes[-2]
        )
        self.enc = Encoder(
            channels,
            channel_len=channel_len,
            kernel_size=3,
            shapes=shapes,
            upsl_size=updown_size,
            batchnorm="default",
            dosum=dosum,
        )

        self.final_conv_layer = base_conv1_layer(
            self.enc.last_channels_num,
            self.enc.last_channels_num,
            5,
            padding=calculate_matching_padding(shapes[0], shapes[0], 5, 1, 1),
        )
        self.self_att_layer = attention_module(embed_dim=self.enc.last_channels_num)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(self.enc.last_channels_num * 20, output)
        self.with_att = with_att

    def forward(self, x, output_att=False):
        x = x[0]
        x = x.permute(0, 2, 1)
        dec_outs = self.dec(x)
        x = dec_outs[-1]
        x = self.middle_layer(x)
        x = self.enc(x, dec_outs)
        x = self.final_conv_layer(x)
        #         print(x.size())
        x = x.permute(0, 2, 1)
        if self.with_att:
            x, att1 = self.self_att_layer(x, x, x)
        x = self.activation_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        if output_att and self.with_att:
            return (x, att1)
        else:
            return x


class CNNAttLSTM(torch.nn.Module):
    def __init__(self, n_h=16, n_l=4):
        super(CNNAttLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            3, n_h, n_l, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.att = attention_module(embed_dim=n_h * 2)
        self.att1 = attention_module(embed_dim=n_h * 2)
        self.context = CNNContext(output=n_h * 2)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(1 * n_h * 2, 1)

    def forward(self, x, output_att=False):
        query = self.context(x)
        query = query.unsqueeze(1)
        x = x[0]
        x, (c, h) = self.lstm(x)
        x, att1 = self.att(x, x, x)
        x = self.activation_layer(x)
        x, att2 = self.att1(query, x, x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        if output_att:
            return (x, att1, att2)
        return x


############# UNET feature extractor ###############


class UnetFeatreExtractor(nn.Module):
    def __init__(
        self,
        channels=[3, 8, 16],
        channel_len=3,
        kernels=[5, 3, 3],
        updown_size=2,
        batchnorm="default",
        dosum=False,
        down="maxpooling",
        shapes=[20, 10, 5],
        with_att=False,
    ):
        super(UnetFeatreExtractor, self).__init__()

        self.dec = Decoder(
            channels,
            channel_len=channel_len,
            kernels=kernels,
            shapes=shapes,
            maxpool_size=updown_size,
            batchnorm=batchnorm,
            down=down,
        )
        self.middle_layer = middle_layer(
            channels[-1], channels[-2], 3, shapes[-1], shapes[-2]
        )
        self.enc = Encoder(
            channels,
            channel_len=channel_len,
            kernel_size=3,
            shapes=shapes,
            upsl_size=updown_size,
            batchnorm="default",
            dosum=dosum,
        )

        self.final_conv_layer = base_conv1_layer(
            self.enc.last_channels_num,
            self.enc.last_channels_num,
            5,
            padding=calculate_matching_padding(shapes[0], shapes[0], 5, 1, 1),
        )

    def forward(self, x):
        x = x[0]
        x = x.permute(0, 2, 1)
        dec_outs = self.dec(x)
        x = dec_outs[-1]
        x = self.middle_layer(x)
        x = self.enc(x, dec_outs)
        x = self.final_conv_layer(x)
        x = x.permute(0, 2, 1)
        return x


################ CNN+LSTM #################
class CNNALSTMAtt(torch.nn.Module):
    def __init__(self, n_h=16, n_l=4):
        super(CNNALSTMAtt, self).__init__()

        self.cnn = UnetFeatreExtractor()
        self.lstm = torch.nn.LSTM(
            self.cnn.enc.last_channels_num,
            n_h,
            n_l,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.att = attention_module(embed_dim=n_h * 2)
        self.att1 = attention_module(embed_dim=n_h * 2)
        self.context = CNNContext(output=n_h * 2)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(1 * n_h * 2, 1)

    def forward(self, x, output_att=False):
        query = self.context(x)
        query = query.unsqueeze(1)
        x = self.cnn(x)
        x, (c, h) = self.lstm(x)
        x, att1 = self.att(x, x, x)
        x = self.activation_layer(x)
        x, att2 = self.att1(query, x, x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        if output_att:
            return (x, att1, att2)
        return x
