from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class TableEncoderCNNLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        window_size: int,
        num_features: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        bidirectional: bool,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.use_lstm = use_lstm
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(8, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(4, eps=1e-4, affine=False)

        dummy = torch.randn(1, input_channels, num_features, window_size)
        _, channels, height, steps = self._forward_conv(dummy).shape
        step_dim = channels * height

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=step_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            fc_input = lstm_hidden_dim * (2 if bidirectional else 1)
        else:
            fc_input = step_dim * steps
        self.fc = nn.Linear(fc_input, output_dim)

    def _forward_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pool(F.leaky_relu(self.bn1(self.conv1(inputs))))
        return self.pool(F.leaky_relu(self.bn2(self.conv2(outputs))))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self._forward_conv(inputs)
        batch_size, channels, height, steps = outputs.shape
        if self.use_lstm:
            outputs = outputs.permute(0, 3, 1, 2).contiguous().view(batch_size, steps, channels * height)
            outputs, _ = self.lstm(outputs)
            outputs = outputs[:, -1, :]
        else:
            outputs = outputs.view(batch_size, -1)
        return self.fc(outputs)


class ResNet50ImageEncoder(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int, pretrained: bool):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = models.resnet50(weights=weights)
        except Exception as exc:
            print(f"Failed to load pretrained ResNet50 weights: {exc}. Falling back to random init.")
            backbone = models.resnet50(weights=None)

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=3, bias=False)
        if input_channels == 3:
            self.conv1.weight.data = backbone.conv1.weight.data
        else:
            expanded = backbone.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
            self.conv1.weight.data = expanded

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, feature_dim)

        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("layer4") or name.startswith("fc")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer4(self.layer3(self.layer2(self.layer1(outputs))))
        outputs = self.avgpool(outputs)
        return self.fc(outputs.view(outputs.size(0), -1))


class FusionIBLayer(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)
        self.fc_logvar = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, table_feature: torch.Tensor, image_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fusion_input = torch.cat([table_feature, image_feature], dim=1)
        mu = self.fc_mu(fusion_input)
        logvar = self.fc_logvar(fusion_input)
        std = torch.exp(0.5 * logvar)
        feature = mu + torch.randn_like(std) * std
        kl_loss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - logvar - 1, dim=1).mean()
        return feature, kl_loss


class L2MLPFusion(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        hidden_dim = max(input_dim, bottleneck_dim)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim))

    def forward(self, table_feature: torch.Tensor, image_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fusion_input = torch.cat([table_feature, image_feature], dim=1)
        l2_sum = torch.zeros(1, device=fusion_input.device, dtype=fusion_input.dtype)
        total_params = 0
        for parameter in self.parameters():
            l2_sum = l2_sum + parameter.pow(2).sum()
            total_params += parameter.numel()
        return self.net(fusion_input), l2_sum / max(total_params, 1)


class AttentionConcatFusion(nn.Module):
    def __init__(self, table_dim: int, image_dim: int, bottleneck_dim: int):
        super().__init__()
        self.table_projection = nn.Linear(table_dim, bottleneck_dim)
        self.image_projection = nn.Linear(image_dim, bottleneck_dim)
        self.attention = nn.Sequential(nn.Linear(table_dim + image_dim, bottleneck_dim), nn.Tanh(), nn.Linear(bottleneck_dim, 2))
        self.output_projection = nn.Sequential(nn.Linear(bottleneck_dim * 2, bottleneck_dim), nn.GELU())

    def forward(self, table_feature: torch.Tensor, image_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fusion_input = torch.cat([table_feature, image_feature], dim=1)
        weights = torch.softmax(self.attention(fusion_input), dim=1)
        attended = torch.cat(
            [
                weights[:, 0:1] * self.table_projection(table_feature),
                weights[:, 1:2] * self.image_projection(image_feature),
            ],
            dim=1,
        )
        return self.output_projection(attended), torch.zeros(1, device=fusion_input.device, dtype=fusion_input.dtype)


class SimpleConcatFusion(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, table_feature: torch.Tensor, image_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fusion_input = torch.cat([table_feature, image_feature], dim=1)
        return self.projection(fusion_input), torch.zeros(1, device=fusion_input.device, dtype=fusion_input.dtype)


class UniCO(nn.Module):
    def __init__(
        self,
        table_input_channels: int,
        table_feature_dim: int,
        num_features: int,
        window_size: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        bidirectional: bool,
        image_input_channels: int,
        image_feature_dim: int,
        fusion_bottleneck_dim: int,
        final_dim: int,
        fusion_type: str,
        pretrained_image_encoder: bool,
        use_lstm: bool = True,
        use_image: bool = True,
    ):
        super().__init__()
        self.use_image = use_image
        self.table_encoder = TableEncoderCNNLSTM(
            table_input_channels,
            table_feature_dim,
            window_size,
            num_features,
            lstm_hidden_dim,
            lstm_layers,
            bidirectional,
            use_lstm,
        )

        if use_image:
            self.image_encoder = ResNet50ImageEncoder(image_input_channels, image_feature_dim, pretrained_image_encoder)
            fusion_input_dim = table_feature_dim + image_feature_dim
            if fusion_type == "ib":
                self.fusion_layer = FusionIBLayer(fusion_input_dim, fusion_bottleneck_dim)
            elif fusion_type == "l2_mlp":
                self.fusion_layer = L2MLPFusion(fusion_input_dim, fusion_bottleneck_dim)
            elif fusion_type == "attention_concat":
                self.fusion_layer = AttentionConcatFusion(table_feature_dim, image_feature_dim, fusion_bottleneck_dim)
            elif fusion_type == "concat":
                self.fusion_layer = SimpleConcatFusion(fusion_input_dim, fusion_bottleneck_dim)
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")
            self.test_projection = nn.Linear(table_feature_dim, fusion_bottleneck_dim)
            self.fc_final = nn.Linear(table_feature_dim + fusion_bottleneck_dim, final_dim)
        else:
            self.image_encoder = None
            self.fusion_layer = None
            self.test_projection = None
            self.fc_final = nn.Linear(table_feature_dim, final_dim)

    def forward(self, table_data: torch.Tensor, image_data: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        table_feature = self.table_encoder(table_data)
        if not self.use_image:
            return self.fc_final(table_feature), torch.zeros(1, device=table_data.device, dtype=table_data.dtype)
        if image_data is None:
            fused_feature = self.test_projection(table_feature)
            regularization_loss = torch.zeros(1, device=table_data.device, dtype=table_data.dtype)
        else:
            image_feature = self.image_encoder(image_data)
            fused_feature, regularization_loss = self.fusion_layer(table_feature, image_feature)
        return self.fc_final(torch.cat([table_feature, fused_feature], dim=1)), regularization_loss
