"""
HAAC (Hybrid-Action Actor-Critic) 网络
用于方案3：伴随法作为超级动作
"""

import torch
import torch.nn as nn


class HAAC_Network(nn.Module):
    def __init__(self,
                 input_channels: int = 2,
                 base_channels: int = 32,
                 n_cells_x: int = 120,
                 n_cells_y: int = 48):
        """
        HAAC: Hybrid-Action Actor-Critic Network
        专为 120x48 网格设计的混合动作网络
        """
        super().__init__()

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.n_pixels = n_cells_x * n_cells_y

        # ================================
        # 1. 编码器 (Encoder)
        # ================================
        self.enc1 = self._conv_block(input_channels, base_channels)      # -> [B, 32, 120, 48]
        self.pool1 = nn.MaxPool2d(2)                                     # -> [B, 32, 60, 24]

        self.enc2 = self._conv_block(base_channels, base_channels * 2)   # -> [B, 64, 60, 24]
        self.pool2 = nn.MaxPool2d(2)                                     # -> [B, 64, 30, 12]

        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)  # -> [B, 128, 30, 12]
        self.pool3 = nn.MaxPool2d(2)                                        # -> [B, 128, 15, 6]

        # 瓶颈层 (Bottleneck)
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)  # -> [B, 256, 15, 6]

        # ================================
        # 2. 空间策略头 (Pixel Head)
        # ================================
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)  # Skip from enc3

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)  # Skip from enc2

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)      # Skip from enc1

        self.pixel_out = nn.Conv2d(base_channels, 1, kernel_size=1)  # -> [B, 1, 120, 48]

        # ================================
        # 3. 全局策略头 (Adjoint)
        # ================================
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.adjoint_fc = nn.Sequential(
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ================================
        # 4. 评价头 (Critic)
        # ================================
        self.critic_fc = nn.Sequential(
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """标准的 Conv-BN-ReLU 块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 2, 120, 48] (Structure + Gradient)
        Returns:
            action_logits: [B, 5761] (5760 pixels + 1 adjoint)
            state_value: [B, 1]
        """
        # --- Encoding ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))  # [B, 256, 15, 6]

        # --- Pixel Head (Decoding with Skips) ---
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        pixel_logits = self.pixel_out(d1).view(x.size(0), -1)  # [B, 5760]

        # --- Global Head (Adjoint) ---
        global_feat = self.global_pool(b).flatten(1)  # [B, 256]
        adjoint_logit = self.adjoint_fc(global_feat)  # [B, 1]

        # --- Fusion ---
        action_logits = torch.cat([pixel_logits, adjoint_logit], dim=1)

        # --- Critic ---
        state_value = self.critic_fc(global_feat)

        return action_logits, state_value

    def get_action_mask(self, last_action_was_adjoint: bool) -> torch.Tensor:
        """
        生成动作掩码
        如果上一步刚跑了伴随法，这一步禁止再跑伴随法 (mask=1 表示屏蔽)
        """
        mask = torch.zeros(self.n_pixels + 1)
        if last_action_was_adjoint:
            mask[-1] = 1.0
        return mask
