"""
Enhanced Sign Language Recognition Model (SignNet-V2)
=====================================================

Multi-stream spatiotemporal transformer architecture for Bengali Sign Language recognition.
Improvements over baseline BDSLW_SPOTER:
1. Multi-stream input (body + hands + face)
2. Hierarchical temporal modeling
3. Spatial attention mechanisms
4. Advanced augmentation pipeline
5. Mixed precision training support

Author: BDSL Recognition Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
import math
from typing import Optional, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_seq_length: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(
            max_seq_length + 2, d_model
        )  # +2 for class token and potential padding
        position = torch.arange(0, max_seq_length + 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length+2, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """1D Temporal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal convolution.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, channels, seq_len)
        """
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class SpatialAttentionBlock(nn.Module):
    """Spatial attention mechanism for landmark sequences."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply spatial attention to input features.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Attended features of same shape
        """
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attended))
        return x


class StreamSpecificEncoder(nn.Module):
    """Encoder for a specific input stream (body, left_hand, right_hand, face)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), Dropout(dropout * 0.5)
        )

        # Temporal convolutional layers
        self.temporal_convs = nn.ModuleList(
            [
                TemporalConvBlock(d_model, d_model, kernel_size=3, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Spatial attention
        self.spatial_attention = SpatialAttentionBlock(d_model, num_heads, dropout)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Encode stream-specific features.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            seq_length: Actual sequence length (before padding)

        Returns:
            Encoded features of shape (batch, seq_len, d_model)
        """
        # Project input
        x = self.input_projection(x)

        # Apply temporal convolutions
        for conv in self.temporal_convs:
            x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x_conv = conv(x_conv)
            x = x_conv.transpose(1, 2)  # (batch, seq_len, d_model)

        # Apply spatial attention with mask
        mask = self._create_attention_mask(seq_length, x.size(1), x.device)
        x = self.spatial_attention(x, mask)

        # Add positional encoding
        x = self.positional_encoding(x)

        return x

    def _create_attention_mask(
        self, seq_length: int, max_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(max_len, max_len, device=device), diagonal=1
        ).bool()
        return mask


class CrossStreamFusion(nn.Module):
    """Multi-stream fusion using cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_streams: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_streams = num_streams

        # Cross-attention for each stream
        self.cross_attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_streams)
            ]
        )

        # Fusion projections
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * num_streams, d_model),
            nn.LayerNorm(d_model),
            Dropout(dropout),
        )

        self.norm = LayerNorm(d_model)

    def forward(self, stream_features: list, stream_lengths: list) -> torch.Tensor:
        """Fuse features from multiple streams using cross-attention.

        Args:
            stream_features: List of tensors, each (batch, seq_len, d_model)
            stream_lengths: List of actual sequence lengths for each stream

        Returns:
            Fused features of shape (batch, seq_len, d_model)
        """
        batch_size = stream_features[0].size(0)
        max_len = max(f.size(1) for f in stream_features)

        # Pad all streams to same length
        padded_streams = []
        for features, length in zip(stream_features, stream_lengths):
            if features.size(1) < max_len:
                padding = torch.zeros(
                    batch_size,
                    max_len - features.size(1),
                    self.d_model,
                    device=features.device,
                    dtype=features.dtype,
                )
                padded = torch.cat([features, padding], dim=1)
            else:
                padded = features
            padded_streams.append(padded)

        # Stack streams as additional "time" dimension: (batch, num_streams * seq_len, d_model)
        # Or concat along feature dimension
        concat_features = torch.cat(
            padded_streams, dim=-1
        )  # (batch, max_len, d_model * num_streams)

        # Project to single stream
        fused = self.fusion_proj(concat_features)

        # Apply cross-attention between streams
        attended = []
        for i, stream in enumerate(padded_streams):
            attn_output, _ = self.cross_attentions[i](stream, fused, fused)
            attended.append(attn_output)

        # Combine attended features
        combined = torch.mean(torch.stack(attended), dim=0)  # Average across streams
        combined = self.norm(combined + stream_features[0])  # Residual connection

        return combined


class HierarchicalTemporalEncoder(nn.Module):
    """Hierarchical temporal encoder using multi-scale temporal windows."""

    def __init__(
        self,
        d_model: int,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_scales = num_scales

        # Multi-scale temporal attention
        self.temporal_attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_scales)
            ]
        )

        # Temporal pooling for each scale
        self.temporal_pools = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=2**s, stride=2**s) if s > 0 else nn.Identity()
                for s in range(num_scales)
            ]
        )

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.LayerNorm(d_model),
            Dropout(dropout),
        )

        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical temporal encoding.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Temporally encoded features
        """
        batch_size, seq_len, d_model = x.shape

        # Apply multi-scale processing
        scale_features = []
        for i, (attention, pool) in enumerate(
            zip(self.temporal_attentions, self.temporal_pools)
        ):
            if i == 0:
                scale_x = x
            else:
                # Pool and interpolate back
                x_pooled = x.transpose(1, 2)  # (batch, d_model, seq_len)
                x_pooled = pool(x_pooled)
                scale_len = x_pooled.size(2)
                scale_x = x_pooled.transpose(1, 2)  # (batch, scale_len, d_model)

                # Interpolate to original length
                if scale_len != seq_len:
                    scale_x = F.interpolate(
                        scale_x.transpose(1, 2),
                        size=seq_len,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

            # Apply temporal attention
            attended, _ = attention(scale_x, scale_x, scale_x)
            scale_features.append(attended)

        # Concatenate and fuse
        concat = torch.cat(
            scale_features, dim=-1
        )  # (batch, seq_len, d_model * num_scales)
        fused = self.scale_fusion(concat)

        return self.norm(x + fused)


class GlobalTemporalEncoder(nn.Module):
    """Transformer encoder for global temporal modeling."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer encoding to temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional mask for padding (True = mask)

        Returns:
            Encoded features of same shape
        """
        encoded = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        return self.norm(encoded)


class ClassificationHead(nn.Module):
    """Multi-layer classification head with residual connections."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input features.

        Args:
            x: Input tensor of shape (batch, d_model)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        return self.classifier(x)


class SignNetV2(nn.Module):
    """
    SignNet-V2: Enhanced Multi-Stream Spatiotemporal Transformer for Sign Language Recognition

    Architecture:
    1. Multi-stream input processing (body: 33 landmarks, left_hand: 21, right_hand: 21, face: 468)
    2. Stream-specific encoders with temporal convolutions and spatial attention
    3. Cross-stream fusion using attention mechanisms
    4. Hierarchical temporal encoder for multi-scale modeling
    5. Global transformer encoder for sequence understanding
    6. Classification head for sign prediction

    Key improvements over baseline SPOTER:
    - Multi-stream architecture captures hand and facial expressions
    - Hierarchical temporal modeling handles variable-length sequences
    - Cross-stream attention learns inter-stream relationships
    - Enhanced regularization through dropout and label smoothing
    """

    def __init__(
        self,
        num_classes: int = 72,
        body_dim: int = 99,  # 33 landmarks * 3 coords
        hand_dim: int = 63,  # 21 landmarks * 3 coords
        face_dim: int = 1404,  # 468 landmarks * 3 coords
        d_model: int = 128,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.2,
        max_seq_length: int = 150,
        use_face: bool = True,
        use_hands: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.use_face = use_face
        self.use_hands = use_hands

        # Calculate input dimensions based on enabled streams
        self.body_dim = body_dim
        self.hand_dim = hand_dim
        self.face_dim = face_dim

        # Stream-specific encoders
        self.body_encoder = StreamSpecificEncoder(
            input_dim=body_dim,
            d_model=d_model,
            num_layers=2,
            num_heads=num_heads // 2,
            dropout=dropout,
        )

        if use_hands:
            self.left_hand_encoder = StreamSpecificEncoder(
                input_dim=hand_dim,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads // 4,
                dropout=dropout,
            )

            self.right_hand_encoder = StreamSpecificEncoder(
                input_dim=hand_dim,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads // 4,
                dropout=dropout,
            )

        if use_face:
            self.face_encoder = StreamSpecificEncoder(
                input_dim=face_dim,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads // 2,
                dropout=dropout,
            )

        # Cross-stream fusion
        num_streams = 1 + (2 if use_hands else 0) + (1 if use_face else 0)
        self.cross_stream_fusion = CrossStreamFusion(
            d_model=d_model,
            num_streams=num_streams,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Hierarchical temporal encoder
        self.hierarchical_encoder = HierarchicalTemporalEncoder(
            d_model=d_model, num_scales=3, num_heads=num_heads, dropout=dropout
        )

        # Global temporal encoder
        self.global_encoder = GlobalTemporalEncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Class token for classification
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        # Classification head
        self.classifier = ClassificationHead(
            d_model=d_model, num_classes=num_classes, dropout=dropout
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of SignNet-V2.

        Args:
            body_pose: Body pose landmarks (batch, seq_len, body_dim)
            left_hand: Left hand landmarks (batch, seq_len, hand_dim) or None
            right_hand: Right hand landmarks (batch, seq_len, hand_dim) or None
            face: Face landmarks (batch, seq_len, face_dim) or None
            attention_mask: Attention mask for padding (batch, seq_len)

        Returns:
            Class logits (batch, num_classes)
        """
        batch_size = body_pose.size(0)

        # Get sequence length from body pose
        seq_length = body_pose.size(1)

        # Encode each stream
        stream_features = []
        stream_lengths = []

        # Body pose encoding
        body_features = self.body_encoder(body_pose, seq_length)
        stream_features.append(body_features)
        stream_lengths.append(seq_length)

        # Hand encoding
        if self.use_hands and left_hand is not None:
            left_features = self.left_hand_encoder(left_hand, seq_length)
            stream_features.append(left_features)
            stream_lengths.append(seq_length)

        if self.use_hands and right_hand is not None:
            right_features = self.right_hand_encoder(right_hand, seq_length)
            stream_features.append(right_features)
            stream_lengths.append(seq_length)

        # Face encoding
        if self.use_face and face is not None:
            face_features = self.face_encoder(face, seq_length)
            stream_features.append(face_features)
            stream_lengths.append(seq_length)

        # Cross-stream fusion
        fused_features = self.cross_stream_fusion(stream_features, stream_lengths)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, fused_features], dim=1)

        # Update attention mask to include class token
        if attention_mask is not None:
            class_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            full_mask = torch.cat([class_mask, attention_mask], dim=1)
            # Convert to transformer format (True = mask)
            transformer_mask = full_mask == 0
        else:
            transformer_mask = None

        # Hierarchical temporal encoding
        x = self.hierarchical_encoder(x)

        # Global temporal encoding
        x = self.global_encoder(x, transformer_mask)

        # Use class token for classification
        class_representation = x[:, 0]

        # Classification
        logits = self.classifier(class_representation)

        return logits

    def get_embedding(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get feature embedding for downstream tasks."""
        batch_size = body_pose.size(0)
        seq_length = body_pose.size(1)

        # Encode each stream
        stream_features = []
        stream_lengths = [seq_length] * len(stream_features)

        body_features = self.body_encoder(body_pose, seq_length)
        stream_features.append(body_features)

        if self.use_hands and left_hand is not None:
            left_features = self.left_hand_encoder(left_hand, seq_length)
            stream_features.append(left_features)

        if self.use_hands and right_hand is not None:
            right_features = self.right_hand_encoder(right_hand, seq_length)
            stream_features.append(right_features)

        if self.use_face and face is not None:
            face_features = self.face_encoder(face, seq_length)
            stream_features.append(face_features)

        # Cross-stream fusion
        fused = self.cross_stream_fusion(stream_features, stream_lengths)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, fused], dim=1)

        # Apply encoders
        x = self.hierarchical_encoder(x)
        x = self.global_encoder(x)

        return x[:, 0]  # Return class token embedding


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Test model
    model = SignNetV2(
        num_classes=72,
        body_dim=99,
        hand_dim=63,
        face_dim=1404,
        d_model=128,
        num_encoder_layers=4,
        num_heads=8,
        d_ff=512,
        dropout=0.2,
        use_face=True,
        use_hands=True,
    )

    params = count_parameters(model)
    print(
        f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable"
    )
    print(f"Model size: {params['total'] * 4 / 1024**2:.2f} MB")

    # Test forward pass
    batch_size = 4
    seq_length = 150

    body_pose = torch.randn(batch_size, seq_length, 99)
    left_hand = torch.randn(batch_size, seq_length, 63)
    right_hand = torch.randn(batch_size, seq_length, 63)
    face = torch.randn(batch_size, seq_length, 1404)
    mask = torch.ones(batch_size, seq_length)

    with torch.no_grad():
        logits = model(body_pose, left_hand, right_hand, face, mask)

    print(
        f"Input shapes: body={body_pose.shape}, hands=({left_hand.shape}, {right_hand.shape}), face={face.shape}"
    )
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 72)")
