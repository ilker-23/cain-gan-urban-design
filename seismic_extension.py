"""
Seismic-Aware CAIN-GAN Extension
=================================
CAIN-GAN'a sismik risk farkındalığı ekleyen uzantı modülü.

Yenilikler (orijinal CAIN-GAN'a göre):
1. SeismicAttentionGate: yüksek sismik riskli bölgelerde dikkat ağırlığı
2. seismic_loss: yüksek riskli bölgelerde yüksek bina cezalandırması
3. terrain_loss: topografik tutarlılık (eğimli arazide bina yerleşimi)
4. CityConditionalNorm: şehir-spesifik feature normalizasyonu

Akademik gerekçe:
- Türkiye 1. derece deprem ülkesi → yapay yer planlamasında sismik kısıt zorunlu
- Mevcut GAN-tabanlı urban design literatüründe sismik farkındalık yok
- Bu uzantı Q1 SCI dergisi için özgün metodolojik katkıdır
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# 1. Seismic Attention Gate
# ============================================================

class SeismicAttentionGate(nn.Module):
    """
    Sismik risk haritası ile feature map'leri modüle eden attention gate.
    Yüksek riskli bölgelerde generator'ın daha temkinli üretim yapmasını sağlar.

    Input:
        features: (B, C, H, W) — encoder/bottleneck çıktısı
        seismic_map: (B, 1, H, W) — normalize edilmiş PGA haritası [0,1]

    Output:
        gated_features: (B, C, H, W) — sismik-uyarlanmış özellikler
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.zeros(1))  # öğrenilebilir karışım katsayısı

    def forward(self, features: torch.Tensor, seismic_map: torch.Tensor) -> torch.Tensor:
        # Sismik haritayı feature boyutuna eşle
        if seismic_map.shape[-2:] != features.shape[-2:]:
            seismic_map = F.interpolate(
                seismic_map,
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Gate hesaplama
        combined = torch.cat([features, seismic_map], dim=1)
        gate = self.gate_conv(combined)

        # Sismik bölgelerde feature suppression
        # gate ∈ [0,1]: yüksek risk → gate düşük → feature azalır
        gated = features * gate

        # Residual karışım (öğrenilebilir)
        return features + self.beta * (gated - features)


# ============================================================
# 2. City-Conditional Normalization
# ============================================================

class CityConditionalNorm(nn.Module):
    """
    Şehir-spesifik instance normalization.
    Elazığ ve İstanbul için ayrı affine parametreleri öğrenir.

    AdaIN benzeri bir yaklaşım — model multi-city eğitimde
    şehir karakteristiklerini ayrıştırabilir.
    """

    def __init__(self, num_features: int, num_cities: int = 2):
        super().__init__()
        self.num_features = num_features
        self.num_cities = num_cities

        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)

        # Her şehir için ayrı affine parametreler
        self.gamma = nn.Parameter(torch.ones(num_cities, num_features))
        self.beta = nn.Parameter(torch.zeros(num_cities, num_features))

    def forward(self, x: torch.Tensor, city_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            city_index: (B,) — LongTensor, her örnek için şehir indeksi
        """
        normalized = self.instance_norm(x)

        # Şehir-spesifik affine: (B, C, 1, 1)
        gamma = self.gamma[city_index].unsqueeze(-1).unsqueeze(-1)
        beta = self.beta[city_index].unsqueeze(-1).unsqueeze(-1)

        return gamma * normalized + beta


# ============================================================
# 3. Seismic-Aware Loss Functions
# ============================================================

def seismic_height_loss(
    predicted_height: torch.Tensor,
    seismic_map: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Yüksek sismik riskli bölgelerde yüksek binaları cezalandıran kayıp.

    L_seismic = mean(predicted_height · seismic_map · indicator(risk > threshold))

    Args:
        predicted_height: (B, 1, H, W) — model çıktısı, [0,1]
        seismic_map: (B, 1, H, W) — sismik risk, [0,1]
        threshold: bu değerin üstündeki bölgeler "yüksek risk"

    Returns:
        Skaler kayıp (yüksek = kötü)
    """
    high_risk_mask = (seismic_map > threshold).float()
    penalized = predicted_height * seismic_map * high_risk_mask
    return penalized.mean()


def terrain_consistency_loss(
    predicted_footprint: torch.Tensor,
    dem: torch.Tensor,
    slope_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Aşırı eğimli arazilerde bina yerleşimini cezalandırır.

    Sobel-türevi ile yerel eğim hesaplanır → eğim büyük olduğunda
    bina footprint'i azaltılmaya teşvik edilir.

    Args:
        predicted_footprint: (B, 1, H, W)
        dem: (B, 1, H, W) — normalize edilmiş yükseklik
        slope_threshold: bu değerin üstündeki eğimler cezalandırılır
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=dem.dtype, device=dem.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=dem.dtype, device=dem.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(dem, sobel_x, padding=1)
    grad_y = F.conv2d(dem, sobel_y, padding=1)
    slope = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # Normalize ve eşikle
    slope_normalized = slope / (slope.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    steep_mask = (slope_normalized > slope_threshold).float()

    penalty = predicted_footprint * steep_mask
    return penalty.mean()


def composite_cain_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    discriminator_output: torch.Tensor,
    real_label: torch.Tensor,
    seismic_map: Optional[torch.Tensor] = None,
    dem: Optional[torch.Tensor] = None,
    output_type: str = "footprint",
    lambda_rec: float = 100.0,
    lambda_adv: float = 1.0,
    lambda_seismic: float = 10.0,
    lambda_terrain: float = 5.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Tam Seismic-CAIN-GAN generator kayıp fonksiyonu.

    L_total = λ_rec · L1 + λ_adv · L_adv
             + λ_seismic · L_seismic (sadece height aşamasında)
             + λ_terrain · L_terrain (sadece footprint aşamasında)

    Returns:
        (toplam_loss, kayıp_bileşenleri_dict)
    """
    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    L_rec = l1(predicted, target)
    L_adv = bce(discriminator_output, real_label)

    loss = lambda_rec * L_rec + lambda_adv * L_adv
    components = {
        "L_rec": L_rec.item(),
        "L_adv": L_adv.item(),
    }

    # Aşamaya özel uzantılar
    if output_type == "height" and seismic_map is not None:
        L_seismic = seismic_height_loss(predicted, seismic_map)
        loss = loss + lambda_seismic * L_seismic
        components["L_seismic"] = L_seismic.item()

    if output_type == "footprint" and dem is not None:
        L_terrain = terrain_consistency_loss(predicted, dem)
        loss = loss + lambda_terrain * L_terrain
        components["L_terrain"] = L_terrain.item()

    components["L_total"] = loss.item()
    return loss, components


# ============================================================
# 4. Seismic-Aware Generator (CAIN-GAN uzantısı)
# ============================================================

class SeismicCAINGenerator(nn.Module):
    """
    CAIN-GAN generator'a seismic attention gate eklenmiş varyant.
    Mevcut cain_architecture.py'daki generator'larla uyumlu drop-in replacement.

    Mimari:
        Encoder → DualPath Bottleneck → SeismicAttentionGate → Decoder
    """

    def __init__(
        self,
        conditional_channels: int = 10,
        output_channels: int = 1,
        ngf: int = 64,
        use_city_norm: bool = True,
        num_cities: int = 2,
    ):
        super().__init__()
        self.ngf = ngf
        self.use_city_norm = use_city_norm
        self.num_cities = num_cities

        # Lazy import — cain_architecture import döngüsünü önle
        from cain_architecture import ConvBlock, DualPathBottleneck

        self.encoder = nn.Sequential(
            ConvBlock(conditional_channels, ngf, kernel_size=7, padding=3, activation="relu"),
            ConvBlock(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, activation="relu"),
            ConvBlock(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, activation="relu"),
            ConvBlock(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, activation="relu"),
        )

        self.bottleneck = DualPathBottleneck(ngf * 8)

        # Seismic-aware modülasyon
        self.seismic_gate = SeismicAttentionGate(ngf * 8)

        # City-conditional norm (multi-city için)
        if self.use_city_norm:
            self.city_norm = CityConditionalNorm(ngf * 8, num_cities=num_cities)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        conditional: torch.Tensor,
        seismic_map: Optional[torch.Tensor] = None,
        city_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            conditional: (B, C, 256, 256) — tüm conditional kanallar
            seismic_map: (B, 1, 256, 256) — sismik risk haritası (opsiyonel)
            city_index: (B,) — şehir indeksleri (opsiyonel)
        """
        features = self.encoder(conditional)
        features = self.bottleneck(features)

        # Sismik modülasyon
        if seismic_map is not None:
            features = self.seismic_gate(features, seismic_map)

        # Şehir-spesifik norm
        if self.use_city_norm and city_index is not None:
            features = self.city_norm(features, city_index)

        return self.decoder(features)


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    print("Seismic-CAIN-GAN Extension — Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 10, 256, 256

    # Test inputs
    conditional = torch.randn(B, C, H, W).to(device)
    seismic = torch.rand(B, 1, H, W).to(device)
    dem = torch.rand(B, 1, H, W).to(device)
    city_idx = torch.tensor([0, 1]).to(device)

    # Model
    gen = SeismicCAINGenerator(conditional_channels=C).to(device)

    # Forward
    footprint = gen(conditional, seismic_map=seismic, city_index=city_idx)
    print(f"✅ Generator output: {footprint.shape}")

    # Loss test
    target = torch.rand(B, 1, H, W).to(device)
    d_out = torch.rand(B, 1, 1, 1).to(device)
    real_lbl = torch.ones(B, 1, 1, 1).to(device)

    loss, comps = composite_cain_loss(
        predicted=footprint,
        target=target,
        discriminator_output=d_out,
        real_label=real_lbl,
        seismic_map=seismic,
        dem=dem,
        output_type="footprint",
    )
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"   Bileşenler: {comps}")

    n_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"✅ Toplam parametre: {n_params:,}")
