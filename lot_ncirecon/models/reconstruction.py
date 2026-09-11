"""Interpretable unfolded LOT-NCIRecon generator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
def relu(x, lambd):
    lambd = nn.functional.relu(lambd)
    return nn.functional.relu(x - lambd.to(x.device))


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)

# =============================================================================
# Reproducibility
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

def get_all_conv(net, conv_list=None):
    """Collect Conv2d and ConvTranspose2d layers without using a mutable default list."""
    if conv_list is None:
        conv_list = []
    for layer in net.children():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_list.append(layer)
        else:
            get_all_conv(layer, conv_list)
    return conv_list

class UnfoldedReconstructionNet(nn.Module):
    def __init__(self, kernel_size=3, hidden_layer_width_list=[128, 96, 64], n_classes=64, ista_num_steps=5,
                 lasso_lambda_scalar=0.01):
        """
        ISTA-based UNet for CT Denoising with Multi-Contrast Fusion
        Theory: n = D^S ⊗ S + D^P ⊗ P
                l_tilde = E^S ⊗ S + E^Q ⊗ Q + E^M ⊗ M
                l_hat = F^S ⊗ S + F^Q ⊗ Q
                n_hat = G^S ⊗ S + G^P ⊗ P
        """
        super().__init__()

        self.n_classes = n_classes
        self.ista_num_steps = ista_num_steps
        self.lasso_lambda_scalar = lasso_lambda_scalar
        self.hidden_layer_width_list = hidden_layer_width_list
        self.num_layers = len(hidden_layer_width_list)

        # ==================== Input Projections ====================
        # 增强投影网络 - 使用更深的网络
        self.proj_shared = nn.Sequential(
            nn.Conv2d(1, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, 3, 1, 1)
        )
        self.proj_contrast = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, 3, 1, 1)
        )
        self.proj_enhancement = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, 3, 1, 1)
        )
        self.proj_noise = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, 3, 1, 1)
        )



        # ==================== 特征融合权重 ====================
        self.alpha_S = nn.Parameter(torch.tensor(0.4))  # 共享特征权重
        self.alpha_P = nn.Parameter(torch.tensor(0.6))  # 平扫特异性权重
        self.alpha_Q = nn.Parameter(torch.tensor(0.6))  # 增强特异性权重

        # ==================== Shared Anatomical Features (S) ====================
        # Dictionary for shared features S in non-contrast image (n)
        self.encoder_dictionary_S_n = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='linear') for conv in
         get_all_conv(self.encoder_dictionary_S_n)];
        self.precond_encoder_dictionary_S_n = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_S_n.load_state_dict(self.encoder_dictionary_S_n.state_dict())
        self.adjoint_encoder_dictionary_S_n = adjoint_dictionary_model(self.precond_encoder_dictionary_S_n)

        # Dictionary for shared features S in low-dose contrast image (l_tilde)
        self.encoder_dictionary_S_l = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_S_l = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_S_l.load_state_dict(self.encoder_dictionary_S_l.state_dict())
        self.adjoint_encoder_dictionary_S_l = adjoint_dictionary_model(self.precond_encoder_dictionary_S_l)

        # New: G_S to replace D_S and E_S transposed operations
        self.encoder_dictionary_G_S = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='linear') for conv in
         get_all_conv(self.encoder_dictionary_G_S)];
        self.precond_encoder_dictionary_G_S = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_G_S.load_state_dict(self.encoder_dictionary_G_S.state_dict())
        self.adjoint_encoder_dictionary_G_S = adjoint_dictionary_model(self.precond_encoder_dictionary_G_S)

        # Reconstruction dictionaries for S
        self.decoder_dictionary_S_l = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_S_l.load_state_dict(self.encoder_dictionary_S_n.state_dict())
        self.decoder_dictionary_S_n = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_S_n.load_state_dict(self.encoder_dictionary_S_n.state_dict())

        # ==================== Contrast-specific Features (P) ====================
        self.encoder_dictionary_P_n = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='linear') for conv in
         get_all_conv(self.encoder_dictionary_P_n)];
        self.precond_encoder_dictionary_P_n = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_P_n.load_state_dict(self.encoder_dictionary_P_n.state_dict())
        self.adjoint_encoder_dictionary_P_n = adjoint_dictionary_model(self.precond_encoder_dictionary_P_n)
        self.decoder_dictionary_P_n = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_P_n.load_state_dict(self.encoder_dictionary_P_n.state_dict())


        # ==================== Enhancement-specific Features (Q) ====================
        self.encoder_dictionary_Q_l = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='linear') for conv in
         get_all_conv(self.encoder_dictionary_Q_l)];
        self.precond_encoder_dictionary_Q_l = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_Q_l.load_state_dict(self.encoder_dictionary_Q_l.state_dict())
        self.adjoint_encoder_dictionary_Q_l = adjoint_dictionary_model(self.precond_encoder_dictionary_Q_l)
        self.decoder_dictionary_Q_l = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_Q_l.load_state_dict(self.encoder_dictionary_Q_l.state_dict())


        # ==================== Noise/Artifact Features (M) ====================
        self.encoder_dictionary_M = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_M = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_M.load_state_dict(self.encoder_dictionary_M.state_dict())
        self.adjoint_encoder_dictionary_M = adjoint_dictionary_model(self.precond_encoder_dictionary_M)
        self.decoder_dictionary_M = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_M.load_state_dict(self.encoder_dictionary_M.state_dict())

        # ==================== Stepsize and Lambda Parameters ====================
        with torch.no_grad():
            L_S = self.power_iteration_conv_model(self.encoder_dictionary_S_n, num_simulations=20)
            L_P = self.power_iteration_conv_model(self.encoder_dictionary_P_n, num_simulations=20)
            L_Q = self.power_iteration_conv_model(self.encoder_dictionary_Q_l, num_simulations=20)
            L_M = self.power_iteration_conv_model(self.encoder_dictionary_M, num_simulations=20)

        # 调整步长 - 为不同特征设置不同的学习率
        # Use ParameterList, otherwise these learnable step sizes will not be registered by PyTorch.
        self.ista_stepsize_S = nn.ParameterList([nn.Parameter(torch.ones(1) / L_S * 0.8) for _ in range(ista_num_steps)])
        self.ista_stepsize_P = nn.ParameterList([nn.Parameter(torch.ones(1) / L_P * 1.2) for _ in range(ista_num_steps)])
        self.ista_stepsize_Q = nn.ParameterList([nn.Parameter(torch.ones(1) / L_Q * 1.2) for _ in range(ista_num_steps)])
        self.ista_stepsize_M = nn.ParameterList([nn.Parameter(torch.ones(1) / L_M * 1.5) for _ in range(ista_num_steps)])

        # Lambda parameters for each layer - 调整稀疏约束
        _lambda_S_list = [[nn.Parameter(lasso_lambda_scalar * 0.8 * torch.ones(1, width, 1, 1))
                           for width in hidden_layer_width_list] for _ in range(ista_num_steps)]
        _lambda_P_list = [[nn.Parameter(lasso_lambda_scalar * 1.2 * torch.ones(1, width, 1, 1))
                           for width in hidden_layer_width_list] for _ in range(ista_num_steps)]
        _lambda_Q_list = [[nn.Parameter(lasso_lambda_scalar * 1.2 * torch.ones(1, width, 1, 1))
                           for width in hidden_layer_width_list] for _ in range(ista_num_steps)]
        _lambda_M_list = [[nn.Parameter(lasso_lambda_scalar * 2.0 * torch.ones(1, width, 1, 1))
                           for width in hidden_layer_width_list] for _ in range(ista_num_steps)]

        # Use ParameterList, otherwise lambda parameters will not be optimized.
        self.lambda_S_list = nn.ParameterList([item for sublist in _lambda_S_list for item in sublist])
        self.lambda_P_list = nn.ParameterList([item for sublist in _lambda_P_list for item in sublist])
        self.lambda_Q_list = nn.ParameterList([item for sublist in _lambda_Q_list for item in sublist])
        self.lambda_M_list = nn.ParameterList([item for sublist in _lambda_M_list for item in sublist])

        # ==================== Proximal Operators (ResNet Blocks) ====================
        # 增强近端算子的复杂度
        self.proximal_S = nn.ModuleList([
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[0], kernel_size=3, n_resblocks=3),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[1], kernel_size=3, n_resblocks=5),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[2], kernel_size=3, n_resblocks=7)
        ])

        self.proximal_P = nn.ModuleList([
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[0], kernel_size=3, n_resblocks=2),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[1], kernel_size=3, n_resblocks=4),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[2], kernel_size=3, n_resblocks=6)
        ])

        self.proximal_Q = nn.ModuleList([
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[0], kernel_size=3, n_resblocks=2),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[1], kernel_size=3, n_resblocks=4),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[2], kernel_size=3, n_resblocks=6)
        ])

        self.proximal_M = nn.ModuleList([
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[0], kernel_size=3, n_resblocks=1),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[1], kernel_size=3, n_resblocks=3),
            ResidualGroup(conv=default_conv, n_feat=hidden_layer_width_list[2], kernel_size=3, n_resblocks=5)
        ])

        # ==================== Reconstruction Layers ====================
        # 改进重建网络 - 使用更深的网络
        self.recon_S_l = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.recon_Q_l = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.recon_S_n = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.recon_P_n = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        # self.recon_M = nn.Conv2d(n_classes, 1, 3, 1, 1)
        self.recon_M = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        # Multi-scale feature fusion (HFF)
        self.HFF_fuse_P = nn.Conv2d(n_classes * (ista_num_steps - 1), n_classes, 1, 1)
        self.HFF_fuse_Q = nn.Conv2d(n_classes * (ista_num_steps - 1), n_classes, 1, 1)
        self.HFF_fuse_M = nn.Conv2d(n_classes * (ista_num_steps - 1), n_classes, 1, 1)

        self.relu = nn.ReLU()
        self.cpmpress_z = nn.Conv2d(n_classes * 2, n_classes, 3, 1, 1)
        # 添加竖直梯度计算核
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

    def init_features(self, non_contrast_img, low_dose_img):
        """改进的特征初始化 - 增强特征分离"""
        # 1. 共享特征S初始化: 从n和l_tilde的concat中提取共享解剖结构
        # 使用差异引导的共享特征初始化
        # with torch.no_grad():
        #     diff = torch.abs(non_contrast_img - low_dose_img)
        #     similarity = 1.0 / (1.0 + diff)  # 相似性权重
        #
        # shared_input = torch.cat([non_contrast_img * similarity, low_dose_img * similarity], dim=1)

        # The paper defines S as the common anatomical component shared by NCCT and LD-CECT.
        # Therefore, we initialize S from a simple common anatomical proxy rather than using
        # an additional CrossSwinIR module that is not described in the manuscript.
        # shared_input = 0.5 * (non_contrast_img + low_dose_img)
        shared_input = low_dose_img
        shared_features = self.proj_shared(shared_input)
        S_adjoint_n = self.adjoint_encoder_dictionary_S_n(shared_features)
        S_adjoint_l = self.adjoint_encoder_dictionary_S_l(shared_features)

        # 2. P的独特特征初始化
        P_features = self.proj_contrast(non_contrast_img)
        P_adjoint = self.adjoint_encoder_dictionary_P_n(P_features)

        # 3. Q的独特特征初始化
        Q_features = self.proj_enhancement(low_dose_img)
        Q_adjoint = self.adjoint_encoder_dictionary_Q_l(Q_features)

        # 4. 噪声特征M初始化: 从残差中提取噪声特征
        vertical_gradient = F.conv2d(low_dose_img, self.sobel_y, padding=1)
        noise_features = self.proj_noise(vertical_gradient)

        M_adjoint = self.adjoint_encoder_dictionary_M(noise_features)

        S_list, P_list, Q_list, M_list = [], [], [], []

        for i in range(self.num_layers):
            # Initialize with proximal operator (soft thresholding)
            lambda_S = self.ista_stepsize_S[0] * self.lambda_S_list[i]
            S_i = relu(self.ista_stepsize_S[0].to(non_contrast_img.device) * (S_adjoint_n[i] + S_adjoint_l[i]) / 2,
                       lambd=lambda_S.to(non_contrast_img.device))
            S_list.append(S_i)

            lambda_P = self.ista_stepsize_P[0] * self.lambda_P_list[i]
            P_i = relu(self.ista_stepsize_P[0].to(non_contrast_img.device) * P_adjoint[i],
                       lambd=lambda_P.to(non_contrast_img.device))
            P_list.append(P_i)

            lambda_Q = self.ista_stepsize_Q[0] * self.lambda_Q_list[i]
            Q_i = relu(self.ista_stepsize_Q[0].to(low_dose_img.device) * Q_adjoint[i],
                       lambd=lambda_Q.to(low_dose_img.device))
            Q_list.append(Q_i)

            lambda_M = self.ista_stepsize_M[0] * self.lambda_M_list[i]
            M_i = relu(self.ista_stepsize_M[0].to(low_dose_img.device) * M_adjoint[i],
                       lambd=lambda_M.to(low_dose_img.device))
            M_list.append(M_i)

        return S_list, P_list, Q_list, M_list, shared_features, P_features, Q_features, noise_features

    def update_S(self, idx, non_contrast_img, low_dose_img, S_list, P_list, Q_list, M_list):
        """改进的共享特征更新 - 减少特异性信息干扰"""
        # New formula:
        # W_t = concat((D^S ⊗ S + D^P ⊗ P - n), (E^S ⊗ S + E^Q ⊗ Q - l_tilde))
        # ∇f(S) = G_S ⊗ W_t + (E^S)^~ ⊗ (E^M ⊗ M)

        # Reconstruction from non-contrast image
        n_recon = self.encoder_dictionary_S_n(S_list) + self.encoder_dictionary_P_n(P_list)
        n_error = n_recon - non_contrast_img

        # Reconstruction from low-dose image (without M for this part)
        l_tilde_recon_without_M = self.encoder_dictionary_S_l(S_list) + self.encoder_dictionary_Q_l(Q_list)
        l_tilde_error_without_M = l_tilde_recon_without_M - low_dose_img

        # Compute gradient using G_S
        # z = self.cpmpress_z(torch.cat([n_error, l_tilde_error_without_M], 1))
        # _,z = self.Fusion(n_error, l_tilde_error_without_M)
        # According to Eq. (S update), W_t is obtained by concatenating the NCCT and LD-CECT residuals.
        # Compress the concatenated residual features to n_classes channels through G_S.
        z = self.cpmpress_z(torch.cat([n_error, l_tilde_error_without_M], dim=1))
        # z = torch.cat([n_error, l_tilde_error_without_M], dim=1)
        # z = self.cross_modal_interaction(z)
        # err_z = self.encoder_dictionary_G_S(S_list) - self.recon_z(z)
        err_z = self.encoder_dictionary_G_S(S_list) - z
        adj_err_list_z = self.adjoint_encoder_dictionary_G_S(err_z)

        # Additional term: (E^S)^~ ⊗ (E^M ⊗ M)
        E_M_M = self.encoder_dictionary_M(M_list)
        grad_extra = self.adjoint_encoder_dictionary_S_l(E_M_M)

        # 跨模态交互增强
        # S_features = torch.cat([self.encoder_dictionary_S_n(S_list),
        #                         self.encoder_dictionary_S_l(S_list)], dim=1)
        # attended_S = self.cross_modal_interaction(S_features)

        stepsize = self.ista_stepsize_S[idx]

        # Update each layer
        for i in range(self.num_layers):
            # gradient = adj_err_list_z[i] + grad_extra[i] + attended_S
            gradient = adj_err_list_z[i] + grad_extra[i]
            S_list[i] = S_list[i] - stepsize.to(S_list[i].device) * gradient
            S_list[i] = self.proximal_S[i](self.relu(S_list[i]))

        return S_list

    def update_P(self, idx, non_contrast_img, S_list, P_list):
        """改进的对比特异性特征更新 - 增强特征独立性"""
        n_recon = self.encoder_dictionary_S_n(S_list) + self.encoder_dictionary_P_n(P_list)
        n_error = n_recon - non_contrast_img

        grad_P_n = self.adjoint_encoder_dictionary_P_n(n_error)
        stepsize = self.ista_stepsize_P[idx]

        for i in range(self.num_layers):
            gradient = grad_P_n[i]
            P_list[i] = P_list[i] - stepsize.to(P_list[i].device) * gradient
            P_list[i] = self.proximal_P[i](self.relu(P_list[i]))

        return P_list

    def update_Q(self, idx, low_dose_img, S_list, Q_list, M_list):
        """改进的增强特异性特征更新 - 专注于增强特征"""
        l_tilde_recon = (self.encoder_dictionary_S_l(S_list) +
                         self.encoder_dictionary_Q_l(Q_list) +
                         self.encoder_dictionary_M(M_list))
        l_tilde_error = l_tilde_recon - low_dose_img

        grad_Q = self.adjoint_encoder_dictionary_Q_l(l_tilde_error)
        stepsize = self.ista_stepsize_Q[idx]

        for i in range(self.num_layers):
            gradient = grad_Q[i]
            Q_list[i] = Q_list[i] - stepsize.to(Q_list[i].device) * gradient
            Q_list[i] = self.proximal_Q[i](self.relu(Q_list[i]))

        return Q_list

    def update_M(self, idx, low_dose_img, S_list, Q_list, M_list):
        """改进的噪声特征更新 - 增强噪声提取能力"""
        l_tilde_recon = (self.encoder_dictionary_S_l(S_list) +
                         self.encoder_dictionary_Q_l(Q_list) +
                         self.encoder_dictionary_M(M_list))
        l_tilde_error = l_tilde_recon - low_dose_img

        grad_M = self.adjoint_encoder_dictionary_M(l_tilde_error)
        stepsize = self.ista_stepsize_M[idx]

        for i in range(self.num_layers):
            M_list[i] = M_list[i] - stepsize.to(M_list[i].device) * grad_M[i]
            M_list[i] = self.proximal_M[i](self.relu(M_list[i]))

        return M_list

    def initialize_sparse_codes(self, x, rand_bool=False):
        """Initialize sparse codes for power iteration"""
        code_list = []
        num_samples = x.shape[0]
        input_spatial_dim_1 = x.shape[2]
        input_spatial_dim_2 = x.shape[3]

        initializer = torch.rand if rand_bool else torch.zeros

        for i in range(self.num_layers):
            feature_map_dim_1 = int(input_spatial_dim_1 / (2 ** i))
            feature_map_dim_2 = int(input_spatial_dim_2 / (2 ** i))
            code_tensor = initializer(num_samples, self.hidden_layer_width_list[self.num_layers - i - 1],
                                      feature_map_dim_1, feature_map_dim_2)
            code_list.append(code_tensor)

        code_list.reverse()
        return code_list

    def power_iteration_conv_model(self, conv_model, num_simulations: int):
        """Power iteration to estimate Lipschitz constant"""
        eigen_vec_list = self.initialize_sparse_codes(x=torch.zeros(1, 3, 64, 64), rand_bool=True)
        adjoint_conv_model = adjoint_dictionary_model(conv_model)

        for _ in range(num_simulations):
            eigen_vec_list = adjoint_conv_model(conv_model(eigen_vec_list))
            flatten_x_norm = torch.norm(torch.cat([x.flatten() for x in eigen_vec_list]))
            eigen_vec_list = [x / flatten_x_norm for x in eigen_vec_list]

        eigen_vecs_flatten = torch.cat([x.flatten() for x in eigen_vec_list])
        linear_trans_eigen_vecs_list = adjoint_conv_model(conv_model(eigen_vec_list))
        linear_trans_eigen_vecs_list_flatten = torch.cat([x.flatten() for x in linear_trans_eigen_vecs_list])

        numerator = torch.dot(eigen_vecs_flatten, linear_trans_eigen_vecs_list_flatten)
        denominator = torch.dot(eigen_vecs_flatten, eigen_vecs_flatten)
        eigenvalue = numerator / denominator

        return eigenvalue

    def forward(self, non_contrast_img, low_dose_img):
        """
        改进的前向传播 - 增强特征分离
        """
        # Initialize features
        S_list, P_list, Q_list, M_list, shared_features, P_features, Q_features, noise_features = self.init_features(
            non_contrast_img, low_dose_img)

        # Store intermediate features for multi-scale fusion
        HFF_P = []  ### HFF_F_U
        HFF_Q = []
        HFF_M = []

        # ISTA iterations - 改进的交替优化顺序
        for iteration in range(1, self.ista_num_steps):
            # 先更新特异性特征，再更新共享特征
            P_list = self.update_P(iteration, non_contrast_img, S_list, P_list)
            Q_list = self.update_Q(iteration, low_dose_img, S_list, Q_list, M_list)

            # cal F_u, F_c , ...
            # P_Du_cur = self.encoder_dictionary_P_n(P_list)  # UNDONE: decoder or encoder!!!
            # P_Dc_cur = non_contrast_img - P_Du_cur
            P_Qu_cur = self.decoder_dictionary_P_n(P_list)
            HFF_P.append(P_Qu_cur)
            # Q_Du_cur = self.encoder_dictionary_Q_l(Q_list)
            # Q_Dc_cur = low_dose_img - Q_Du_cur
            Q_Qu_cur = self.decoder_dictionary_Q_l(Q_list)
            HFF_Q.append(Q_Qu_cur)

            S_list = self.update_S(iteration, non_contrast_img, low_dose_img, S_list, P_list, Q_list, M_list)

            M_list = self.update_M(iteration, low_dose_img, S_list, Q_list, M_list)

            M_recon_cur = self.decoder_dictionary_M(M_list)
            HFF_M.append(M_recon_cur)

        P_fused = self.HFF_fuse_P(torch.cat(HFF_P, 1))
        Q_fused = self.HFF_fuse_Q(torch.cat(HFF_Q, 1))
        M = self.HFF_fuse_M(torch.cat(HFF_M, 1))

        P_Qc = self.decoder_dictionary_S_n(S_list)
        Q_Qc = self.decoder_dictionary_S_l(S_list)

        # Shared anatomical reconstructions are kept directly, consistent with the unfolded formulation.

        P_img_Qu = self.recon_P_n(P_fused)
        Q_img_Qu = self.recon_Q_l(Q_fused)

        P_img_Qc = self.recon_S_n(P_Qc)
        Q_img_Qc = self.recon_S_l(Q_Qc)

        # Final reconstruction - 使用学习到的权重
        # l_hat = F^S ⊗ S + F^Q ⊗ Q (全剂量对比增强CT)
        l_hat = self.alpha_S * Q_img_Qc + self.alpha_Q * Q_img_Qu

        # n_hat = G^S ⊗ S + G^P ⊗ P (增强的非对比CT)
        n_hat = self.alpha_S * P_img_Qc + self.alpha_P * P_img_Qu

        low = self.recon_M(M) + l_hat
        l_hat_1 = low_dose_img - self.recon_M(M)
        l_hat_2 = (l_hat + l_hat_1) / 2

        return n_hat, l_hat, low,l_hat_2

class adjoint_conv_op(nn.Module):
    # The adjoint of a conv module.
    def __init__(self, conv_op):
        super().__init__()
        in_channels = conv_op.out_channels
        out_channels = conv_op.in_channels
        kernel_size = conv_op.kernel_size
        padding = kernel_size[0] // 2

        # transpose convolution
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                                 bias=False)

        # tie the weights of transpose convolution with convolution
        self.transpose_conv.weight = conv_op.weight

    def forward(self, x):
        return self.transpose_conv(x)


class up_block(nn.Module):
    """
    A module that contains:
    (1) an up-sampling operation (implemented by bilinear interpolation or upsampling)
    (2) convolution operations
    """

    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # the up-sampling operation
        self.up = nn.ConvTranspose2d(in_channels, in_channels - 32, kernel_size=2, stride=2, bias=False)

        # the 2d convolution operation
        self.conv = nn.Conv2d((in_channels - 32) * 2, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              bias=False)

    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)


class adjoint_up_block(nn.Module):
    # adjoint of up_block module

    def __init__(self, up_block_model):
        super().__init__()

        # to construct the adjoint model, one should exclude additive biases and use transposed conv for upsampling.

        in_channels = up_block_model.out_channels
        self.adjoint_conv_op = adjoint_conv_op(up_block_model.conv)
        self.adjoint_up = nn.Conv2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False)
        self.adjoint_up.weight = up_block_model.up.weight

    def forward(self, x):
        x = self.adjoint_conv_op(x)
        # input is CHW
        x2 = x[:, :int(x.shape[1] / 2), :, :]
        x1 = x[:, int(x.shape[1] / 2):, :, :]
        x1 = self.adjoint_up(x1)
        return (x1, x2)


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class adjoint_out_conv(nn.Module):
    def __init__(self, out_conv_model):
        super().__init__()
        in_channels = out_conv_model.out_channels
        out_channels = out_conv_model.in_channels

        self.adjoint_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.adjoint_conv.weight = out_conv_model.conv.weight

    def forward(self, x):
        return self.adjoint_conv(x)


class dictionary_model(nn.Module):
    def __init__(self, kernel_size, hidden_layer_width_list, n_classes):
        super(dictionary_model, self).__init__()

        self.hidden_layer_width_list = hidden_layer_width_list

        in_out_list = [[hidden_layer_width_list[i], hidden_layer_width_list[i + 1]] for i in
                       range(len(hidden_layer_width_list) - 1)]

        self.num_hidden_layers = len(in_out_list)

        self.n_classes = n_classes

        # the initial convolution on the bottleneck layer
        self.bottleneck_conv = nn.Conv2d(hidden_layer_width_list[0], hidden_layer_width_list[0],
                                         kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        self.syn_up_list = []

        for layer_idx in range(self.num_hidden_layers):
            new_up_block = up_block(kernel_size, *in_out_list[layer_idx])
            self.syn_up_list.append(new_up_block)

        self.syn_up_list = nn.Sequential(*self.syn_up_list)

        self.syn_outc = out_conv(hidden_layer_width_list[-1], n_classes)

    def forward(self, x_list):

        # x_list is ordered from wide-channel to thin-channel.
        num_res_levels = len(x_list)

        #         x_prev = x_list[0]
        x_prev = self.bottleneck_conv(x_list[0])

        for i in range(1, num_res_levels):
            x = x_list[i]
            syn_up = self.syn_up_list[i - 1]
            x_prev = syn_up(x_prev, x)

        syn_output = self.syn_outc(x_prev)
        return syn_output


class adjoint_dictionary_model(nn.Module):
    def __init__(self, dictionary_model):
        super().__init__()

        self.adjoint_syn_outc = adjoint_out_conv(dictionary_model.syn_outc)
        self.adjoint_syn_bottleneck_conv = adjoint_conv_op(dictionary_model.bottleneck_conv)

        self.adjoint_syn_up_list = []

        self.num_hidden_layers = dictionary_model.num_hidden_layers

        for layer_idx in range(dictionary_model.num_hidden_layers):
            self.adjoint_syn_up_list.append(adjoint_up_block(dictionary_model.syn_up_list[layer_idx]))

    def forward(self, y):
        y = self.adjoint_syn_outc(y)
        x_list = []

        for layer_idx in range(self.num_hidden_layers - 1, -1, -1):
            adjoint_syn_up = self.adjoint_syn_up_list[layer_idx]  # 下采样
            y, x = adjoint_syn_up(y)
            x_list.append(x)
        y = self.adjoint_syn_bottleneck_conv(y)
        x_list.append(y)
        x_list.reverse()
        return x_list

class LOTNCIRecon(nn.Module):
    """
    Multi-Contrast Collaborative Dictionary Learning for CT Denoising
    Main class that wraps the ISTA-UNet denoiser
    """

    def __init__(self, hidden_layer_width_list=[128, 96, 64], n_classes=64, ista_num_steps=5):
        super().__init__()

        self.in_channel = 1
        self.channel_fea = n_classes

        self.denoiser = UnfoldedReconstructionNet(
            kernel_size=3,
            hidden_layer_width_list=hidden_layer_width_list,
            n_classes=n_classes,
            ista_num_steps=ista_num_steps
        )

    def forward(self, non_contrast_img, low_dose_img):
        n_hat, l_hat, low,l_hat_2 = self.denoiser(non_contrast_img, low_dose_img)
        return n_hat, l_hat, low,l_hat_2



