# Mean-anchored self-supervised reconstruction from few speckle measurements in light-field microscopy

## 2. Principle and method

The proposed method reconstructs a nonnegative 3D fluorescence distribution from a short sequence of speckle-illuminated light-field measurements. It uses a mean-derived volume as the reconstruction anchor and combines this estimate with variance-derived structure and features from individual frames. Training constrains the reconstructed volume through the light-field forward model using complementary measurements. At inference, all inputs and intensity calibration are obtained from the same ten-frame sequence.

### 2.1. Image formation and speckle statistics

Let \(f\geq0\) denote the discretized fluorophore distribution, \(s_n\) the illumination intensity in exposure n, and \(y_n\) the corresponding detector image. Under linear fluorescence response,

\[
y_n=\mathcal H(f\odot s_n)+\eta_n,\qquad n=1,\ldots,N_{\mathrm{acq}},
\tag{1}
\]

where \(\mathcal H\) maps object voxels to detector pixels, \(\odot\) denotes elementwise multiplication, and \(\eta_n\) represents measurement noise. The specimen is assumed to remain stationary during one sequence. The forward operator retains the dependence of the light-field point spread function (PSF) on depth and lateral position within a microlens period, following the wave-optics description of LFM [2]. Each phase-specific kernel propagates the corresponding object samples, and the contributions are summed at the detector.

For a frame set \(\mathcal I\) with \(N=|\mathcal I|\), the measured mean and sample variance are

\[
\mu_{\mathcal I}=\frac{1}{N}\sum_{n\in\mathcal I}y_n,
\qquad V_{\mathcal I}=\frac{1}{N-1}\sum_{n\in\mathcal I}(y_n-\mu_{\mathcal I})^2.
\tag{2}
\]

The mean records the average fluorescence response, while the variance depends on illumination-induced fluctuations. To make the statistical approximation explicit, let \(h_{pi}\) be the response at detector pixel p to object voxel i and \(C^s_{ij}\) the illumination covariance between voxels i and j. For additive noise independent of the illumination,

\[
\operatorname{Var}(y_p)=\sum_{i,j}h_{pi}h_{pj}f_i f_j C^s_{ij}+\operatorname{Var}(\eta_p).
\tag{3}
\]

Illumination covariance therefore contributes to the spatial information encoded in the measurements [5]. With spatially uniform mean illumination and a diagonal approximation to its covariance, the reconstruction uses

\[
\widehat\mu=\mathcal H(f),\qquad \widehat V\simeq\mathcal H_2(f^2),
\tag{4}
\]

where \(\mathcal H_2\) is formed by squaring each PSF kernel elementwise. This is the squared-PSF approximation used in variance-based speckle LFM [7]. Spatially constant illumination factors are absorbed into the intensity convention or removed by variance normalization. The approximation omits correlations between different object voxels; its accuracy can therefore depend on speckle size and specimen structure. The present training run uses simulated intensities before detector-noise generation and sets additional shot-noise and read-noise terms to zero.

### 2.2. Statistical initialization and feature fusion

Two preliminary volumes provide the network with mean- and variance-derived descriptions of the same input sequence. Both are computed using three multiplicative backprojection updates. For an input image y, forward operator \(\mathcal A\), and corresponding implemented backprojector \(\mathcal B\), the update is

\[
x^{(0)}=\mathcal B(y),\qquad
x^{(k+1)}=x^{(k)}\odot\frac{\mathcal B(y)}{\mathcal B(\mathcal A(x^{(k)}))},\quad k=0,1,2.
\tag{5}
\]

Division is elementwise, with undefined zero-over-zero updates set to zero. Applying Eq. (5) to \(\mu_{\mathcal I}\) with the PSF and its backprojection kernels gives the mean estimate \(M_{\mathcal I}\). Applying it to \(V_{\mathcal I}\) with elementwise-squared forward and backprojection kernels gives an estimate of squared object intensity; its nonnegative square root defines \(T_{\mathcal I}\). Both initializations use only the N input frames and retain their floating-point intensity scale.

Separate 3D encoders process \(M_{\mathcal I}\) and \(T_{\mathcal I}\). Each volume is divided by its root-mean-square value, floored at \(10^{-6}\), before feature extraction. The physical mean anchor is retained separately at its original scale. Each encoder has three lateral scales with 16, 32, and 64 channels; downsampling reduces the lateral dimensions by factors of two while preserving the axial sampling. Convolutional blocks contain two 3 × 3 × 3 convolutions, each followed by group normalization and a sigmoid linear unit (SiLU).

The third branch extracts features from the centered frames \(d_n=y_n-\mu_{\mathcal I}\). A shared 2D encoder, with 8, 16, and 32 channels and corresponding 3 × 3 convolutional blocks, processes each frame. At each scale, the feature mean and standard deviation are pooled over frames, concatenated, and projected with a 1 × 1 convolution. The standard deviation uses the population second moment with an added \(10^{-8}\) before the square root. The pooled features are broadcast over depth, concatenated with the coordinate \(z/(100\,\mu\mathrm m)\), and passed through a 3D convolutional block. This aggregation makes the reconstruction invariant to frame order while retaining information extracted before pooling.

At scale l, the mean and variance features are concatenated and projected to a joint tensor \(P_l\). The lifted frame-set features \(S_l\) enter through a learned gate:

\[
F_l=P_l+\alpha_l G_l\odot C_l(S_l),\qquad
G_l=\sigma\!\left(D_l([P_l,S_l])\right).
\tag{6}
\]

Here brackets denote channel concatenation, \(C_l\) and \(D_l\) are 1 × 1 × 1 convolutions, and \(\sigma\) is the sigmoid function. Each gate assigns one value per voxel shared across channels. The scale coefficient \(\alpha_l\) is sigmoid-parameterized and initialized to 0.05. A 3D decoder combines the fused features through trilinear upsampling and lateral-scale skip connections, producing a signed residual volume \(R_\theta\).

### 2.3. Mean anchoring and intensity calibration

The mean estimate defines the volume that the learned residual modifies. Its initial scale is obtained by least-squares matching to the input mean:

\[
\beta_0=\max\!\left(10^{-8},\frac{\langle\mathcal H(M_{\mathcal I}),\mu_{\mathcal I}\rangle}{\|\mathcal H(M_{\mathcal I})\|_{2}^{2}+10^{-8}}\right),
\qquad \beta=\beta_0[1+0.2\tanh(\xi)].
\tag{7}
\]

The learned scalar \(\xi\) is shared across examples and initialized to zero. With \(A=\beta M_{\mathcal I}\), the intermediate reconstruction is

\[
f_0=
\begin{cases}
A+R_\theta,&R_\theta\geq0,\\
A\exp\!\left[R_\theta/\max(A,\epsilon_p)\right],&R_\theta<0,
\end{cases}
\qquad \epsilon_p=10^{-8}.
\tag{8}
\]

Equation (8) permits additive enhancement and multiplicative attenuation while maintaining nonnegativity. The decoder output layer is initialized to zero, making the initial output equal to A. The anchor thus defines the reconstruction starting point without imposing a small-residual constraint.

We separate the reconstructed structure from its overall intensity:

\[
q=\frac{f_0}{\max(\sum_i f_{0,i},10^{-30})},\qquad \hat f=a q.
\tag{9}
\]

For a nonzero volume, q has unit sum. Its intensity scale is calibrated against the same input mean,

\[
a_0=\max\!\left(0,\frac{\langle\mathcal H(q),\mu_{\mathcal I}\rangle}{\max(\|\mathcal H(q)\|_{2}^{2},10^{-30})}\right),
\qquad a=a_0[1+0.2\tanh(\gamma)],
\tag{10}
\]

where \(\gamma\) is another shared scalar initialized to zero. During training, q is detached when calculating \(a_0\) and the mean prediction used for intensity fitting. This makes it possible to control intensity and structural mean consistency through separate loss terms. The two scales have distinct roles: \(\beta\) scales the anchor before residual correction, whereas a sets the intensity of the final normalized structure.

### 2.4. Self-supervised constraints and gradient control

For training, each 100-frame sequence is partitioned into a ten-frame input set \(\mathcal I\) and a complementary 90-frame constraint set \(\mathcal C\), with \(\mathcal I\cap\mathcal C=\varnothing\). Only \(\mathcal I\) is used to form network inputs and estimate intensity scales. The statistics \(\mu_{\mathcal C}\) and \(V_{\mathcal C}\) provide targets in detector space. Training therefore requires additional measurements of each training object but does not use a 3D ground-truth volume as a target.

Let \(\mathcal N(u)=u/\max(\operatorname{mean}_{xy}u,10^{-30})\), where the mean is over detector pixels. Let \(\rho(u,v)\) denote the pixel-averaged SmoothL1 loss, with a quadratic region for \(|u-v|<1\) and a linear region otherwise. The variance constraint compares normalized log-variance images:

\[
L_V=\rho\!\left(\log[\mathcal N(\mathcal H_2(q^2))+\epsilon_v],
\log[\mathcal N(V_{\mathcal C})+\epsilon_v]\right),\qquad \epsilon_v=10^{-6}.
\tag{11}
\]

Normalization removes a global variance scale, allowing this term to act on spatial structure. Mean consistency is separated into intensity and structural components,

\[
L_{M,a}=\rho\!\left(\frac{a\mathcal H(\operatorname{sg}(q))}{s_\mu},\frac{\mu_{\mathcal C}}{s_\mu}\right),
\qquad L_{M,q}=\rho\!\left(\mathcal N(\mathcal H(q)),\mathcal N(\mu_{\mathcal C})\right),
\tag{12}
\]

where \(s_\mu=\max(\operatorname{mean}_{xy}|\mu_{\mathcal C}|,10^{-8})\) and \(\operatorname{sg}\) denotes stop-gradient. The intensity term updates \(\gamma\) without directly updating q. The structural term compares the spatial distribution of the predicted and measured means independently of their overall intensity.

The structural mean term receives an adaptive coefficient that limits its gradient relative to the variance term for each training example:

\[
c_t=\operatorname{sg}\!\left[\min\!\left(1,
 b_t\frac{\|\nabla_q L_V\|_2}{\|\nabla_q L_{M,q}\|_2+10^{-12}}\right)\right],
\qquad b_t=\min(t/50,1).
\tag{13}
\]

The optimizer-update index t starts at one. The budget increases linearly during the first 50 updates and then remains at one, so the weighted mean-structure gradient does not exceed the variance gradient in norm, up to numerical tolerance. The coefficient is detached to avoid differentiating through the gradient-norm calculation. This constraint acts in normalized-volume space and controls the relative gradient magnitude without assuming that the two gradients point in the same direction.

The training objective is

\[
L=L_{M,a}+L_V+c_t L_{M,q}+10^{-5}\operatorname{TV}_{0.5}(q\operatorname{sg}(a)),
\tag{14}
\]

with the anisotropic total-variation penalty

\[
\operatorname{TV}_{0.5}(u)=\operatorname{mean}|D_xu|+\operatorname{mean}|D_yu|+0.5\operatorname{mean}|D_zu|.
\tag{15}
\]

Here \(D_x\), \(D_y\), and \(D_z\) are adjacent-voxel differences without division by voxel pitch. The scalar a is detached in this term, so regularization updates structure at the calibrated intensity scale. Validation uses the object-averaged score \(L_{M,a}+L_V+10^{-5}\operatorname{TV}_{0.5}\), excluding the adaptive structural mean term. Ground-truth metrics do not enter this score.

### 2.5. Training implementation and inference

The present model was trained on eleven simulated objects, with three objects assigned to validation and three to evaluation. Each object supplied 100 measurements and ten fixed input/constraint partitions, yielding 110 training and 30 validation instances. Object assignments were disjoint; partitions of the same acquisition could overlap across instances. The modeled volumes contain 260 × 260 lateral samples and ten axial planes at 10–100 µm in 10 µm increments. The selected PSF uses 49 × 49 phase positions within a microlens period. The simulation configuration specifies detection and illumination numerical apertures of 0.15 and 0.05, respectively.

Training used Adam for 400 optimizer updates with a global batch size of eight. Initial learning rates were \(10^{-3}\) for the network and \(10^{-4}\) for the scalars \(\xi\) and \(\gamma\); after 200 updates, they were reduced to \(10^{-4}\) and \(10^{-5}\), respectively. The initialization seed was 20260901. Validation was performed every 20 updates, and the minimum validation score occurred at update 400 in the specified run. The model has 1,188,377 trainable parameters. Computation used FP32 with automatic mixed precision and TensorFloat-32 disabled.

At inference, the ten input frames are used to calculate \(\mu_{\mathcal I}\), \(V_{\mathcal I}\), the two preliminary volumes, and the centered-frame features. The trained network then predicts the residual, applies the mean-anchored nonnegative mapping, and calibrates the output intensity through Eqs. (9) and (10). No constraint frames or reference volume are used, and the network weights remain fixed.

## References cited in Section 2

[2] M. Broxton, L. Grosenick, S. Yang, et al., “Wave optics theory and 3-D deconvolution for the light field microscope,” **Opt. Express 21**(21), 25418–25439 (2013). [DOI](https://doi.org/10.1364/OE.21.025418).

[5] J. Idier, S. Labouesse, M. Allain, et al., “On the superresolution capacity of imagers using unknown speckle illuminations,” **IEEE Trans. Comput. Imaging 4**(1), 87–98 (2018). [DOI](https://doi.org/10.1109/TCI.2017.2771729).

[7] M. A. Taylor, T. Nöbauer, A. Pernia-Andrade, et al., “Brain-wide 3D light-field imaging of neuronal activity with speckle-enhanced resolution,” **Optica 5**(4), 345–353 (2018). [DOI](https://doi.org/10.1364/OPTICA.5.000345).

## 作者待补说明（不属于投稿正文）

本节按指定的 mean_anchor 400 步运行撰写，引用编号延续已交付的 Introduction。方法公式与当前已核对的输入构建、网络和训练实现对应；真实实验完成后，应按实际使用的模型及配置更新。

> **[待补：物理适用性实验]**在后续实验部分补充有限散斑相关长度、探测噪声和 PSF 失配的影响。式（4）的对角协方差近似和当前无探测器噪声训练不能代替这些验证，也不能单独推导实际分辨率增益。

> **[待补：真实系统与组织实验]**在实验设置部分补齐 PSF 获取及标定流程、照明与探测波段、实际 NA、曝光和照明切换时间，以及菠菜根与鼠脑切片的制备、独立样本数和参考成像。当前仿真 NA 和轴向采样不能写成实机参数或实测分辨率。

> **[待补：复现与独立评价]**补充实验时的软件版本、GPU 型号、公开代码/权重的准确位置；保留初始重建的 FFT 边界及非负数值处理说明。增加多随机种子和锁定方案后的独立测试，并完成少帧与高帧基线在相同细节标准下的比较。已有开发比较不能重新表述为最终盲测。
