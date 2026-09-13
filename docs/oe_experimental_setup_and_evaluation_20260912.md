# Mean-anchored self-supervised reconstruction from few speckle measurements in light-field microscopy

## 3. Experimental setup and evaluation

The evaluation addresses whether reconstruction from ten speckle measurements preserves spatial detail while controlling volumetric error and false structures. Simulations with known object geometry provide the current quantitative evidence. The experimental extension will assess fluorescent spinach roots and mouse brain sections using independently acquired structural references. Network training and inference follow Section 2; the acquisition conditions, comparators, and evaluation criteria are specified below.

### 3.1. Numerical data and simulation conditions

The dataset contains eleven training objects, three validation objects, and three evaluation objects. The training set combines five planar phantoms at different depths with six volumetric objects. Each object provides 100 simulated measurements and ten fixed partitions into ten input frames and 90 complementary constraint frames. All measurements belonging to an object remain in the same data partition. These partitions yield 110 training, 30 validation, and 30 evaluation instances, with the ten instances of each object representing repeated measurement subsets.

The object grid has 260 × 260 lateral samples at approximately 1.12245 µm spacing. Reconstructions retain ten native depth planes from 10 to 100 µm at 10 µm intervals. The light-field PSF has a detection NA of 0.15 and 49 × 49 lateral phase positions within a microlens period. Depth planes are selected from the PSF's stored physical coordinates. The three evaluation objects test complementary aspects of spatial fidelity (Table 1).

**Table 1. Synthetic objects used for the current evaluation. Each object is evaluated with ten fixed input subsets.**

| Object | Geometry and evaluation purpose |
|---|---|
| T02 | Curved tubular structures with known interruptions; centerline recovery, false breaks, and false connections. |
| T03 | Horizontal and vertical three-bar groups at 50 µm depth; feature detection and lateral separation across six widths. |
| T04 | Axially overlapping thin-line pairs and single-layer controls; depth localization, axial separation, and false peaks. |

For the volumetric objects, a random phase field is filtered by a circular illumination pupil and propagated through depth using the angular-spectrum method. The illumination NA is 0.05 and the wavelength is 488 nm. The generator uses a 520 × 520 padded grid at 1.125 µm spacing and retains the central 260 × 260 region. Each exposure uses one complex pupil field shared across all depth planes, preserving axial correlations in the resulting speckle intensities. Fluorescence is calculated by multiplying the object and illumination volumes and applying the depth-dependent light-field forward model. The planar training phantoms retain their original single-plane generation pipeline, including the quantized source product used for P01. Network inputs and statistical targets are read from the floating-point detector-intensity stream before detector-noise generation. No additional shot noise or read noise is included in the present run.

> **[待补：仿真适用范围]**补充噪声、散斑相关长度和 PSF 失配实验，并给出实际扫描范围。当前照明与物体网格的间距分别为 1.125 µm 和 1.12245 µm，正式复现材料中应保留这一区别。当前数据不支持将结果表述为已验证的低光子鲁棒性。

### 3.2. Optical setup and biological specimens

The experimental system will combine speckle fluorescence excitation with light-field detection. The excitation path will be documented by **[TO SUPPLY: source and wavelength, speckle-generation device, illumination NA, and sample-plane power]**. The detection path will be specified by **[TO SUPPLY: objective model, magnification and NA, emission filters, microlens pitch and focal length, and camera model and pixel pitch]**. Calibration will determine the effective sample-plane pixel size, depth coordinates, and PSF over the evaluated field and depth range. **[TO SUPPLY: PSF measurement or calculation procedure and calibration data.]** The numerical settings in Section 3.1 will not be substituted for experimental calibration.

For each field of view, the acquisition record will include the number of illumination realizations, exposure duration, illumination-switching and camera-readout times, and any discarded frames. Detector offsets, flat-field correction, and geometric rectification will be documented. Statistical measurements will be calculated from consistently calibrated linear-intensity images. Any exposure or illumination-power normalization will use the recorded acquisition settings, and frames selected for ten-frame inference will be identified explicitly.

Spinach-root specimens will be described by **[TO SUPPLY: plant source, root region, preparation or fixation, fluorescent label or autofluorescence channel, and mounting medium]**. Mouse brain sections will be described by **[TO SUPPLY: tissue source, animal characteristics where relevant, brain region, section thickness, fixation, labeling, and mounting]**. Independent specimen counts and fields of view per specimen remain **[TO SUPPLY]**. Mouse-tissue provenance and the applicable ethics committee and approval details will be reported from the actual records.

An independent 3D reference will be acquired using **[TO SUPPLY: reference modality, calibrated resolution, and registration procedure]**. Regions of interest will be defined from the reference or specimen anatomy before method-specific comparisons. For spinach roots, evaluation will focus on visible tissue boundaries and structural continuity; for mouse brain sections, the targets will be determined by the actual label and resolvable anatomy. A high-frame-count reconstruction will provide an additional computational comparator. Where no independent reference is available, forward statistical agreement will be reported as a consistency measure, with depth accuracy and resolution left unverified.

> **[待补：正式组织实验]**本小节保留将来时和占位信息。已有菠菜根探索性迁移记录可说明数据可处理，但没有独立三维真值，不能替代这里的正式重复实验与参考成像；鼠脑切片实验仍需补齐。

### 3.3. Comparison methods and measurement budgets

The current ten-frame comparison includes the mean-derived initialization, the square root of the variance-derived initialization, a Taylor-anchored network, and the proposed mean-anchored network. The initializations use the three-step multiplicative update in Eq. (5) and are denoted Mean-RL3 and Taylor-RL3-sqrt to match the stored evaluation labels. The two networks use the same feature branches, data partitions, initialization seed, and 400-update training schedule. The Taylor-anchored model uses the variance-derived volume as its output anchor, with its analytic anchor scale matched accordingly. Both networks retain the same mean-gradient budget and intensity-calibration mechanism. Comparison at update 400 therefore assesses the choice of reconstruction anchor under a matched training schedule.

The acquisition-budget experiment will compare the proposed ten-frame reconstruction with mean- and variance-based iterative reconstructions using 5, 10, 20, 50, and 100 measurements. Nested subsets from fixed random frame orderings will support paired comparisons across measurement counts. Each method's input statistics will be recomputed from its own selected frames. Conventional iteration counts will be selected using development data over **[TO SUPPLY: iteration range and stopping criterion]**; the three-step network initializations will remain identifiable as initialization controls. The proposed network will use only its designated ten frames during inference.

A common fidelity target will be fixed before the final comparison. It will combine a specified spatial-separation criterion with limits on volumetric error and false structures. A frame-reduction factor will be reported only when both compared configurations meet that target:

\[
R_N=\frac{N_{\mathrm{baseline}}}{N_{\mathrm{proposed}}},\qquad N_{\mathrm{proposed}}=10.
\tag{16}
\]

The target thresholds remain **[TO SUPPLY: feature scale, tolerated error, and false-feature limits]**. Fixed exposure per frame and fixed total exposure will be evaluated separately. Acquisition time and reconstruction time will also be measured separately, with reconstruction timing including statistical preprocessing, both initial reconstructions, network execution, and final calibration. Timing will use matched hardware and report warm-up, synchronization, and the number of repeated measurements.

The planned ablations will test the mean and variance feature branches, the individual-frame branch, intensity/structure separation, and the adaptive mean-structure constraint. Each trainable variant will be retrained with matched data and optimization budgets. A fixed-coefficient mean loss will provide a comparison for adaptive gradient control. Perturbation experiments will vary illumination correlation, detector noise, and PSF mismatch, with the settings chosen on development data and reported explicitly.

> **[待补：核心减帧与消融实验]**完成十帧网络对高帧迭代重建的质量匹配比较，并补充合理迭代数基线、消融及计时；另评估可与本测量模型匹配的独立物理自监督基线，说明适配和训练条件。若扩展到其他输入帧数的网络，必须说明是否重训；100 帧输入的自监督训练需另有不重叠约束帧。当前不填写减帧倍数或提速结果。

### 3.4. Spatial separation and structural specificity

The T03 chart contains three equal-width bars with equal-width gaps in each group. The widths are approximately 2.245, 3.367, 4.490, 6.735, 8.980, and 11.224 µm, evaluated separately in the two orientations. Each group uses a geometry-defined region. The native reconstruction plane with the largest regional intensity sum is selected and must lie within 10 µm of the target depth. A transverse profile is averaged over the central 60% of the bar length. Detected peaks are assigned one-to-one to the three expected centers within a tolerance of two lateral pixels and must exceed 10% of the reconstruction's full-volume maximum. For neighboring matched peaks, separation requires

\[
r_{\mathrm{valley}}=\frac{I_{\mathrm{valley}}}{\min(I_{\mathrm{peak},1},I_{\mathrm{peak},2})}\leq0.8.
\tag{17}
\]

Here the valley is the minimum profile value strictly between the two peaks. A group passes only if all three peaks are localized and both intervening valleys satisfy Eq. (17). Thresholds of 5% and 20% are retained as sensitivity analyses. Unmatched peaks are reported separately. This criterion measures feature detection, depth localization, and separation jointly; the bar width is reported together with its equal-width gap, rather than converted to a PSF width.

For T04, intensities are summed within fixed lateral regions to form axial profiles on the ten native depth samples. Peaks are matched to the expected layers within 10 µm. The same valley criterion is applied to eligible pairs with an intervening depth sample. Pairs separated by 10 µm are assessed for localization and expected-layer intensity, with valley-based separation marked not applicable. Single-layer controls assess spurious axial peaks. Interpolation is not used to create intermediate samples for these decisions.

For T02, the known tube centerlines are sampled every 0.5 µm along their lateral paths. A centerline location is detected if signal above the selected threshold occurs within two lateral pixels and one axial sample. The localized fraction and the fraction of tube segments containing a missed location are reported. A false connection across a known interruption is counted when all nine sampled locations spanning the central 60% of the gap exceed the threshold at its target depth. These operational definitions distinguish recovered continuity from filled-in gaps.

Experimental spatial resolution will be assessed with **[TO SUPPLY: bead diameter or line-target geometry, sampled depths and field positions, and replicate counts]**. Lateral and axial profiles will be reported separately with the fitting or separation rule stated in advance. Finite bead size, background treatment, and reference resolution will be documented when interpreting full width at half maximum (FWHM). The 10 µm reconstruction depth interval is a sampling parameter and will not be treated as a measured axial resolution.

### 3.5. Volumetric fidelity, repeatability, and reporting

For simulations, volumetric shape fidelity is quantified by intensity-scale-aligned normalized root-mean-square error:

\[
\operatorname{NRMSE}_{\mathrm{shape}}=\frac{\|\alpha^*\hat f-f_{\mathrm{GT}}\|_{2}}{\max(\|f_{\mathrm{GT}}\|_{2},10^{-12})},
\qquad \alpha^*=\max\!\left(0,\frac{\langle\hat f,f_{\mathrm{GT}}\rangle}{\max(\|\hat f\|_{2}^{2},10^{-12})}\right).
\tag{18}
\]

The fitted scalar is used only for evaluation and does not spatially register the reconstruction. Raw intensity error is retained separately. Axial fidelity is measured by the Wasserstein-1 distance between unit-mass axial profiles. For normalized profiles p and g and their cumulative sums \(F_p\) and \(F_g\),

\[
W_z=\Delta z\sum_k|F_p(k)-F_g(k)|,\qquad \Delta z=10\,\mu\mathrm m.
\tag{19}
\]

A complementary background measure is the fraction of reconstructed intensity outside a lateral support mask. The mask is obtained by thresholding the ground-truth maximum-intensity projection (MIP) at 10% of its maximum and applying two iterations of binary dilation with four-connected neighbors. The same lateral mask is applied through all depth planes. Structural similarity (SSIM) [12] is also calculated between the lateral MIPs. The current implementation uses an 11 × 11 uniform window with zero padding, \(K_1=0.01\) and \(K_2=0.03\), and the combined intensity range of the two images; no gain alignment is applied for this score. MIP-based measures supplement the volumetric and native-plane evaluations.

Subset repeatability is evaluated after normalizing each nonnegative reconstruction to unit mass. For K = 10 subsets, with normalized volumes \(q_j\) and their mean \(\bar q\), the relative dispersion is

\[
D_{\mathrm{subset}}=\frac{\sqrt{K^{-1}\sum_{j=1}^{K}\|q_j-\bar q\|_{2}^{2}}}{\max(\|\bar q\|_{2},10^{-30})}.
\tag{20}
\]

Pairwise axial Wasserstein distances are averaged over the 45 subset pairs for each object. Metrics are first summarized within each object and then averaged with equal weight across the three evaluation objects. Individual object values and subset variation accompany the aggregate values. The current results describe one training seed and three evaluation geometries; the 30 object–subset instances and 45 within-object pairs are not independent object replicates. These geometries have been used in development comparisons, so a further evaluation on newly acquired or generated objects will follow model and threshold locking.

For biological validation, independent roots and independent source animals or donor tissues will define the biological replicate, with sections and fields of view nested within them. Paired method differences will be summarized at that level. The final statistical analysis will specify **[TO SUPPLY: primary outcome, specimen count, uncertainty interval or hypothesis test, and handling of multiple comparisons]** before analysis of the independent evaluation cohort. Image comparisons will retain native-plane views and fixed region coordinates; display scaling will be stated, and quantitative metrics will be calculated from the underlying floating-point data.

## References cited in Section 3

[12] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” **IEEE Trans. Image Process. 13**(4), 600–612 (2004). [DOI](https://doi.org/10.1109/TIP.2003.819861).

## 作者待补说明（不属于投稿正文）

本节沿用指定 mean_anchor 400 步模型及其保存的评价结果。公式编号接续第2节，参考文献新增为[12]。正文中的 **[TO SUPPLY]** 和中文提示需在相应实验、参数和统计方案确定后替换，不表示实验已完成。

优先补齐四组证据：①质量匹配的少帧/高帧比较及完整重建计时；②多深度、多视野位置的实测分辨率；③菠菜根和鼠脑切片的独立重复与参考成像；④关键模块消融、噪声/相关长度/PSF失配及多训练种子。实际硬件参数、组织信息和样本数只从实验记录填写。

第2节已给出训练数据规模和优化配置，本节保留理解评价所需的数据划分；全文合并时可压缩第2节中重复的样本数量描述。当前交付不改动已经确认的前两节。
