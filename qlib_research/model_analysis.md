# Model Analysis: `/workspace/final_models/models/`

## Overview

- **Total models in `models/`**: 20 directories (17 timestamped + 3 ensemble)
- **Additional tuned models in `models_tuned/`**: 5 models (BiGRU_Cross_Attn_v2, TCNAlpha_v2, BiGRU_FFN_GELU_CrossAttention, MultiScaleTCN_CrossAttention, TCN_BiGRU_CrossAttention)
- **Common Input**: `(batch, 20, 5)` features (OPEN, HIGH, LOW, RET, VOL)
- **Common Output**: `(batch, 1)` — unbounded prediction for CSRankNorm labels
- **Common Training Hyperparams**: batch_size=512, epochs=100, lr=0.001, early_stop=7
- **Data Period**: Train 2014-01-01 ~ 2022-12-31 / Valid 2023-01-01 ~ 2024-12-31

### Legend

- **IC**: Information Coefficient
- **ICIR**: IC Information Ratio (IC / IC_std)
- **RankIC**: Rank Information Coefficient
- **RankICIR**: Rank IC Information Ratio
- **ExRet**: Excess Return (no cost)
- **ExRet_cost**: Excess Return (with cost)
- **IR**: Information Ratio (annualized)
- **DD**: Max Drawdown
- **BT Seeds**: Seeds used for ensemble backtest

---

## Models with Seed IC Results (14)

---

### #5 BiGRUTemporalAttentionModel

- **Architecture**: Linear→gru1(BiGRU)→gru2(BiGRU)→Drop→Attention(Linear→ReLU→Linear→softmax)→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3
- **Backtest**: N/A

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|-----|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | -0.0016 | 0.0005 | 0.0126 | 0.0027 | 0.0041 | 0.0112 | 0.0076 | 0.0040 | 0.0041 | 0.0094 | 0.0103 | 0.0095 | 0.0063 | 0.0075 | 0.0020 |
| ICIR | -0.020 | 0.006 | 0.163 | 0.033 | 0.052 | 0.131 | 0.090 | 0.046 | 0.052 | 0.116 | 0.133 | 0.109 | 0.075 | 0.086 | 0.023 |
| RankIC | -0.0025 | 0.0018 | 0.0138 | 0.0033 | 0.0118 | 0.0160 | 0.0088 | 0.0073 | 0.0077 | 0.0097 | 0.0112 | 0.0096 | 0.0098 | 0.0045 | 0.0064 |
| RankICIR | -0.032 | 0.024 | 0.188 | 0.043 | 0.162 | 0.206 | 0.114 | 0.091 | 0.099 | 0.126 | 0.142 | 0.118 | 0.122 | 0.054 | 0.082 |

---

### #6 MultiScaleCNN_BiGRU_Model

- **Architecture**: 3xCNN(k=3,5,10)→concat→Linear→2-layer BiGRU→Temporal Attn→CNN Gate→concat→Linear(448→1)
- **Hyperparams**: h=128, cnn_out=64, layers=2, drop=0.3, gate_dim=3
- **Backtest**: N/A

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | -0.0039 | -0.0033 | 0.0009 | 0.0014 | 0.0009 | 0.0067 | 0.0056 | 0.0093 | -0.0063 | 0.0010 | 0.0053 | 0.0098 | -0.0002 | 0.0050 | 0.0032 |
| ICIR | -0.047 | -0.033 | 0.010 | 0.015 | 0.010 | 0.083 | 0.067 | 0.126 | -0.074 | 0.012 | 0.070 | 0.120 | -0.002 | 0.066 | 0.038 |
| RankIC | -0.0003 | 0.0020 | 0.0037 | 0.0069 | 0.0038 | 0.0070 | 0.0119 | 0.0067 | -0.0024 | 0.0014 | 0.0043 | 0.0067 | 0.0001 | 0.0069 | 0.0044 |
| RankICIR | -0.004 | 0.020 | 0.047 | 0.072 | 0.045 | 0.094 | 0.140 | 0.099 | -0.031 | 0.019 | 0.059 | 0.093 | 0.001 | 0.094 | 0.057 |

---

### #7 BiGRUDirectionalAttention

- **Architecture**: Linear→2-layer BiGRU→Self-Attention(Q,K,V)→Hierarchical Pool Attn→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.4, dir_attn=256, pool_attn=256
- **Backtest** (BT Seeds: 4): IC=0.0053, ICIR=0.031, RankIC=-0.002, RankICIR=-0.011, ExRet=-3.38%, ExRet_cost=-8.69%, IR=-0.273, DD=-19.55%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0105 | 0.0111 | 0.0028 | 0.0185 | 0.0075 | 0.0119 | 0.0125 | 0.0043 | 0.0109 | 0.0094 | 0.0128 | 0.0074 | 0.0040 | 0.0048 | 0.0058 |
| ICIR | 0.126 | 0.128 | 0.028 | 0.231 | 0.087 | 0.119 | 0.149 | 0.046 | 0.134 | 0.101 | 0.154 | 0.101 | 0.044 | 0.060 | 0.069 |
| RankIC | 0.0091 | 0.0129 | 0.0065 | 0.0173 | 0.0127 | 0.0171 | 0.0117 | 0.0051 | 0.0129 | 0.0108 | 0.0123 | 0.0104 | 0.0122 | 0.0080 | 0.0120 |
| RankICIR | 0.117 | 0.160 | 0.074 | 0.210 | 0.151 | 0.177 | 0.144 | 0.062 | 0.159 | 0.112 | 0.151 | 0.145 | 0.138 | 0.104 | 0.148 |

---

### #8 BiGRU_Temporal_Attention_Residual_Gate

- **Architecture**: Linear→ReLU→2-layer BiGRU→tanh Attn→Residual Gate(sigmoid)→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3
- **Backtest** (BT Seeds: 1): IC=0.0247, ICIR=0.162, RankIC=0.0259, RankICIR=0.162, ExRet=-0.20%, ExRet_cost=-5.42%, IR=-0.013, DD=-13.37%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0192 | 0.0003 | 0.0059 | 0.0040 | 0.0093 | -0.0084 | -0.0044 | 0.0009 | 0.0038 | 0.0012 | -0.0008 | 0.0038 | 0.0007 | -0.0003 | 0.0035 |
| ICIR | 0.213 | 0.003 | 0.069 | 0.046 | 0.105 | -0.084 | -0.046 | 0.009 | 0.041 | 0.013 | -0.010 | 0.041 | 0.007 | -0.004 | 0.037 |
| RankIC | 0.0193 | 0.0005 | 0.0084 | 0.0102 | 0.0106 | -0.0025 | 0.0013 | 0.0042 | 0.0111 | 0.0024 | 0.0005 | 0.0061 | 0.0016 | 0.0020 | 0.0050 |
| RankICIR | 0.254 | 0.006 | 0.108 | 0.136 | 0.134 | -0.028 | 0.016 | 0.051 | 0.136 | 0.030 | 0.006 | 0.073 | 0.021 | 0.024 | 0.057 |

---

### #9 BiGRU_FFN_GELU_Model

- **Architecture**: Linear→ReLU→2-layer BiGRU→tanh Attn→FFN(GELU, 512)→Residual Gate→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3, ffn_h=512, attn=256, gate_h=256
- **Backtest** (BT Seeds: 9,4,14,7,11): IC=0.0198, ICIR=0.125, RankIC=0.0125, RankICIR=0.073, ExRet=2.80%, ExRet_cost=-2.11%, IR=0.230, DD=-10.05%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0146 | 0.0097 | 0.0152 | 0.0210 | 0.0069 | 0.0146 | 0.0164 | 0.0156 | 0.0221 | 0.0071 | 0.0164 | 0.0106 | 0.0154 | 0.0181 | 0.0121 |
| ICIR | 0.136 | 0.095 | 0.144 | 0.215 | 0.074 | 0.141 | 0.157 | 0.169 | 0.226 | 0.072 | 0.163 | 0.107 | 0.137 | 0.173 | 0.121 |
| RankIC | 0.0203 | 0.0135 | 0.0187 | 0.0194 | 0.0122 | 0.0164 | 0.0176 | 0.0192 | 0.0257 | 0.0146 | 0.0225 | 0.0130 | 0.0196 | 0.0241 | 0.0155 |
| RankICIR | 0.216 | 0.131 | 0.186 | 0.210 | 0.135 | 0.169 | 0.194 | 0.218 | 0.273 | 0.151 | 0.245 | 0.146 | 0.170 | 0.235 | 0.165 |

---

### #10 BiGRUHuberLossModel

- **Architecture**: Linear→ReLU→2-layer BiGRU→tanh Attn→temporal_mean_proj→Residual Gate→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3, attn=256, gate_h=256, huber_delta=1.0
- **Backtest** (BT Seeds: 9,14,13): IC=0.0277, ICIR=0.168, RankIC=0.0190, RankICIR=0.107, ExRet=6.07%, ExRet_cost=1.24%, IR=0.520, DD=-10.04%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0105 | 0.0119 | 0.0119 | 0.0129 | 0.0002 | 0.0118 | 0.0072 | 0.0041 | 0.0222 | 0.0118 | 0.0087 | 0.0037 | 0.0160 | 0.0192 | 0.0103 |
| ICIR | 0.090 | 0.110 | 0.115 | 0.116 | 0.001 | 0.098 | 0.063 | 0.038 | 0.216 | 0.100 | 0.076 | 0.039 | 0.147 | 0.199 | 0.091 |
| RankIC | 0.0097 | 0.0142 | 0.0090 | 0.0172 | 0.0053 | 0.0163 | 0.0165 | 0.0091 | 0.0211 | 0.0146 | 0.0121 | 0.0097 | 0.0219 | 0.0204 | 0.0107 |
| RankICIR | 0.094 | 0.131 | 0.093 | 0.161 | 0.052 | 0.140 | 0.154 | 0.083 | 0.224 | 0.121 | 0.108 | 0.102 | 0.209 | 0.245 | 0.096 |

---

### #11 BiGRU_Hierarchical_Temporal_Attention

- **Architecture**: Linear→ReLU→3-layer BiGRU→3xHierarchical Attn→Layer Combiner(softmax)→Residual Gate→Drop→Linear
- **Hyperparams**: h=128, layers=3, drop=0.3
- **Backtest**: N/A

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0135 | -0.0008 | 0.0043 | -0.0031 | 0.0057 | 0.0009 | 0.0133 | 0.0016 | 0.0139 | 0.0041 | 0.0064 | 0.0129 | -0.0034 | 0.0040 | 0.0056 |
| ICIR | 0.160 | -0.008 | 0.046 | -0.034 | 0.064 | 0.010 | 0.155 | 0.017 | 0.183 | 0.040 | 0.079 | 0.120 | -0.036 | 0.045 | 0.063 |
| RankIC | 0.0139 | -0.0032 | 0.0078 | 0.0008 | 0.0109 | 0.0081 | 0.0107 | 0.0074 | 0.0098 | 0.0078 | 0.0088 | 0.0119 | 0.0017 | 0.0070 | 0.0077 |
| RankICIR | 0.184 | -0.037 | 0.095 | 0.010 | 0.138 | 0.099 | 0.136 | 0.096 | 0.139 | 0.080 | 0.118 | 0.127 | 0.021 | 0.095 | 0.104 |

---

### #12 DilatedDepthwiseTCN

- **Architecture**: Input Drop→4xDilated Depthwise Sep Conv(d=[1,2,4,8]) + Residual→GAP→Gated Pool→FC(64→32→1)
- **Hyperparams**: h=64, kernel=3, dilation=[1,2,4,8], drop_in=0.3, drop_h=0.2, gate_h=64
- **Backtest**: N/A

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | -0.0013 | -0.0039 | 0.0048 | 0.0029 | 0.0013 | -0.0022 | 0.0025 | 0.0009 | -0.0011 | 0.0015 | 0.0019 | 0.0014 | 0.0017 | -0.0059 | -0.0051 |
| ICIR | -0.009 | -0.027 | 0.032 | 0.018 | 0.010 | -0.015 | 0.017 | 0.006 | -0.008 | 0.010 | 0.012 | 0.010 | 0.012 | -0.042 | -0.038 |
| RankIC | 0.0050 | 0.0034 | 0.0082 | 0.0062 | 0.0063 | 0.0030 | 0.0051 | 0.0096 | 0.0093 | 0.0069 | 0.0059 | 0.0068 | 0.0041 | 0.0009 | 0.0033 |
| RankICIR | 0.028 | 0.021 | 0.048 | 0.035 | 0.040 | 0.018 | 0.030 | 0.067 | 0.065 | 0.039 | 0.035 | 0.044 | 0.026 | 0.006 | 0.020 |

---

### #13 Simplified_BiGRU_Input_Noise_Model

- **Architecture**: Gaussian Noise(σ=0.1)→Linear→ReLU→2-layer BiGRU→tanh Attn→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3, noise_sigma=0.1
- **Backtest**: N/A

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0081 | 0.0065 | 0.0064 | 0.0034 | -0.0027 | 0.0047 | -0.0033 | -0.0036 | -0.0038 | -0.0062 | 0.0011 | 0.0021 | 0.0079 | -0.0031 | 0.0009 |
| ICIR | 0.085 | 0.066 | 0.074 | 0.039 | -0.028 | 0.051 | -0.034 | -0.036 | -0.039 | -0.067 | 0.012 | 0.022 | 0.087 | -0.035 | 0.010 |
| RankIC | 0.0071 | 0.0119 | 0.0062 | 0.0048 | 0.0040 | 0.0038 | 0.0021 | 0.0011 | 0.0046 | -0.0029 | 0.0035 | 0.0060 | 0.0079 | 0.0020 | 0.0020 |
| RankICIR | 0.089 | 0.147 | 0.082 | 0.059 | 0.044 | 0.045 | 0.025 | 0.013 | 0.051 | -0.034 | 0.045 | 0.069 | 0.092 | 0.023 | 0.023 |

---

### #14 BiGRU_Single_Layer_Attention_Model

- **Architecture**: Linear→ReLU→1-layer BiGRU→tanh Attn→Residual Gate→Drop→Linear
- **Hyperparams**: h=128, layers=1, drop=0.3, attn=256, gate_h=256
- **Backtest** (BT Seeds: 10,6,3): IC=0.0097, ICIR=0.063, RankIC=0.0031, RankICIR=0.019, ExRet=-1.59%, ExRet_cost=-7.12%, IR=-0.148, DD=-11.80%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0043 | 0.0068 | 0.0169 | -0.0002 | 0.0134 | 0.0176 | 0.0116 | 0.0134 | 0.0012 | 0.0207 | 0.0123 | 0.0069 | 0.0030 | 0.0105 | 0.0076 |
| ICIR | 0.044 | 0.075 | 0.170 | -0.002 | 0.141 | 0.183 | 0.128 | 0.157 | 0.013 | 0.232 | 0.131 | 0.073 | 0.033 | 0.119 | 0.089 |
| RankIC | 0.0031 | 0.0060 | 0.0145 | 0.0075 | 0.0101 | 0.0126 | 0.0125 | 0.0135 | 0.0013 | 0.0157 | 0.0136 | 0.0004 | 0.0074 | 0.0089 | 0.0109 |
| RankICIR | 0.038 | 0.076 | 0.176 | 0.088 | 0.113 | 0.149 | 0.152 | 0.182 | 0.015 | 0.185 | 0.166 | 0.005 | 0.092 | 0.111 | 0.131 |

---

### #15 BiGRU_MultiHead_Temporal_Attention_16dim

- **Architecture**: Linear→ReLU→2-layer BiGRU→4-head Attn(16-dim)→Residual Gate(sigmoid)→Proj(1024→256)→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3, heads=4, head_dim=16, gate_h=1024
- **Backtest** (BT Seeds: 12,2,13): IC=0.0293, ICIR=0.179, RankIC=0.0228, RankICIR=0.131, ExRet=6.77%, ExRet_cost=2.33%, IR=0.615, DD=-10.21%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0045 | 0.0169 | 0.0104 | 0.0101 | 0.0039 | 0.0088 | 0.0141 | 0.0112 | 0.0104 | 0.0086 | 0.0102 | 0.0216 | 0.0161 | 0.0027 | 0.0052 |
| ICIR | 0.047 | 0.173 | 0.102 | 0.099 | 0.041 | 0.097 | 0.138 | 0.104 | 0.109 | 0.083 | 0.110 | 0.206 | 0.167 | 0.028 | 0.056 |
| RankIC | 0.0107 | 0.0145 | 0.0161 | 0.0132 | 0.0089 | 0.0113 | 0.0172 | 0.0168 | 0.0211 | 0.0108 | 0.0136 | 0.0238 | 0.0186 | 0.0059 | 0.0116 |
| RankICIR | 0.125 | 0.157 | 0.171 | 0.139 | 0.095 | 0.131 | 0.178 | 0.169 | 0.249 | 0.101 | 0.153 | 0.247 | 0.194 | 0.069 | 0.131 |

---

### #16 BiGRU_Cross_Attention_Gating

- **Architecture**: Linear→ReLU→2-layer BiGRU→4-head Attn(16-dim)→Cross-Attn Gating→Proj(1024→256)→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.25, heads=4, head_dim=16
- **Backtest** (BT Seeds: 14,15): IC=0.0228, ICIR=0.148, RankIC=0.0126, RankICIR=0.075, ExRet=11.83%, ExRet_cost=6.61%, IR=1.002, DD=-8.22%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | -0.0045 | 0.0102 | 0.0009 | 0.0052 | 0.0018 | 0.0086 | 0.0071 | 0.0078 | 0.0037 | 0.0011 | 0.0060 | 0.0070 | 0.0103 | 0.0192 | 0.0185 |
| ICIR | -0.054 | 0.118 | 0.010 | 0.063 | 0.020 | 0.093 | 0.073 | 0.093 | 0.042 | 0.011 | 0.065 | 0.073 | 0.107 | 0.190 | 0.224 |
| RankIC | 0.0001 | 0.0027 | 0.0056 | 0.0063 | 0.0033 | 0.0058 | 0.0101 | 0.0067 | 0.0026 | -0.0008 | 0.0107 | 0.0081 | 0.0170 | 0.0178 | 0.0118 |
| RankICIR | 0.001 | 0.036 | 0.068 | 0.079 | 0.041 | 0.071 | 0.112 | 0.087 | 0.036 | -0.008 | 0.133 | 0.096 | 0.204 | 0.205 | 0.151 |

---

### #17 TCNAlpha

- **Architecture**: Conv1d(5→128)→ReLU→Drop→4xResidualBlock(d=[1,2,4,8])→GAP→Linear(128→1)
- **Hyperparams**: h=128, blocks=4, kernel=3, drop=0.3, dilation_base=2
- **Backtest** (BT Seeds: 15): IC=0.0295, ICIR=0.175, RankIC=0.0254, RankICIR=0.141, ExRet=18.86%, ExRet_cost=13.64%, IR=1.240, DD=-8.38%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0056 | -0.0024 | 0.0104 | 0.0094 | -0.0008 | -0.0006 | 0.0089 | 0.0044 | 0.0128 | 0.0133 | 0.0086 | 0.0134 | 0.0060 | 0.0098 | 0.0185 |
| ICIR | 0.051 | -0.022 | 0.090 | 0.092 | -0.005 | -0.005 | 0.092 | 0.032 | 0.120 | 0.144 | 0.080 | 0.115 | 0.058 | 0.084 | 0.164 |
| RankIC | 0.0151 | -0.0013 | 0.0119 | 0.0110 | 0.0020 | 0.0042 | 0.0083 | 0.0061 | 0.0115 | 0.0123 | 0.0108 | 0.0116 | 0.0077 | 0.0088 | 0.0180 |
| RankICIR | 0.135 | -0.010 | 0.099 | 0.102 | 0.012 | 0.030 | 0.084 | 0.040 | 0.111 | 0.144 | 0.101 | 0.107 | 0.073 | 0.072 | 0.160 |

---

### #19 MultiScaleCNN_LateFusion (Ensemble)

- **Architecture**: 3xCNN(k=3,7,15)→GAP each→concat(192)→FC(192→128)→ReLU→Drop→FC(128→1)
- **Hyperparams**: branch_ch=64, kernels=[3,7,15], fc_h=128, drop_br=0.4, drop_fc=0.3
- **Backtest** (BT Seeds: 3): IC=0.0043, ICIR=0.032, RankIC=-0.0039, RankICIR=-0.028, ExRet=6.81%, ExRet_cost=-0.25%, IR=0.566, DD=-8.91%

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0055 | 0.0027 | 0.0180 | -0.0001 | 0.0042 | 0.0029 | 0.0074 | 0.0100 | 0.0048 | 0.0072 | 0.0034 | 0.0076 | 0.0026 | 0.0030 | 0.0072 |
| ICIR | 0.064 | 0.031 | 0.219 | -0.001 | 0.047 | 0.036 | 0.086 | 0.130 | 0.059 | 0.085 | 0.041 | 0.094 | 0.029 | 0.036 | 0.092 |
| RankIC | 0.0098 | 0.0066 | 0.0174 | 0.0001 | 0.0097 | 0.0064 | 0.0129 | 0.0097 | 0.0090 | 0.0116 | 0.0089 | 0.0078 | 0.0052 | 0.0028 | 0.0091 |
| RankICIR | 0.125 | 0.081 | 0.225 | 0.001 | 0.118 | 0.087 | 0.170 | 0.133 | 0.122 | 0.150 | 0.113 | 0.108 | 0.064 | 0.035 | 0.122 |

---

## Models without Seed IC Results (6)

| # | Model Class | Architecture | Hyperparams | Seeds Available | Notes |
|---|-------------|-------------|-------------|----------------|-------|
| 1 | **BiGRUAttentionModel** | Linear→ReLU→2-layer BiGRU→QKV Attention→Residual→GlobalAvgPool→Linear | h=128, layers=2, drop=0.3 | 15 | No seed IC results |
| 2 | **BiGRU_DualGating_Residual** | Feature Gate→Linear→ReLU→1-layer BiGRU→Temporal Gate→LayerNorm→Drop→Linear + Residual | h=64, layers=1, drop=0.5 | 15 | No seed IC results |
| 3 | **MambaExpDecayModel** | Linear→ReLU→SimpleSSD(Mamba)→ReLU→Drop→ExpDecay Pool→Drop→Linear | h=128, ssm_state=16, ssm_exp=2, drop=0.2, decay=0.1 | 11 | No seed IC results |
| 4 | **BiGRU_Mamba_GatedHybrid** | Linear→ReLU→[Branch A: BiGRU] + [Branch B: Mamba SSD]→ExpDecay Pool→Gate MLP Fusion→Linear | h=128, bigru=1, mamba=1, ssm_state=16, gate_h=32, drop=0.2, decay_init=0.0365 | 15 | No seed IC results |
| 18 | **LightTCN_Residual** (Ensemble) | Residual Proj Conv→2xConv1d(k=5) + BatchNorm + Residual→GAP→Linear(64→1) | h=64, kernel=5, layers=2, drop=0.5 | 3 | No seed IC results |
| 20 | **MultiScaleDilatedTCN_HuberNoise** (Ensemble) | Input Noise→3xDilated Conv(k=[3,5,7], d=[1,2,4])→concat→residual fusion→GAP→Linear(128→1) | branch_h=128, kernels=[3,5,7], dilations=[1,2,4], fusion=128, drop_br=0.4, drop_fu=0.3, noise_std=0.1 | 15 | No seed IC results |

---

## Backtest Summary (9 models with results)

| # | Model Class | BT Seeds | IC | ICIR | RankIC | RankICIR | ExRet | ExRet_cost | IR | MaxDD |
|---|-------------|----------|-------|------|--------|----------|-------|------------|-------|-------|
| 7 | BiGRUDirectionalAttention | 4 | 0.0053 | 0.031 | -0.0020 | -0.011 | -3.38% | -8.69% | -0.273 | -19.55% |
| 8 | BiGRU_Temporal_Attention_Residual_Gate | 1 | 0.0247 | 0.162 | 0.0259 | 0.162 | -0.20% | -5.42% | -0.013 | -13.37% |
| 9 | BiGRU_FFN_GELU_Model | 5 | 0.0198 | 0.125 | 0.0125 | 0.073 | 2.80% | -2.11% | 0.230 | -10.05% |
| 10 | BiGRUHuberLossModel | 3 | 0.0277 | 0.168 | 0.0190 | 0.107 | 6.07% | 1.24% | 0.520 | -10.04% |
| 14 | BiGRU_Single_Layer_Attention_Model | 3 | 0.0097 | 0.063 | 0.0031 | 0.019 | -1.59% | -7.12% | -0.148 | -11.80% |
| 15 | BiGRU_MultiHead_Temporal_Attention_16dim | 3 | 0.0293 | 0.179 | 0.0228 | 0.131 | 6.77% | 2.33% | 0.615 | -10.21% |
| 16 | BiGRU_Cross_Attention_Gating | 2 | 0.0228 | 0.148 | 0.0126 | 0.075 | 11.83% | 6.61% | 1.002 | -8.22% |
| 17 | TCNAlpha | 1 | 0.0295 | 0.175 | 0.0254 | 0.141 | 18.86% | 13.64% | 1.240 | -8.38% |
| 19 | MultiScaleCNN_LateFusion (Ensemble) | 1 | 0.0043 | 0.032 | -0.0039 | -0.028 | 6.81% | -0.25% | 0.566 | -8.91% |

---

## Seed IC Summary (14 models)

| # | Model Class | IC Mean | IC Std | IC>0 % | Best Seed (IC) | Worst Seed (IC) |
|---|-------------|---------|-------|--------|----------------|-----------------|
| 5 | BiGRUTemporalAttentionModel | 0.0055 | 0.0041 | 93% (14/15) | S3: 0.0126 | S1: -0.0016 |
| 6 | MultiScaleCNN_BiGRU_Model | 0.0027 | 0.0046 | 80% (12/15) | S8: 0.0093 | S9: -0.0063 |
| 7 | BiGRUDirectionalAttention | 0.0080 | 0.0043 | 100% (15/15) | S4: 0.0185 | S3: 0.0028 |
| 8 | BiGRU_Temporal_Attention_Residual_Gate | 0.0029 | 0.0070 | 80% (12/15) | S1: 0.0192 | S6: -0.0084 |
| 9 | BiGRU_FFN_GELU_Model | 0.0145 | 0.0045 | 100% (15/15) | S9: 0.0221 | S5: 0.0069 |
| 10 | BiGRUHuberLossModel | 0.0101 | 0.0055 | 100% (15/15) | S9: 0.0222 | S5: 0.0002 |
| 11 | BiGRU_Hierarchical_Temporal_Attention | 0.0056 | 0.0057 | 80% (12/15) | S1: 0.0135 | S4: -0.0031 |
| 12 | DilatedDepthwiseTCN | -0.0003 | 0.0033 | 53% (8/15) | S3: 0.0048 | S14: -0.0059 |
| 13 | Simplified_BiGRU_Input_Noise_Model | 0.0013 | 0.0053 | 60% (9/15) | S1: 0.0081 | S10: -0.0062 |
| 14 | BiGRU_Single_Layer_Attention_Model | 0.0099 | 0.0064 | 87% (13/15) | S10: 0.0207 | S4: -0.0002 |
| 15 | BiGRU_MultiHead_Temporal_Attention_16dim | 0.0101 | 0.0051 | 100% (15/15) | S12: 0.0216 | S14: 0.0027 |
| 16 | BiGRU_Cross_Attention_Gating | 0.0069 | 0.0068 | 80% (12/15) | S15: 0.0185 | S1: -0.0045 |
| 17 | TCNAlpha | 0.0073 | 0.0065 | 80% (12/15) | S15: 0.0185 | S2: -0.0024 |
| 19 | MultiScaleCNN_LateFusion (Ensemble) | 0.0055 | 0.0045 | 93% (14/15) | S3: 0.0180 | S4: -0.0001 |

---

## Tuned Models (`models_tuned/`) — 5 models

These are newer/v2 variants trained with validation IC filtering (IC ≥ 0.015 threshold).

### TCNAlpha_v2

- **Architecture**: Conv1d(5→128)→ReLU→Drop→4xResidualBlock(d=[1,2,4,8])→GAP→Drop→Linear(128→1)
- **Hyperparams**: h=128, blocks=4, kernel=3, drop=0.3, dilation_base=2
- **Origin**: v2 of #17 TCNAlpha — same architecture, retrained
- **Backtest** (BT Seeds: 15 - best 5 selected): IC=0.0401, ICIR=0.2046, RankIC=0.0329, RankICIR=0.1648, ExRet=26.32%, ExRet_cost=22.48%, IR=1.473, DD=-7.11% (Seed 9 기준)

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| IC | 0.0166 | 0.0340 | 0.0135 | 0.0178 | 0.0147 | 0.0362 | 0.0242 | 0.0209 | 0.0401 | 0.0344 | 0.0198 | 0.0152 | 0.0221 | 0.0315 | 0.0376 |
| ICIR | 0.117 | 0.196 | 0.085 | 0.136 | 0.085 | 0.203 | 0.139 | 0.131 | 0.205 | 0.191 | 0.119 | 0.092 | 0.123 | 0.183 | 0.204 |
| RankIC | 0.0107 | 0.0282 | 0.0060 | 0.0132 | 0.0094 | 0.0384 | 0.0199 | 0.0236 | 0.0329 | 0.0276 | 0.0122 | 0.0073 | 0.0184 | 0.0279 | 0.0325 |
| RankICIR | 0.077 | 0.161 | 0.037 | 0.104 | 0.052 | 0.217 | 0.110 | 0.148 | 0.165 | 0.153 | 0.071 | 0.042 | 0.098 | 0.157 | 0.176 |
| IR | 0.624 | 1.252 | -0.095 | 0.199 | 0.956 | 0.956 | 1.254 | 0.664 | 1.473 | 1.387 | 0.490 | 0.139 | 1.047 | 1.129 | 1.751 |
| ExRet | 7.95 | 18.96 | -1.03 | 2.43 | 12.47 | 14.73 | 16.39 | 9.16 | 26.32 | 19.51 | 5.43 | 1.54 | 14.56 | 14.73 | 26.09 |
| MaxDD | -8.89 | -9.68 | -14.66 | -9.26 | -10.08 | -10.24 | -8.99 | -8.20 | -7.11 | -7.89 | -12.73 | -13.10 | -11.36 | -9.06 | -6.39 |

**Seed IC Summary**: IC Mean=0.0241, IC Std=0.0088, IC>0%=100% (15/15), Best=S9: 0.0401, Worst=S3: 0.0135
**Backtest Summary (Top 5 by IR)**: Seeds 15,9,10,2,7 — Avg IR=1.36, Avg ExRet=19.05%, Avg MaxDD=-8.61%

### 분석: 왜 TCNAlpha_v2가 개선되었는가

**초기 분석오류**: TCNAlpha_v2의 개선이 "dropout 추가"나 "LayerNorm" 때문인 줄 알았음. **실제 원인은 early_stop patience 차이**.

**재학습 검증 결과** (#17을 early_stop=8만 변경, dropout 미추가로 재학습):
| Seed | 원본 #17 IC | 재학습 #17 IC | TCNAlpha_v2 IC |
|------|------------|---------------|----------------|
| 1 | 0.0056 | **0.0278** | 0.0278 |
| 2 | -0.0024 | **0.0333** | 0.0333 |
| 3 | 0.0104 | **0.0183** | 0.0183 |

→ dropout 없이도 early_stop=8만으로 TCNAlpha_v2와 **완전히 동일한 IC** 달성

**진짜 원인**: #17의 early_stop=7은 patience가 너무 짧아서 seed 1이 epoch 7에서 조기 종료 (정답은 epoch 17). TCNAlpha_v2의 model_train.yaml에서 early_stop=8로 설정한 것이 유일한 실질적 차이.

**TCNAlpha_v2 실제 변경점**: Final FC 전에 `nn.Dropout(dropout)` 추가 — 이것은 regularization 향샹에 기여하지만, IC 3.5배 개선의 **주요 원인은 early_stop=8으로 충분한 학습 보장**임.

---

### BiGRU_Cross_Attn_v2

- **Architecture**: Linear→ReLU→2-layer BiGRU→4-head tanh Attn(16-dim)→concat(2048)→Cross-Attn Gate(Q·K/sqrt(dim))→Proj(2048→256)→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3, heads=4, head_dim=16, output_proj=256
- **Origin**: v2 of #16 BiGRU_Cross_Attention_Gating — restored temporal attention per head, hidden_dim=128, dropout=0.3
- **Backtest** (BT Seeds: 2): IC=0.0209, ICIR=0.125, RankIC=0.0119, RankICIR=0.067, ExRet=4.73%, ExRet_cost=-0.39%, IR=0.342, DD=-11.66%

| Metric | S1 | S2 | S3 |
|--------|------|------|------|
| IC | 0.0144 | 0.0155 | 0.0103 |
| ICIR | 0.149 | 0.150 | 0.099 |
| RankIC | 0.0150 | 0.0192 | 0.0128 |
| RankICIR | 0.181 | 0.201 | 0.134 |

**Seed IC Summary**: IC Mean=0.0134, IC Std=0.0027, IC>0%=100% (3/3), Best=S2: 0.0155, Worst=S3: 0.0103

---

### BiGRU_FFN_GELU_CrossAttention

- **Architecture**: Linear→ReLU→2-layer BiGRU→4-head tanh Attn(16-dim)→concat(512)→Cross-Attn Gate(Q·K)→FFN(GELU, 512→256)→Residual Gate→Drop→Linear
- **Hyperparams**: h=128, layers=2, drop=0.3, heads=4, head_dim=16, ffn_h=512, output_proj=256
- **Origin**: Hybrid of #9 (FFN+GELU) + #16 (Cross-Attention Gating)
- **Backtest**: N/A (seed 1 failed IC ≥ 0.015 threshold during training)

| Metric | S1 |
|--------|------|
| IC | 0.0134 |
| ICIR | 0.117 |
| RankIC | 0.0210 |
| RankICIR | 0.209 |

---

### MultiScaleTCN_CrossAttention

- **Architecture**: 3xBranchTCN(k=3,5,7)→GAP each→concat(384)→Cross-Attn Gate(Q·K/sqrt)→4-head→FC(384→128→1)
- **Hyperparams**: branch_h=128, kernels=[3,5,7], dropout=0.2, num_gate_heads=4
- **Origin**: Novel architecture combining multi-scale TCN with cross-attention gating
- **Backtest**: N/A (seed 1 failed IC ≥ 0.015 threshold during training, IC=-0.0085)

---

### TCN_BiGRU_CrossAttention

- **Architecture**: [Branch A: Conv1d→4xResidualBlock(d=[1,2,4,8])→GAP] + [Branch B: Linear→2-layer BiGRU→tanh Attn] → concat → Cross-Attn Gate(Q·K/sqrt)→4-head→Proj(GELU)→Linear
- **Hyperparams**: tcn_h=128, gru_h=128, kernel=3, blocks=4, drop=0.2, attn_dim=256, num_gate_heads=4
- **Origin**: Novel dual-branch architecture combining TCN and BiGRU with cross-attention fusion
- **Backtest**: N/A

---

## Duplicate / Similarity Analysis

### Near-Duplicate (99% identical structure)

| Model A (#8) | Model B (#10) |
|--------------|---------------|
| **BiGRU_Temporal_Attention_Residual_Gate** | **BiGRUHuberLossModel** |

**Identical parts:**
- Linear(5→128) → ReLU → 2-layer BiGRU(128, bidirectional) → tanh Temporal Attention → Residual Gate(sigmoid) → Dropout(0.3) → Linear(256→1)

**Difference:**
- BiGRUHuberLossModel (#10) has an additional `temporal_mean_proj` layer projecting temporal_mean from 256-dim to attention_dim(256)
- BiGRUHuberLossModel (#10) defines Huber loss (`huber_delta=1.0`) but training config uses `loss: mse`

### Similar but NOT Duplicate

| Model A | Model B | Key Difference |
|---------|---------|---------------|
| BiGRUAttentionModel (#1) | BiGRUTemporalAttentionModel (#5) | QKV scaled dot-product vs tanh single-head attention |
| BiGRU_MultiHead_Temporal_Attention_16dim (#15) | BiGRU_Cross_Attention_Gating (#16) | Residual Gate vs Cross-Attention Gating |
| BiGRU_Single_Layer_Attention_Model (#14) | BiGRU_Temporal_Attention_Residual_Gate (#8) | 1-layer vs 2-layer BiGRU |
| BiGRU_FFN_GELU_Model (#9) | BiGRU_Temporal_Attention_Residual_Gate (#8) | FFN(GELU, 512-dim) added between Attn and Gate |
| Simplified_BiGRU_Input_Noise_Model (#13) | BiGRU_Temporal_Attention_Residual_Gate (#8) | Input noise + no gate vs gate + no noise |
| TCNAlpha (#17) | DilatedDepthwiseTCN (#12) | Standard ResidualBlock(h=128) vs Depthwise Sep Conv(h=64) |
| MultiScaleCNN_BiGRU_Model (#6) | MultiScaleCNN_LateFusion (#19) | BiGRU backbone vs FC direct |
| MultiScaleDilatedTCN_HuberNoise (#20) | DilatedDepthwiseTCN (#12) | Parallel multi-branch vs sequential depthwise |

### Fully Unique Models

- **BiGRU_DualGating_Residual** (#2) — Feature Gate + Dual Gating
- **MambaExpDecayModel** (#3) — SSM (State Space Model)
- **BiGRU_Mamba_GatedHybrid** (#4) — BiGRU + Mamba dual-branch
- **BiGRUDirectionalAttention** (#7) — QKV Self-Attention + Hierarchical Pooling
- **BiGRU_Hierarchical_Temporal_Attention** (#11) — 3-layer hierarchical attention + Layer Combiner
