# MicroRTS PPO 訓練框架

## 專案概述
本專案提供基於 Proximal Policy Optimization (PPO) 演算法，在 MicroRTS 環境下進行強化學習訓練的腳本與設定。包含兩種主要實現：  
- `ppo_diverse_impala.py`：結合 Impala 架構的多樣化 PPO  
- `ppo_diverse_maxcho.py`：針對 MicroRTS 特性優化的 PPO  

此外，透過 Git submodule 引用了 [gym-microrts-paper](gym-microrts-paper) 作為環境與基準實現。

---

## 目錄結構
```
/max52015/microrts_ppo
├── gym-microrts-paper/       # Submodule：MicroRTS 環境與基準程式
├── .gitignore
├── .gitmodules
├── ppo_diverse_impala.py     # Impala + PPO 訓練腳本
├── ppo_diverse_maxcho.py     # Maxcho 自訂 PPO 訓練腳本
├── requirements.txt          # Python 相依套件
└── test.py                   # CUDA & 環境可用性測試
```

---

## 前置需求
- Python 3.8+  
- NVIDIA GPU（建議 CUDA 11.8+）  
- Git（用於管理 submodule）  
- 建議使用虛擬環境：conda 或 venv

---

## 安裝步驟

1. **Clone（含 submodule）**  
   ```bash
   git clone --recursive <你的專案-url>.git
   cd microrts_ppo
   ```  
   若已經 clone 遺漏 `--recursive`，可補上：  
   ```bash
   git submodule update --init --recursive
   ```

2. **準備虛擬環境**  
   ```bash
   # conda
   conda create -n microrts_ppo python=3.8 -y
   conda activate microrts_ppo

   # 或 venv
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安裝相依套件**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 使用說明

### 1. 測試 CUDA 與環境  
```bash
python test.py
```  
應輸出：
```
True
NVIDIA GeForce RTX 3060 Ti
```

### 2. 執行訓練腳本  
- **ppo_diverse_impala.py**  
  ```bash
  python ppo_diverse_impala.py     --env-id MicrortsMapXXX     --num-envs 8     --total-timesteps 10000000     # ... 更多參數
  ```

- **ppo_diverse_maxcho.py**  
  ```bash
  python ppo_diverse_maxcho.py     --env-id MicrortsMapXXX     --num-envs 4     --total-timesteps 5000000     # ... 更多參數
  ```

> 詳細參數請參考腳本中的 `--help`：  
> ```bash
> python ppo_diverse_impala.py --help
> python ppo_diverse_maxcho.py --help
> ```

---

## 常見問題

- **GPU 記憶體不足？**  
  可調整 `--num-envs`、`--batch-size` 或減少 `--ppo-epochs`。

- **Submodule 更新**  
  ```bash
  cd gym-microrts-paper
  git pull
  cd ..
  git add gym-microrts-paper
  git commit -m "Update submodule gym-microrts-paper"
  ```

---

## 授權與貢獻
本專案採用 MIT 授權，歡迎提出 Issue 或 Pull Request。

---

> **懷疑效果？**  
> 不妨同時執行兩個腳本，實測比較再下結論！
