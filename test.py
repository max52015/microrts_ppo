# test.py

import torch
import numpy as np
import random
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
import gym_microrts
from gym_microrts import microrts_ai

# 將兩個版本的 agent 模組匯入
import ppo_diverse_impala as impala_mod  # ﹣ impala 的 Agent 定義在此檔案中 :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
import ppo_diverse_maxcho as maxcho_mod  # ﹣ maxcho 的 Agent 定義在此檔案中 :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
# 1. 參數設定（與原始檔案保持一致）
NUM_ENVS = 4
SEED = 42

# 2. 隨機種子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 3. 建立環境（與原始程式相同）
envs = MicroRTSVecEnv(
    num_envs=NUM_ENVS,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(NUM_ENVS)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
# 原始檔中還套用了 StatsRecorder、Monitor、VecPyTorch 等包裝器，
# 但為了測試 action/value 的一致性，可以只用最基本的 envs.reset()/step()。

# 4. 建立兩個 Agent 實例
# --- Impala 版環境與 Agent ---
impala_args = impala_mod.parse_args()
device_impala, envs_for_impala, _, _, _, _ = impala_mod.set_environment(impala_args)
agent_impala = impala_mod.Agent(envs_for_impala, impala_args).to(device_impala)

# --- Maxcho 版環境與 Agent ---
args_maxcho, device_maxcho, envs_for_maxcho, _, _, _, _ = maxcho_mod.set_environment()
agent_maxcho = maxcho_mod.Agent(envs_for_maxcho, device_maxcho).to(device_maxcho)

# 5. 取一次 batch 的觀測（回傳 Tensor）
obs = envs_for_impala.reset()

# 6. 不用 from_numpy，直接 cast＆搬到 GPU/CPU
obs_tensor = obs.float().to(device_impala)

with torch.no_grad():
    val_impala = agent_impala.get_value(obs_tensor)
    val_maxcho = agent_maxcho.get_value(obs_tensor.to(device_maxcho))


print("Value comparison:")
print(" ImpalaAgent:", val_impala.cpu().numpy())
print(" MaxchoAgent:", val_maxcho.cpu().numpy())
print(" Values equal:", np.allclose(val_impala.cpu().numpy(), val_maxcho.cpu().numpy()))

# 7. 比較 get_action（採 sample 模式）
with torch.no_grad():
    # impala 的 get_action: returns (action, logprob, entropy, masks)
    act_impala, logp_impala, ent_impala, mask_impala = agent_impala.get_action(obs_tensor)
    # maxcho 的 get_action: returns (action, logprob, entropy, masks)
    act_maxcho, logp_maxcho, ent_maxcho, mask_maxcho = agent_maxcho.get_action(obs_tensor)

print("\nAction comparison:")
print(" ImpalaAgent action:", act_impala)
print(" MaxchoAgent action:", act_maxcho)
print(" Actions equal:", torch.equal(act_impala, act_maxcho))

print("\nLogProb comparison:")
print(" ImpalaAgent logp:", logp_impala)
print(" MaxchoAgent logp:", logp_maxcho)
print(" LogProbs close:", torch.allclose(logp_impala, logp_maxcho))

print("\nEntropy comparison:")
print(" ImpalaAgent entropy:", ent_impala)
print(" MaxchoAgent entropy:", ent_maxcho)
print(" Entropies close:", torch.allclose(ent_impala, ent_maxcho))

print("\nMask comparison:")
print(" ImpalaAgent mask:", mask_impala)
print(" MaxchoAgent mask:", mask_maxcho)
print(" Masks equal:", torch.equal(mask_impala, mask_maxcho))
