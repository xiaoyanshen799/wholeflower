# CPU Warm-up 与校准脚本

入口：`femnist/warmup_control.py`。外部独立流程，不需要 Flower server，不修改在线 Pacer 的 target/gamma 协议，也不会在校准完成后自动开始正式联邦训练。

## 默认流程

1. 所有选中客户端同时使用 30%、50%、70%、90% CPU，依次执行四批，每批 50 轮。
2. 每批原始日志保留全部 50 轮，拟合时仅舍弃第 1 轮，使用剩余 49 轮。每个客户端、每档配额分别拟合正半轴截断 Logistic，保存 `theta_s`、`k_s`。
3. 每个客户端用四个 `(cpu, theta)` 点拟合 `theta(c) = mu * c**(-beta) + theta_floor_s`。
4. 固定 `target = max_i(theta_i_at_90_percent)`，这里是 90% 档实测样本拟合的 θ，不是 CPU 曲线预测值、最大样本、P90 或 deadline。
5. 对 CPU 曲线求逆，按可执行粒度取整，得到各客户端的初始配额。
6. 所有客户端按当前配额重新运行 30 轮。默认也舍弃重启后的第 1 轮，使用剩余 29 轮重新拟合 θ。
7. 计算 `abs(measured_theta - target) / target`，只有大于 0.03 的客户端调整配额，其他客户端配额不变。下一批仍重新测量全部客户端。
8. 重复 30 轮验证，直到同一批中全员达标，或者达到验证次数上限。两种情况均输出最后一批实测 CPU 配置并退出，不启动正式训练；达到上限但未全员达标时明确标记为未收敛。

CPU 百分比是 **一个逻辑 CPU 的配额百分比**，不是整台机器的百分比。`taskset` 固定每个客户端的核，systemd `CPUQuota` 限制该客户端整个进程组的 CPU 时间。默认从当前进程允许使用的 CPU 中选择不同物理核；`cpu_ids` 可手动指定不同逻辑核。CPU 训练期间隐藏 GPU，并限制 TF/BLAS 线程数为 1。

## 配置与运行

使用 `/home/xiaoyan/wholeflower/venv/bin/python`。配置例子在 `femnist/examples/warmup.example.json`，先按实际实验修改 `dataset`、`model`、`data_dir`、`client_ids`、`epochs`、`batch_size` 等参数。示例路径和客户端 ID 不是自动检测后的实验配置。

配置里的相对路径以 `femnist/` 为基准，与启动命令的工作目录无关。客户端文件名沿用现有入口的 `client_00000.npz` 格式。省略 `client_ids` 会读取数据目录中的全部分区；可增加 `"cpu_ids": [0, 2, 4, 6]`，与 `client_ids` 顺序对应。

默认仅检查配置并打印计划，不启动客户端、不创建 systemd 单元：

```bash
/home/xiaoyan/wholeflower/venv/bin/python /home/xiaoyan/wholeflower/femnist/warmup_control.py \
  --config /home/xiaoyan/wholeflower/femnist/examples/warmup.example.json
```

无真实训练的模拟测试，不要求真实数据目录，结果位于独立的 `output_dir.simulation` 目录，不占用真实实验输出目录：

```bash
/home/xiaoyan/wholeflower/venv/bin/python /home/xiaoyan/wholeflower/femnist/warmup_control.py \
  --config /home/xiaoyan/wholeflower/femnist/examples/warmup.example.json --simulate
```

将来确认实验参数后，显式增加 `--execute` 才会执行真实 warm-up。需要 Linux cgroup v2、systemd 255+、`taskset` 和持续可用的非交互 `sudo -n` 权限。`sudo -v` 只会临时刷新凭据，可能在长实验途中过期；应由管理员确认实验所需的权限配置。脚本不会修改 sudoers 或系统权限。每个服务另设 `RuntimeMaxSec` 作为控制器意外退出后的最长运行时间保护。

实验中断后，使用相同配置、相同输出目录加 `--execute --resume`。完整且校验通过的批次会复用；未完成批次会新建 `attempt_002` 等目录重跑，旧日志不删除。参数、代码、分区文件大小/修改时间、依赖版本变化会拒绝混用旧数据。外部引用的数据文件内容变化需要自行使用新的输出目录。

## 校准规则与边界

- 配额范围默认 `[0.05, 1.0]`，步长 `0.001`，即 0.1 个百分点；测量扫描范围仍是 30% 到 90%。低于 30% 或高于 90% 的初始预测会标为外推，并必须通过实际验证。
- 修正时使用当前实测值相对 CPU 模型预测的比例修正，并用历史实测的上下界约束更新方向。偏慢增加 CPU，偏快减少 CPU。不会通过改 epochs、batch size、丢弃慢样本或放宽 target 来制造达标结果。
- 默认最多 5 批验证（代码默认值和示例 JSON 均为 5）。达到上限仍未全部达标时，将最后一批实际使用的配额导出为 `final_cpu_config.csv`，不再计算一组未验证的新配额。CSV 保留每个客户端的 `relative_error`、`passed`，并设置全局 `converged=false`、`export_reason=max_iterations`；`status.json` 为 `max_iterations_reached`，进程退出码为 2，表示文件已生成但校准未全部达标。生成的启动脚本允许使用此配置，但会打印警告。可增加 `max_iterations` 后恢复；也允许恢复时增加启动/阶段超时，但不得改变工作负载、target 定义、误差阈值或 CPU 边界。训练进程失败、原始数据不完整或撞到资源边界导致提前中止，仍以退出码 1 失败，不自动导出。
- `validation_discard` 默认 1；如明确要保留验证的全部 30 轮，可设为 0。每次都会重启进程，保留首轮可能包含图编译/冷启动开销。
- CPU 模型使用通用单调幂律拟合，`beta` 工程边界是 `[0.05, 5]`，不强制论文更窄的指数假设。常数项命名为 `theta_floor_s`，与 Pacer 分布幂指数 `gamma` 无关。
- Logistic 拟合使用 SciPy 数值最大似然估计，`k_s` 为尺度参数、单位秒。`ks_distance` 和经验 P90 仅作拟合诊断。3% 判断针对 θ 的点估计，不是统计置信区间，也不是在线 SLO 概率区间。
- 同步屏障等待全部客户端完成数据/模型初始化后才开始本批。各客户端在每轮训练前后读取实际 cgroup 配额和 CPU 亲和性；不一致立即失败。计时来自现有 `FlowerClient.fit` 的 `train_time`，不包含服务器等待和网络传输。
- 每批固定客户端随机种子并重新初始化模型，因此该脚本校准的是固定本地训练工作负载。正式联邦训练中的模型状态、其他系统负载和数据访问变化可能引起漂移，不能把一次校准当成永久保证。

## 输出

`output_dir` 中主要文件：

所有原始计时、训练日志和进程日志都保留在 `stages/<stage>/attempt_<n>/`，完成后不会删除。每个完整阶段另导出 `timing_exports/scan_30.csv`、`scan_50.csv`、`scan_70.csv`、`scan_90.csv` 或 `validate_001.csv` 等汇总，`timing_exports/index.json` 列出来源。汇总保留全部轮次，以 `used_for_fit` 标识是否参与拟合，不从原始文件删除首轮。

旧实验可在不训练、不改原始数据的情况下补导出这些汇总：

```bash
/home/xiaoyan/wholeflower/venv/bin/python /home/xiaoyan/wholeflower/femnist/warmup_logs.py \
  /home/xiaoyan/wholeflower/femnist/logs/mnistdata/warmup_femnist20_run01
```

| 文件 | 内容 |
| --- | --- |
| `manifest.json` / `latest_config.json` | 配置、客户端/CPU 对应关系、分区元数据、代码及依赖指纹 |
| `profiles.csv` | 每个客户端四档 CPU 的 Logistic θ、k、样本数、拟合诊断 |
| `cpu_models.json` | 各客户端的 μ、β、时间下界和相对拟合 RMSE |
| `target.json` | 固定 target 和 90% CPU 下的最慢客户端 |
| `initial_cpu_config.csv` | 模型逆推的待验证配置，不是最终配置 |
| `validation_history.csv` | 每批、每客户端的 CPU、实测 θ、相对误差和达标状态 |
| `stages/*/attempt_*/client_<cid>.jsonl` | 未四舍五入的逐轮原始计时和实际资源读回 |
| `stages/validate_*/adjustments.json` | 仅列出本批需要修改的客户端及新旧 CPU |
| `status.json` | running / validating / converged / max_iterations_reached / exported_unconverged / failed |
| `final_cpu_config.csv` | 真实测量全员达标或达到验证上限时生成，可由原 launcher 读取；必须结合 passed、converged 和误差列判断质量 |
| `launch_final_clients.sh` | 仅生成、不执行，保留校准时的工作负载、CPU 配额及核绑定 |

模拟模式只生成 `simulated_cpu_config.csv`，不生成真实最终 CSV 或启动脚本。不得把模拟数字用于实验结论。

## 已完成实验补导出

旧版本跑满 20 批仍未收敛、没有最终 CSV 时，不需要重新 warm-up，也不要为导出而使用 `--resume`。执行：

```bash
/home/xiaoyan/wholeflower/venv/bin/python /home/xiaoyan/wholeflower/femnist/warmup_control.py \
  --export-last /home/xiaoyan/wholeflower/femnist/logs/mnistdata/warmup_femnist20_run01
```

该命令读取原实验保存的配置，校验最后一个完整验证批次的 CPU、全部客户端原始计时、拟合结果和验证历史；不启动服务、不重新训练、不重新拟合、不修改 target。导出 `final_cpu_config.csv` 和 `launch_final_clients.sh`，保留原始文件，原状态备份为 `status.before_export.json`，导出来源记录在 `last_export.json`。已完成旧实验允许使用新版本导出工具，但训练恢复仍保留原来的代码指纹校验。退出码 0 表示全员达标，2 表示已导出但仍有客户端超差，1 表示导出失败。

之后正式训练时，先自行启动参数匹配的 Flower server，再执行输出目录中的：

```bash
bash /absolute/output_dir/launch_final_clients.sh HOST:PORT
```

该脚本只启动本次校准过的客户端，设置 `CPU_MAP_ONLY=1`、`CPU_ONLY=1`，并调用现有 `launch_clients.sh`。普通启动逻辑仍可使用旧 `new_cpu` 列；新结果使用 `cpu` 列，单位均为 0 到 1 的比例。已修正旧 launcher 中计算了配额却仍写死 `CPUQuota=100%` 的问题，新增可选 `cpu_affinity` 列。

## 正式训练审计与核隔离

2026-09-13 起，正式客户端 launcher 每次新建 `femnist/logs/clients_<日期>_<时间>_<随机后缀>/`，打印 `Run logs` 路径，保存每个客户端的完整 JSONL、进程日志、训练日志及 CPU 配置。也可用 `CLIENT_LOG_DIR` 指定一个尚不存在的绝对路径。旧 `data_partitions/client_*.log` 保留，不再混写本轮日志。

校准后的 launcher 要求每个客户端绑定不同的单个逻辑 CPU。客户端启动时及每轮 fit 前后检查实际 `cpu.max` 和所有存活线程的亲和性，不一致即失败。正式 server CSV 新增 `partition_id`（真实 0-19 编号）、`training_run_id`、实际配额、绑核、cgroup 使用时间和限流统计。旧 `client_id` 连接地址列仍保留。按 `partition_id` 分组比较，不要按端口顺序分组，也不要用 `num_examples` 充当唯一身份，样本数可能重复。

原有 `cpu_freq_start_mhz/cpu_freq_end_mhz` 是全机平均频率，保留供旧分析使用；新增 `bound_cpu_freq_mhz` 和 JSONL 中的 `bound_cpu_frequencies_mhz` 才对应允许使用的核。`sched_runqueue_time_s` 是主线程统计，不等于整个进程被限流的时间。cgroup 增量统计覆盖 fit 处理前后的区间，包含资源探测、设置权重等开销，不能当成纯 `model.fit` CPU 时间。

频率探测和日志输出已移出训练计时区间，新记录用 `timing_definition=model_fit_v2` 标记；FEMNIST 计时为 `model.fit`，流式数据集还包含其数据集构造。旧日志未改写，汇总标为 `legacy_v1`。新模型应使用新 warm-up，不能直接把旧 target 当作新计时定义的已验证结果。本次 FEMNIST 配置输出改为 `warmup_femnist20_run02`，旧 run01 保留。

单核亲和性不是独占核。4a6 当前有 32 个物理核、64 个逻辑 CPU，CPU i 与 i+32 共享物理核。客户端使用 0-19 时，server 应同时避开 0-19 和 32-51。例如：

```bash
cd /home/xiaoyan/wholeflower/femnist
CUDA_VISIBLE_DEVICES= taskset -c 20-31,52-63 ../venv/bin/python -m run_server \
  --dataset femnist --model cnn --clients 20 --rounds 50 --reporting-fraction 1.0 \
  --address 0.0.0.0:8081 --client-lr 0.003 --batch-size 64 --local-epochs 1 \
  --downlink-num-bits 0 --csv-path logs/mnistdata/calibrated_run02.csv
```

上述核列表仅对应 4a6 当前配置；更换客户端核后要重新计算。这只隔开本实验 server/client，不会禁止其他用户或系统进程进入这些核，也不会固定 CPU 频率。未修改系统隔离、频率策略或其他用户进程。独立本地 warm-up 不包含正式联邦训练的 server/gRPC 等负载，不能保证正式训练仍处于原有 3% 范围；需要结合新日志做同条件验证。

不要向旧 `calibrated.csv` 追加新字段，新 server 会拒绝不兼容的已有表头。使用新的 `--csv-path`，旧数据完整保留。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/xiaoyan/wholeflower/femnist \
  /home/xiaoyan/wholeflower/venv/bin/python -m unittest discover \
  -s /home/xiaoyan/wholeflower/femnist/tests -p 'test_*.py' -v
```

测试使用合成样本和模拟训练接口，覆盖拟合、选择性调整、固定 target、49/29 样本计数、恢复、资源检查、客户端 JSONL 入口、启动失败清理和最终导出边界；不运行真实模型训练。真实 CPU 隔离效果、数据集耗时和收敛轮数需要后续实际实验验证。
