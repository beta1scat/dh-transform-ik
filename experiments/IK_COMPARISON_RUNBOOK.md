# 第2章逆运动学重新实验运行说明

## 实验协议

本次重新计算解析IK与三种数值IK方法的回代误差、失败次数和目标级总耗时，生成两张替换表。所有方法使用同一批目标、原始MDH正运动学、实际关节限位和毫米单位下的矩阵平均绝对误差阈值 `ME <= 1e-3`。目标由正运动学生成，因此评价范围为已知可达目标上的求解表现及计算开销。

数值法每个目标最多尝试10个初值，首个为关节范围中点，其余9个在实际限位内均匀采样；三种数值方法共享相同初值列表。目标种子为42，初值种子为142，目标生成关节角不作为初值。找到首个通过统一判据的解后停止，否则记录失败。解析法计算并验收8个分支，以有效分支中的最小回代误差解汇总。

成功判据要求解为有限值、存在满足关节限位的周期等价解且满足回代容差。周期处理仅允许加减整周，不截断关节角。工具箱的 `success` 状态独立保存，不额外作为数值法的成功门槛。新协议与旧默认单次调用协议不同，两张表及相应正文应一起更新，不能沿用旧表数字。

计时包含求解、数值重启、周期处理和回代验收；所有成功及失败目标均纳入平均耗时。模型构建、预处理、初值生成、目标格式转换和文件写入不计时。每种机器人、每种方法先进行100次不计入统计的单次调用预热。正式测量在同一进程中完成。

## 1 进入源码目录

以下命令均使用uv，固定Python 3.10及依赖版本。不需要手动激活虚拟环境，也不要同时运行其他高负载任务。

```bash
cd /Users/niu/Documents/papers/graduate_thesis/code/dh-transform-ik
```

## 2 一次生成并保留公共目标集

四种机器人各10,000个目标，共40,000个。仅生成一次，后续预运行和正式实验读取同一文件。

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.generate_ik_targets --trials 10000 --seed 42 --output data/chapter2/ik_recompute_20261028/ik_targets.csv
```

## 3 预运行

每种机器人取前200个目标，用于检查依赖接口、输出和运行耗时，不并入正式结果，也不据此挑选初值种子。程序会先核对工具箱与原始MDH链的正运动学和单位是否一致。

```bash
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.run_ik_comparison --input data/chapter2/ik_recompute_20261028/ik_targets.csv --output-dir data/chapter2/ik_recompute_20261028/pilot_s142 --trials 200 --attempts 10 --seed 142 --warmup 100
```

## 4 正式重新实验

```bash
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.run_ik_comparison --input data/chapter2/ik_recompute_20261028/ik_targets.csv --output-dir data/chapter2/ik_recompute_20261028/s142 --trials 10000 --attempts 10 --seed 142 --warmup 100
```

输出目录必须尚不存在。程序每完成100个目标输出一次进度。中断运行会标记为未完成；当前不支持断点续跑，重新运行应使用新的输出目录。不要将未完成目录与完整运行结果合并。

## 5 导出两张论文表格

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.export_ik_comparison_tables --runs data/chapter2/ik_recompute_20261028/s142 --output-dir data/chapter2/ik_recompute_20261028/tables_s142
```

导出文件：

- `ik_accuracy_table.tex`：成功样本的平均位置误差、旋转误差及目标失败次数，保留原标签 `tab:ik_accuracy`。
- `ik_timing_table.tex`：全部目标的平均总耗时，保留原标签 `tab:ik_efficiency`。
- `table_values.csv`：可查看的数值汇总。
- `table_manifest.json`：来源、协议和输入文件校验信息。

正式运行应有16行方法汇总，即4种机器人乘4种方法；每行成功数加失败数应为10,000。`targets.csv` 应有160,000条数据记录。失败次数和精度以实际输出为准，不预设解析方法全部成功。

将两份LaTeX表格替换正文中原有的完整表格环境，避免重复标签。相邻实验协议、耗时定义和比较结论也需要依据新结果更新。Mac上的200目标预运行已经完成，正式10,000目标结果尚未生成，论文原有数值未替换。

## 6 可选的三个初值种子重复

主实验142之外，可在相同电脑、相同依赖和相同源码下再运行143与144，保持目标文件及其他参数不变。

```bash
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.run_ik_comparison --input data/chapter2/ik_recompute_20261028/ik_targets.csv --output-dir data/chapter2/ik_recompute_20261028/s143 --trials 10000 --attempts 10 --seed 143 --warmup 100
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.run_ik_comparison --input data/chapter2/ik_recompute_20261028/ik_targets.csv --output-dir data/chapter2/ik_recompute_20261028/s144 --trials 10000 --attempts 10 --seed 144 --warmup 100
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.export_ik_comparison_tables --runs data/chapter2/ik_recompute_20261028/s142 data/chapter2/ik_recompute_20261028/s143 data/chapter2/ik_recompute_20261028/s144 --output-dir data/chapter2/ik_recompute_20261028/tables_3seeds
```

多种子表中误差为各次运行成功样本均值的平均，失败次数为各次运行失败数的均值及样本标准差，耗时为各次运行目标平均耗时的平均。三次使用相同目标，不能宣称每种机器人有30,000个独立目标。单种子表与三种子表选择其中一套用于论文。

## 原始记录

| 文件 | 内容 |
|---|---|
| `manifest.json` | 完成状态、依赖版本、源码和目标校验值、初值规则、计时口径与工具箱接口默认参数 |
| `initial_guesses.csv` | 每个目标预生成的全部数值初值 |
| `attempts.csv` | 实际数值尝试及解析分支的验收结果、终止状态、误差与数值尝试耗时 |
| `targets.csv` | 每个目标和方法的最终结果、实际尝试数与总耗时 |
| `summary.csv` | 成功率、失败数、成功样本误差、平均耗时、中位数和P95等 |

如需分析失败原因，可区分关节限位、回代容差、非有限解及求解异常；不能仅依据工具箱状态解释失败。求解异常的原始原因保留于逐次记录。输出目录完整保留用于复核。

## Windows平台迁移与GitHub提交

本项目源码仓库为 `code/dh-transform-ik/`，拥有独立的 `.git`。在该目录提交源码修改，包含新增脚本、删除的旧入口、README和数据说明。生成数据、实验输出、虚拟环境及缓存由 `.gitignore` 排除。不要将Mac的 `.venv` 复制到Windows。

Windows使用64位Python 3.10，由uv选择或安装。PowerShell进入克隆后的 `dh-transform-ik` 目录后，前述单行 `uv run` 命令可以直接使用，路径中的 `/` 无需改写。不要使用Bash的反斜杠换行语法，也不要添加 `--group benchmark`，本次实验依赖以命令中的固定版本为准。

目标文件有两种准备方式，选择一种：

1. 从Mac复制 `data/chapter2/ik_recompute_20261028/ik_targets.csv` 到Windows相同相对路径，保持原始文件字节不变。
2. 使用第2步命令重新生成目标。固定种子和NumPy版本可复现生成过程；正式实验记录输入SHA256，以实际校验值确认目标文件一致。

现有Mac目标文件SHA256为：

```text
d4859200cc8d751aaa2149f59b73f5f5dcbe46daf7928bf9b4253e6680fe1f89
```

Windows可在PowerShell中查看复制或重新生成文件的SHA256：

```powershell
Get-FileHash data/chapter2/ik_recompute_20261028/ik_targets.csv -Algorithm SHA256
```

Windows结果使用独立目录，推荐先执行同协议预运行：

```powershell
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.run_ik_comparison --input data/chapter2/ik_recompute_20261028/ik_targets.csv --output-dir data/chapter2/ik_recompute_20261028/windows_pilot_s142 --trials 200 --attempts 10 --seed 142 --warmup 100
```

正式运行及表格导出：

```powershell
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.run_ik_comparison --input data/chapter2/ik_recompute_20261028/ik_targets.csv --output-dir data/chapter2/ik_recompute_20261028/windows_s142 --trials 10000 --attempts 10 --seed 142 --warmup 100
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.export_ik_comparison_tables --runs data/chapter2/ik_recompute_20261028/windows_s142 --output-dir data/chapter2/ik_recompute_20261028/windows_tables_s142
```

依赖安装若报告没有可用wheel或需要编译器，应保留完整报错再处理，不要直接升级RTB或SciPy，否则实验环境发生变化。

正式实验的四种方法须全部在同一平台运行。Mac与Windows的耗时分别报告，不合并导出。程序记录Python与依赖版本、平台、CPU信息、线程环境、源码SHA256及Git工作区状态。多种子重复应使用同一台机器，保持线程环境及源码一致。

全部结果保存在忽略目录中，需要另外复制或归档；GitHub仅推送代码不会同步实验结果。Mac预运行记录保持原样，其中源码校验信息反映当时实际运行版本，不能改写成整理后的版本。
