# 第2章运动学实验运行说明

## 实验范围

实验包括D-H参数转换一致性、IK有效性与效率比较、腕部奇异性测试，以及直线和圆弧目标序列分析。D-H一致性与IK比较在完整关节限位区间内随机采样；腕部奇异性采用定向构造的目标；路径实验采用固定姿态的随机直线和圆弧。各实验的采样范围、验收条件和输出文件分别说明如下。

## D-H一致性复现

五种机器人各10,000组关节角，覆盖完整关节限位范围，输出`data/thesis_chapter2/processed/dh_validation.csv`与
`dh_validation_summary.csv`。

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.thesis_chapter2.run_dh_validation --trials 10000 --seed 42 --margin 0
```

`run_d_offset_validation.py`用于D-H一致性实验中的单级非零偏移传播检验。

## IK比较与奇异点处理

解析求解器以 `|sin(theta5)| <= 1e-10` 判定数值退化的腕部。
非奇异邻域继续使用解析姿态分解；退化处的 q4/q6 耦合解在目标机器人实际关节限位内联合选择。`reference_q` 使用目标机器人关节坐标、弧度单位，仅影响退化自由度的选择，不替代八分支枚举或统一回代验收。

IK主比较在完整关节限位区间内生成公共目标集，不内缩，显式设置
`--margin 0`。四种机器人各10,000个目标，目标种子42、初值种子142。
解析方法不使用目标生成关节角，也不向独立随机目标传入上一目标的解。
各方法使用相同目标集、源码版本和运行环境。

DH采样和IK目标生成的默认内缩值均为0，命令中仍显式传入0以记录协议。
四种方法读取同一目标文件，结果保存至下列命令指定的目录。

在仓库根目录依次执行：

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.thesis_chapter2.generate_ik_targets --trials 10000 --seed 42 --margin 0 --output data/thesis_chapter2/ik_full_limits_update/ik_targets.csv
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.thesis_chapter2.run_ik_comparison --input data/thesis_chapter2/ik_full_limits_update/ik_targets.csv --output-dir data/thesis_chapter2/ik_full_limits_update/s142 --trials 10000 --attempts 10 --seed 142 --warmup 100 --me-tolerance 0.001
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.thesis_chapter2.export_ik_comparison_tables --runs data/thesis_chapter2/ik_full_limits_update/s142 --output-dir data/thesis_chapter2/ik_full_limits_update/tables_s142
```

腕部专项实验使用两种模式。`nominal` 不传入参考构型；`previous` 首点使用限位中点，此后使用前一个目标的有效解，不使用当前目标的真值关节构型。
两种模式生成相同目标，输出到不同文件：

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.thesis_chapter2.run_singularity --seed 42 --reference-mode nominal --trials-output data/thesis_chapter2/ik_full_limits_update/singularity_nominal_trials.csv --summary-output data/thesis_chapter2/ik_full_limits_update/singularity_nominal_summary.csv
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.thesis_chapter2.run_singularity --seed 42 --reference-mode previous --trials-output data/thesis_chapter2/ik_full_limits_update/singularity_previous_trials.csv --summary-output data/thesis_chapter2/ik_full_limits_update/singularity_previous_summary.csv
```

优先检查 `effective_branches`，其同时要求有限性、周期等价解满足限位和回代误差通过。`pose_passed_branches` 单独反映姿态回代判据。各分支槽位可能在奇异点处合并，分支槽位数量不等同于互不相同的解数量。关节变化指标用于补充分析，不能据此认定已经获得连续可执行的轨迹。

## 直线和圆弧目标序列

参考构型从关节限位两端各内缩15%的
范围采样，且令第五关节绝对值至少为20度。通过正运动学得到参考位姿，
再随机选取直线方向或圆弧平面。直线长80 mm，圆弧半径40 mm、圆心角120度，末端姿态保持不变。参考位姿位于几何路径中部附近，不保证整个路径可达。

所有生成路径均纳入评价，不重采样或预筛除失败路径。每个位姿记录各解析分支的有限性、周期等价解限位检查及回代验收结果。参考构型用于首点的奇异自由度选择，之后使用最近一次有效求解结果。分支变化仅作补充统计。

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.thesis_chapter2.run_path_continuity --seed 42 --line-paths 10 --arc-paths 10 --points 100 --line-length-mm 80 --arc-radius-mm 40 --arc-angle-deg 120 --trials-output data/thesis_chapter2/ik_full_limits_update/path_trials.csv --summary-output data/thesis_chapter2/ik_full_limits_update/path_summary.csv --branches-output data/thesis_chapter2/ik_full_limits_update/path_branches.csv
```

输出：

- `path_trials.csv`：逐点目标位置、旋转、有效候选数量、选定解及失败分类。
- `path_branches.csv`：全部分支的验收结果和关节角，用于诊断失败。
- `path_summary.csv`：路径成功点数、完整性和分支变化。

失败分类：

- `outside_arm_triangle_reach`：目标变换到模板坐标后，前、后肩部几何均违反
  前三关节臂段三角形的可达距离条件，且越界超过1e-7 mm。
- `no_finite_candidate_requires_diagnosis`：没有有限候选，尚不能确定不可达。
- `no_limit_valid_enumerated_candidate`：枚举的有限候选均未通过关节限位验收。
- `pose_tolerance`：存在有限且限位有效的候选，但未通过回代误差阈值。

正的臂段可达裕量只表示通过必要位置几何条件，不能据此认定给定姿态及关节限位下可达。失败归因以逐点和逐分支记录为依据，结论适用于所测试的目标序列。

## 统一验收与计时

主实验包含四种机器人、每种10,000个目标和四种方法。数值法最多使用10个共享初值（限位中点和9个均匀随机初值）；找到首个有效解即停止。
统一验收要求有限解、存在实际限位内的周期等价解、毫米单位下
`ME <= 1e-3`；工具箱 success 字段仅作诊断记录。周期处理不截断关节角。
计时包含求解、重启、周期等价处理和原始MDH回代验收。初值生成、模型构建及文件写入不计时。每种方法、每种机器人预热100次。

输出目录必须尚不存在。正式运行应产生16行汇总和160,000条目标级记录，每行汇总的成功数与失败数之和为10,000。主比较不支持断点续跑。
失败数按实测记录，不预设解析方法100%成功。所有比较须在同一电脑和环境完成，运行期间避免其他高负载工作。

导出表格为 `tables_s142/ik_accuracy_table.tex` 和
`tables_s142/ik_timing_table.tex`，另有 `table_values.csv` 与来源清单。
论文中的实验设置、统计结果和表格应与对应运行记录一致。

## 回归检查

执行以下命令检查奇异腕部耦合解及参考构型映射：

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m unittest discover -s tests -p "test_singular_reference.py" -v
```

## 论文绘图

绘图脚本采用Windows字体文件路径。使用同一组实验数据，并在路径实验完成后绘图；文件默认输出至论文`figures/chapter2/`。以下命令显式补充绘图依赖：

```bash
uv run --python 3.10 --with "numpy==1.23.5" --with "matplotlib==3.7.5" python -m experiments.thesis_chapter2.plot_workspace_samples --input data/thesis_chapter2/ik_full_limits_update/ik_targets.csv
uv run --python 3.10 --with "numpy==1.23.5" --with "matplotlib==3.7.5" python -m experiments.thesis_chapter2.plot_path_results --input data/thesis_chapter2/ik_full_limits_update/path_trials.csv
```
