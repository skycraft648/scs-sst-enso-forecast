# 把项目放到 GitHub（Windows 浏览器操作）

## 第一步：解压

下载 `scs-sst-enso-reconstructed.zip`，**先解压**。不要直接把 ZIP 作为唯一文件上传到仓库，否则 GitHub 无法逐个展示项目代码。

## 第二步：创建仓库

1. 登录 https://github.com/new
2. Repository name：`scs-sst-enso-forecast`
3. Description：`Reconstructed code for South China Sea SST prediction using ENSO features and six models`
4. 选择 **Public**（希望给招聘方展示时使用；不想公开可先选 Private）
5. **不要勾选**创建 README、.gitignore 或 License，因为包内已有前两者。
6. 点击 **Create repository**。

## 第三步：上传完整文件结构

最简单的方法是使用 **GitHub Desktop**（https://desktop.github.com/）：

1. 选择 **File → Add local repository**，选解压后的文件夹；如果不是 Git 仓库，按提示创建仓库。
2. 检查 Changes 列表，**确保没有任何原始 SST NetCDF、处理后的数据、demo_synthetic 等**。
3. Commit summary 写 `Reconstruct SCS SST forecasting pipeline`，点击 **Commit to main**。
4. 点击 **Publish repository**，选择 Public 或 Private，然后发布。

也可以使用网页上传：在新仓库点击 **uploading an existing file**，拖入解压文件夹内的 README、src、docs、tests 等内容，确认目录层级正确后点击 **Commit changes**。个别浏览器不保留空文件夹；这不影响运行。

> **不要把整个 ZIP 当成唯一文件上传。** 项目首页应该直接显示 `README.md`、`src/`、`docs/`、`tests/` 等。

## 第四步：上传成功后检查

仓库首页能自动显示 README，点击 `src/sst_project/models.py` 能看到模型代码，点击 `docs/archived_paper_results.md` 可看到论文中的历史成绩，并明确标注 **not reproduced**。

简历可以写：

**南海 SST 与 ENSO 机器学习预测（研究代码重建）**：根据既有科研论文重建数据预处理、多模型预测和特征解释流程，涵盖 SARIMAX、SVR、随机森林、XGBoost、LSTM；整理时间序列划分和指标评估脚本，GitHub 链接：`https://github.com/你的用户名/scs-sst-enso-forecast`。

简历上的链接必须换成实际仓库地址；没有用真实 SST 数据重新运行之前，不应写“已复现 RMSE=0.303°C”。
