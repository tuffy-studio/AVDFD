import torch
import matplotlib.pyplot as plt
import seaborn as sns

batch_size = 48
features = 768

# 随机初始化特征矩阵
audio_features = torch.randn(batch_size, features, requires_grad=True)  # 可训练
video_features = torch.randn(batch_size, features)

# 优化器
optimizer = torch.optim.Adam([audio_features], lr=0.01)

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 计算相似性损失（以欧几里得距离为例）
    loss = torch.nn.functional.mse_loss(audio_features, video_features)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# 查看最终结果
print("Optimization completed!")


audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
video_features = torch.nn.functional.normalize(video_features, dim=-1)
cl_matrix = torch.mm(audio_features, torch.transpose(video_features, 0, 1))
softmax_cl_matrix = torch.nn.functional.log_softmax(cl_matrix, dim=0)
print(softmax_cl_matrix.shape)
cl_loss = -torch.mean( torch.diag( softmax_cl_matrix ) )
print(cl_loss)

# 将张量转换为 NumPy 数组，便于可视化
cl_matrix_np = cl_matrix.detach().numpy()

# 生成从 1 开始的标签
x_labels = list(range(1, cl_matrix_np.shape[1] + 1))
y_labels = list(range(1, cl_matrix_np.shape[0] + 1))

# 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cl_matrix_np, cmap="viridis", cbar=True, xticklabels=x_labels, yticklabels=y_labels)
plt.title("Contrastive learning loss Matrix")
plt.xlabel("Video Features")
plt.ylabel("Audio Features")
plt.show()
plt.savefig("./utils_data/l_matrix_heatmap.png", dpi=300)  # 保存为 PNG 格式，dpi 设置为 300 提高分辨率
plt.close()  # 关闭图形，释放资源

