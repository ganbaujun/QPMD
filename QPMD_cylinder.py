from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from pydmd import DMD
from torch_geometric.data import Data
from pydiffmap import diffusion_map as dm
import pennylane as qml
from pennylane import numpy as qnp
from tqdm import tqdm
import pandas as pd

from Diffusion import *
from quantum_dkl import *
from quantum_svd import *

#构建qsvd分解
class QSVD:
    def __init__(self, matrix):
        """
        初始化量子奇异值分解器
        :param matrix: 待分解的对称矩阵 (n x n)
        """
        # 验证输入矩阵的对称性
        assert np.allclose(matrix, matrix.T), "Input matrix must be symmetric"
        self.matrix = qnp.array(matrix, requires_grad=False)
        self.n = matrix.shape[0]
        self.num_qubits = int(np.log2(self.n))
        self.params = None
        self.evals = None
        self.evecs = None
        
        # 设置Pennylane设备
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        
        # 定义量子节点
        @qml.qnode(self.dev, interface="torch")#, diff_method="parameter-shift")
        def circuit(theta, state):
            self.quantum_circuit(theta, state)
            return qml.expval(qml.Hermitian(self.matrix, wires=range(self.num_qubits))), qml.state()
        
        self.circuit = circuit
        
    def quantum_circuit(self, theta, state):
        """
        参数化量子电路
        :param theta: 可训练参数 [n_layers, n_qubits]
        :param state: 输入基态
        """
        # 输入态制备
        qml.BasisState(state, wires=range(self.num_qubits))
        
        # 将参数重塑为层结构
        theta = theta.reshape(-1, self.num_qubits)
        n_layers = theta.shape[0]
        
        # 交替旋转层和纠缠层
        for layer in range(n_layers):
            # 单量子比特旋转
            for i in range(self.num_qubits):
                qml.RY(theta[layer, i], wires=i)
            
            # 线性纠缠层
            for i in range(self.num_qubits-1):
                qml.CNOT(wires=[i, i+1])
    
    def initialize_params(self, n_layers=3):
        """
        初始化可训练参数
        :param n_layers: 量子电路层数
        """
        # 使用Xavier初始化
        scale = np.sqrt(2.0 / (self.num_qubits * n_layers))
        self.params = torch.nn.Parameter(
            torch.randn(n_layers, self.num_qubits) * scale,
            requires_grad=True
        )
    
    def loss_function(self):
        """
        计算损失函数：加权特征值的负和
        """
        M_rec = torch.zeros([self.n,self.n])
        M = torch.tensor(self.matrix, dtype=torch.float32)
        
        for i in range(self.n):
            # 生成二进制基态
            state = qnp.array([int(b) for b in f"{i:0{self.num_qubits}b}"])
            # 计算期望值
            eval, evec = self.circuit(self.params, state)
            M_rec = M_rec + eval * evec.reshape(-1,1).conj() @ evec.reshape(1,-1)
        loss = torch.norm(M - M_rec)#, p = 2)
        return loss  # 归一化损失

    def optimize(self, iterations=200, lr=0.1):
        """
        使用PyTorch优化器进行训练
        """
        optimizer = torch.optim.Adam([self.params], lr=lr)
        loss_history = []
        progress = tqdm(range(iterations))
        for epoch in progress:
            optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            progress.set_description(f"Loss {loss}")

        # 获取最终分解结果
        self.post_process()
        return loss_history
    
    def post_process(self):
        """
        后处理获取特征值和特征向量
        """
        with torch.no_grad():
            # 计算所有基态的期望值
            evals = qnp.zeros(self.n)
            evecs = qnp.zeros((self.n, self.n))
            
            for i in range(self.n):
                state = qnp.array([int(b) for b in f"{i:0{self.num_qubits}b}"])
                output = self.circuit(self.params, state)
                evals[i] = output[0]
                evecs[:, i] = output[1].real.numpy()
            
            # 排序特征值
            sorted_indices = np.argsort(evals)[::-1]
            self.evals = evals[sorted_indices]
            self.evecs = evecs[:, sorted_indices]
    
    def get_state_vector(self, state):
        """
        获取量子态向量
        """
        @qml.qnode(self.dev, interface="torch")
        def state_circuit():
            #qml.BasisState(state, wires=range(self.num_qubits))
            self.quantum_circuit(self.params, state)
            return qml.state()
        
        return state_circuit().numpy()


# 获取数据
matrix = get_dataset("C:\\Users\\ASUS\\Desktop\\flow_past_cylinder_Re4000_200")

vtu_num = get_vtu_num("C:\\Users\\ASUS\\Desktop\\flow_past_cylinder_Re4000_200")

vtu_num_  = 2**int(np.log2(vtu_num)) + 20
print(vtu_num_)
snapshot_u = matrix[:, 0::2]
snapshot_v = matrix[:, 1::2]
# 步骤1: 对数据进行归一化
scaler_u = StandardScaler()
scaler_v = StandardScaler()

normalized_u = scaler_u.fit_transform(snapshot_u)
normalized_v = scaler_v.fit_transform(snapshot_v)

# 获取均值和标准差
mean_u = scaler_u.mean_
std_u = scaler_u.scale_

mean_v = scaler_v.mean_
std_v = scaler_v.scale_

# 步骤2: 创建PCA对象，设置降维到12维
pca_u = PCA(n_components=6)
pca_v = PCA(n_components=13)

# 步骤3: 对归一化数据进行PCA降维
reduced_u = pca_u.fit_transform(normalized_u.T)
reduced_v = pca_v.fit_transform(normalized_v.T)


# 步骤4: 从低维矩阵进行逆变换
inverse_u = pca_u.inverse_transform(reduced_u)
inverse_v = pca_v.inverse_transform(reduced_v)

# 步骤5: 逆变换后进行反归一化
original_u = scaler_u.inverse_transform(inverse_u.T)
original_v = scaler_v.inverse_transform(inverse_v.T)

delta_u = snapshot_u - original_u
delta_v = snapshot_v - original_v

# Step 1: 使用 Isomap 计算测地距离矩阵
def compute_geodesic_distances(data, n_neighbors):
    """
    使用 Isomap 计算数据的测地距离矩阵
    :param data: 输入数据矩阵
    :param n_neighbors: 最近邻个数
    :return: 测地距离矩阵
    """
    isomap = Isomap(n_neighbors=n_neighbors, n_components=15, path_method='auto', neighbors_algorithm='auto')
    isomap.fit(data)
    geodesic_dist_matrix = isomap.dist_matrix_
    return geodesic_dist_matrix

# 计算 snapshot_u 和 snapshot_v 的测地距离矩阵
n_neighbors = 15  # 设置最近邻个数
geodesic_dist_u = compute_geodesic_distances(delta_u.T, n_neighbors)
geodesic_dist_v = compute_geodesic_distances(delta_v.T, n_neighbors)

# Step 2: 自定义度量函数
def geodesic_metric_u(x, y):
    """
    计算 x 和 y 在测地距离矩阵中的距离
    :param x: 数据点 x
    :param y: 数据点 y
    :return: x 和 y 的测地距离
    """
    # 找到 x 和 y 对应的索引
    idx_x = np.argmin(np.linalg.norm(delta_u.T - x, axis=1))
    idx_y = np.argmin(np.linalg.norm(delta_u.T - y, axis=1))
    return geodesic_dist_u[idx_x, idx_y]

def geodesic_metric_v(x, y):
    """
    计算 x 和 y 在测地距离矩阵中的距离
    :param x: 数据点 x
    :param y: 数据点 y
    :return: x 和 y 的测地距离
    """
    idx_x = np.argmin(np.linalg.norm(delta_v.T - x, axis=1))
    idx_y = np.argmin(np.linalg.norm(delta_v.T - y, axis=1))
    return geodesic_dist_v[idx_x, idx_y]

# Step 3: 使用自定义度量进行 Dmap 扩散映射
n_evecs = 6
neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
mydmap_u = dm.DiffusionMap.from_sklearn(n_evecs=n_evecs, k = 10, epsilon=55, alpha=1, neighbor_params=neighbor_params, metric=geodesic_metric_u)#16(14.6) 21(14.9)
mydmap_v = dm.DiffusionMap.from_sklearn(n_evecs=n_evecs, k = 10, epsilon=10, alpha=1, neighbor_params=neighbor_params, metric=geodesic_metric_v)

# 使用 Dmap 拟合数据
dmap_u = mydmap_u.fit_transform(delta_u[:, 20:vtu_num_].T)
dmap_v = mydmap_v.fit_transform(delta_v[:, 20:vtu_num_].T)
print('得到测地矩阵')
# Step 6: 多项式特征转换和核岭回归
degree = 1  # 设置多项式的次数 3
poly_u = PolynomialFeatures(degree=degree)
poly_v = PolynomialFeatures(degree=degree)

# 对低维数据进行多项式特征转换
X_poly_u = poly_u.fit_transform(dmap_u)
X_poly_v = poly_v.fit_transform(dmap_v)

# 创建核岭回归模型
alpha = 0.01  # 正则化参数 0.05
kernel = 'poly'  # 核函数选择，这里选择多项式核
degree = 1  # 多项式的次数 1
coef0 = 100  # 多项式核的常数项 100
model1 = KernelRidge(alpha=alpha, kernel=kernel, degree=degree, coef0=coef0)
model2 = KernelRidge(alpha=alpha, kernel=kernel, degree=degree, coef0=coef0)

# 使用多项式特征矩阵拟合目标矩阵
model1.fit(X_poly_u, delta_u[:, 20:vtu_num_].T)
model2.fit(X_poly_v, delta_v[:, 20:vtu_num_].T)

# Ridge Regression to learn W
train_X_u = dmap_u[:-1, :]  # Training data
train_Y_u = dmap_u[1:, :]   # Target data
ridge_u = Ridge(alpha=0.1)
ridge_u.fit(train_X_u, train_Y_u)
W_u = ridge_u.coef_  # Learn the mapping matrix W

# Repeat for V
train_X_v = dmap_v[:-1, :]
train_Y_v = dmap_v[1:, :]
ridge_v = Ridge(alpha=0.1)
ridge_v.fit(train_X_v, train_Y_v)
W_v = ridge_v.coef_
print('得到系数矩阵C')

#将矩阵的每一列变成单位向量
def columns_to_unit_vectors(matrix):
    """
    将矩阵的每一列转换为单位向量(L2归一化)
    单位向量定义:列中各元素的平方和为1(L2范数=1)
    """
    # 计算每列的L2范数（axis=0表示按列计算）
    col_norms = np.linalg.norm(matrix, ord=2, axis=0)
    
    # 处理零向量列（范数为0时，避免除以0，保持原列不变）
    col_norms[col_norms == 0] = 1  # 零向量列除以1，仍为零向量
    
    # 每列除以自身的L2范数，得到单位向量
    unit_vector_matrix = matrix / col_norms
    
    return unit_vector_matrix


#构造量子核函数
#phi = mydmap_u.evecs
#epsilon_u = 0.3 #0.3
#K_u = kernel_matrix(phi,phi)
#构造核函数
phi = mydmap_u.evecs
epsilon_u = 0.1 #0.3
K_u = rbf_kernel(phi, gamma=1 / (epsilon_u ** 2))

def normalize_and_decompose(K):
    """
    Normalize kernel matrix and perform eigenvalue decomposition.
    """
    
    # 输入校验：必须是方阵
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K 需为方阵，当前形状: {K.shape}")
    
    # 1. 计算节点度（每行的和）
    degree = K.sum(axis=1)
    
    # 校验：行和不能为0（否则转移概率无意义）
    if np.any(degree == 0):
        zero_rows = np.where(degree == 0)[0]
        raise ValueError(f"发现 {len(zero_rows)} 行节点度为0(行索引：{zero_rows}),请检查K的有效性")
    
    # 2. 计算平稳分布 π（归一化的节点度）
    total_degree = degree.sum()
    pi = degree / total_degree  # 形状: [n,]
    
    # 3. 构造对角矩阵 D_√π 和其逆矩阵
    D_sqrtπ = np.diag(np.sqrt(pi))          # 对角元: √π[i]
    D_sqrtπ_inv = np.diag(1 / np.sqrt(pi))  # 对角元: 1/√π[i]
    
    # 4. 计算转移矩阵 P（行归一化：每行除以自身节点度）
    R = np.diag(1/degree)
    P = R @ K 
    
    # 5. 矩阵乘法计算 A = D_sqrtπ · P · D_sqrtπ_inv
    A = D_sqrtπ @ P @ D_sqrtπ_inv
    
    qsvd = QSVD(A)
    qsvd.initialize_params(n_layers=50)
    loss_hist = qsvd.optimize(iterations=100, lr=0.01)

    #params_tensor = qsvd.params.data  # 获取参数张量（不包含梯度信息）
    #params_array = params_tensor.numpy()  # 转换为numpy数组
    #pd.DataFrame(params_array).to_csv("qsvd_params_K_u.csv", index=False)
    #plt.plot(loss_hist)
    #plt.xlabel("Iteration")
    #plt.ylabel("Loss")
    #plt.title("Training Progress")
    #plt.show()
    #通过求解A的特征值和特征向量间接求解P的特征值和特征向量
    eigenvalues = qsvd.evals
    eigenvectors = qsvd.evecs
    
    #基变换求解P的特征向量
    eigenvectors = columns_to_unit_vectors(np.dot(D_sqrtπ_inv,eigenvectors))
     
    return eigenvalues, eigenvectors

sigma_u, V_u = normalize_and_decompose(K_u)

#print('特征值',sigma_u[1:13])
#print('特征向量',V_u[:,1:13])
#print('通过QVE得到特征值和对应的特征向量')

# Step 6: Projection of the target function
def project_target_function(W, V, sigma, phi):
    """
    Project the target function onto latent harmonics.

    Parameters:
    - W: The regression matrix (r x r).
    - V: The eigenvectors from the second diffusion map (m x r).
    - sigma: The eigenvalues from the second diffusion map.
    - phi: The low-dimensional representations (k x m), where m is the number of points.
    
    Returns:
    - c: The projection coefficients for each harmonic (r x r array).
    """
    r = W.shape[0]  # Number of rows in W
    m = V.shape[0]  # Number of data points
    c = np.zeros((r, len(sigma)))  # Initialize the coefficients matrix
    
    for j in range(len(sigma)):  # Loop over harmonics
        v_j = V[:, j]  # j-th eigenvector (length m)
        for i in range(r):  # Loop over rows of W
            # Compute <W_i, v_j> as sum_k (W_i * phi_k) * v_j(phi_k)
            c[i, j] = sum(np.dot(W[i, :], phi[:, k]) * v_j[k] for k in range(m))
    
    return c

# Call the function for U and V components
c_u = project_target_function(W_u, V_u[:,1:7], sigma_u[1:7], dmap_u.T)
print(c_u.shape)
#c_v = project_target_function(W_v, V_v, sigma_v, dmap_v.T)

# Step 7: Compute latent harmonics for a new point with normalization
def compute_latent_harmonics(phi_new, V, sigma, reduced, epsilon):
    """
    Compute latent harmonics for a new point with normalized kernel.
    
    Parameters:
    - phi_new: The new point in the reduced space.
    - V: Eigenvectors from the diffusion operator (each column is an eigenvector).
    - sigma: Eigenvalues corresponding to V.
    - reduced: Reduced dataset used to compute the kernel.
    - epsilon: Bandwidth parameter for the RBF kernel.
    
    Returns:
    - V_new: Latent harmonics for the new point.
    """
    # Step 1: Compute the unnormalized kernel values
    # print(type(rbf_kernel(phi_new.reshape(1, -1), reduced, gamma=1 / (2 * epsilon ** 2))))
    # print(reduced.shape)
    # print(phi_new.reshape(1, -1).shape)
    # np.savetxt("phiNew.csv", phi_new.reshape(-1, 1), delimiter=',', fmt='%.6f')
    # np.savetxt("reduced.csv", reduced, delimiter=',', fmt='%.6f')
    #tmp = rbf_kernel(phi_new.reshape(1, -1), reduced, gamma=1 / (2 * epsilon ** 2))
    K_new = rbf_kernel(phi_new.reshape(1, -1), reduced, gamma=1 / (2 * epsilon ** 2)).flatten()
    #print('K_new',K_new)
    # K_new = compute_similarity_matrix(phi_new, reduced) .flatten()
    # Step 2: Normalize the kernel values
    print(K_new)
    K_new_normalized = K_new / K_new.sum()  # Normalize by the sum to ensure it's a probability distribution
   
    # Step 3: Compute the latent harmonics
    V_new = []
    for j in range(len(sigma)):
        v_j = V[:, j]
        V_j_new = (1 / sigma[j]) * np.dot(K_new_normalized, v_j)  # Use the normalized kernel values
        V_new.append(V_j_new)
    return np.array(V_new)


def compute_latent_harmonics_stable(phi_new, V, sigma, reduced, epsilon):
    """
    Compute latent harmonics for a new point using log-sum-exp trick to prevent underflow.
    
    Parameters:
    - phi_new: The new point in the reduced space.
    - V: Eigenvectors from the diffusion operator (each column is an eigenvector).
    - sigma: Eigenvalues corresponding to V.
    - reduced: Reduced dataset used to compute the kernel.
    - epsilon: Bandwidth parameter for the RBF kernel.
    
    Returns:
    - V_new: Latent harmonics for the new point.
    """
    # Step 1: Compute squared Euclidean distances
    # Using np.linalg.norm is efficient for this
    sq_dist = np.linalg.norm(reduced - phi_new, axis=1)**2
    
    # Step 2: Compute the log of the unnormalized kernel values
    # log(K_new[i]) = -sq_dist[i] / (2 * epsilon^2)
    log_K_new = -sq_dist / (2 * epsilon**2)
    
    # Step 3: Normalize the kernel values in log-space using log-sum-exp trick
    # log(K_new_normalized[i]) = log_K_new[i] - log(sum(exp(log_K_new)))
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x)))) for numerical stability
    max_log_K = np.max(log_K_new)
    log_sum_K = max_log_K + np.log(np.sum(np.exp(log_K_new - max_log_K)))
    log_K_new_normalized = log_K_new - log_sum_K
    
    # Step 4: Compute the latent harmonics
    # V_j_new = sum_i K_new_normalized[i] * V[i, j] / sigma[j]
    # In log-space, K_new_normalized[i] = exp(log_K_new_normalized[i])
    V_new = []
    for j in range(len(sigma)):
        v_j = V[:, j]
        # Use exp to get back the normalized kernel weights
        K_weights = np.exp(log_K_new_normalized)
        V_j_new = (1 / sigma[j]) * np.dot(K_weights, v_j)
        V_new.append(V_j_new)
    
    return np.array(V_new)

phi_new_u = dmap_u[-1,:]  # New input point in U
#phi_new_v = reduced_v[-1]  # New input point in V

V_new_u = compute_latent_harmonics_stable(phi_new_u, V_u[:,1:7], sigma_u[1:7], phi, epsilon_u)
#V_new_v = compute_latent_harmonics(phi_new_v, V_v, sigma_v, reduced_v, epsilon_v)

# Step 8: Predict in the probabilistic manifold
def predict_in_manifold(c, V_new):
    """
    Predict target function in the manifold space.
    """
    return np.sum(c* V_new, axis=1)

phi_u_m_next = predict_in_manifold(c_u, V_new_u).reshape(1,-1)

#phi_v_m_next = predict_in_manifold(c_v, V_new_v)
print('得到phi_u_m_next的预测值')

# 步骤2: 创建DMD对象并拟合数据
#DMD_u = DMD(svd_rank=12)  # 设置降维到13维
#DMD_u.fit(original_u[:,2:vtu_num_])  # DMD需要列为时间快照


S = original_u[:,20:]
S = S/np.max(np.max(S))
Sd = S[:,0:128]
Spred = S[:,128:]

from funs import centralize, eig, proj_op, reduce
#Sd_for_pod = Sd[:,0::4]
Sd_for_pod = Sd
Sc_for_pod,sd_mean = centralize(Sd_for_pod)
evals_cla, evecs_cla = eig(Sc_for_pod)

from quantum_svd import QSVD
M = Sc_for_pod.T@Sc_for_pod
qsvd = QSVD(M)

qsvd.initialize_params(n_layers=50)
loss_hist = qsvd.optimize(iterations=100, lr=0.01)
#plt.plot(loss_hist)
#plt.xlabel("Iteration")
#plt.ylabel("Loss")
#plt.title("Training Progress")
#plt.show()


q_evecs = qsvd.evecs
q_evals = qsvd.evals
print(q_evals)

sorted_indices = np.argsort(q_evals)
q_evals = q_evals[sorted_indices[::-1]].real
q_evecs = q_evecs[:,sorted_indices[::-1]].real

# quantum framework using 5 POD bases
r = 6
Pr_q,s_mean,fid = proj_op(Sd_for_pod,q_evals,q_evecs,r)
Sr_q, u_hat_q = reduce(Sd, Pr_q, s_mean)
Spred_r, u_hat_pred = reduce(Spred, Pr_q, s_mean)
u_hat_q_train=u_hat_q#.reshape(1,-1)
u_mean=np.average(u_hat_q_train,axis=1).reshape(-1,1)
u_hat_q_train_cen=u_hat_q_train-u_mean

# quantum DKL
from quantum_dkl import qdkl
qdkl_q=qdkl(u_hat_q_train_cen)


qdkl_q.train(n_epochs=300)
qdkl_q.pred(180)

u_hat_p_q=qdkl_q.u_hat_p.numpy()+u_mean
Scp = Pr_q@u_hat_p_q
Sp_q = Scp+s_mean

 

max_S = np.max(np.max(original_u[:,20:]))
Sp_q = Sp_q*max_S
print(Sp_q)
# Loop to predict from m+1 to m+10
predicted_data = []
predicted_data1 = []
predicted_data.append(phi_u_m_next)
predicted_data1.append(Sp_q[:,1])



for k in range(10):  # Predict from m+1 to m+10
    phi_new_u = predicted_data[k]
    #phi_new_u = phi_new_u[0,:]
    V_new_u = compute_latent_harmonics_stable(phi_new_u, V_u[:,1:7], sigma_u[1:7], phi, epsilon_u)
    phi_u_m_next = predict_in_manifold(c_u, V_new_u).reshape(1,-1)
    predicted_data.append(phi_u_m_next)
    predicted_u = Sp_q[:,k + 2]
    predicted_data1.append(predicted_u)
#print(predicted_data.shape)


   
# 对新数据点进行多项式特征转换
    new_data_point_poly1 = poly_u.transform(predicted_data[k])
    new_data_point_poly2 = poly_v.transform(predicted_data[k])

# 使用拟合后的线性回归模型对转换后的数据点进行预测
    predicted_new_point1 = model1.predict(new_data_point_poly1) + predicted_data1[k] #145-154
    predicted_new_point2 = model2.predict(new_data_point_poly2)



# 拼接预测结果
    velocity_predicted = np.concatenate((predicted_new_point1.T, predicted_new_point2.T), axis=1)
    velocity_predicted = torch.tensor(velocity_predicted).type(torch.float)
    file_path1 = f"C:\\Users\\ASUS\\Desktop\\flow_past_cylinder_Re2000_200\\cylinder_{vtu_num_ + k}.vtu"
    # 读取相应的 mesh 文件
    mesh = meshio.read(file_path1)
    velocity = mesh.point_data["Velocity"][:, [0,1]]  # u,v方向速度
    velocity = torch.tensor(velocity).type(torch.float)
    print(np.linalg.norm(velocity_predicted[:,0]-velocity[:,0]))
    #print('MSE',mean_squared_error(velocity_predicted[:,0], velocity[:,0]))

    
    """ triangle_cells = mesh.get_cells_type('triangle')
    edges = triangles_to_edges(tf.convert_to_tensor(triangle_cells))
    edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
    torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)
    mesh_pos = mesh.points[:,[0,1]]
    u_i=torch.tensor(mesh_pos)[edge_index[0]]
    u_j=torch.tensor(mesh_pos)[edge_index[1]]
    u_ij=u_i-u_j
    u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
    edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)
    cells=torch.tensor(triangle_cells)
    mesh_pos=torch.tensor(mesh_pos).type(torch.float)
    
    data=Data(x=velocity, edge_index=edge_index, edge_attr=edge_attr, cells=cells,mesh_pos=mesh_pos)
    data1=Data(x=velocity_predicted, edge_index=edge_index, edge_attr=edge_attr, cells=cells,mesh_pos=mesh_pos)
    data2=Data(x=velocity_predicted-velocity, edge_index=edge_index, edge_attr=edge_attr, cells=cells,mesh_pos=mesh_pos)
    make_animation(data)
    make_animation(data1)
    make_animation(data2)
     """
# 将新的velocity替换到mesh的数据中
    mesh.point_data["Velocity"][:, 0] = velocity_predicted[:, 0].numpy()  # 只替换第一列（u方向）

    # 将修改后的mesh保存为新的vtu文件
    output_file_path = f"C:\\Users\\ASUS\\Desktop\\flow_past_cylinder_Re4000_200\\圆柱绕流预测数据Q1_6\\cylinder_{vtu_num_ + k}.vtu"
    meshio.write(output_file_path, mesh)

    mesh.point_data["Velocity"][:, 0] = velocity_predicted[:, 0].numpy()-velocity[:, 0].numpy()  # 只替换第一列（u方向）

    # 将修改后的mesh保存为新的vtu文件
    output_file_path = f"C:\\Users\\ASUS\\Desktop\\flow_past_cylinder_Re4000_200\\圆柱绕流误差数据Q1_6\\cylinder_{vtu_num_ + k}.vtu"
    meshio.write(output_file_path, mesh)
print("PMD已完成")


