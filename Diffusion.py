import os
import pandas as pd
import meshio
import numpy as np
import torch
from matplotlib import colors, tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf # type: ignore
from sklearn.manifold import Isomap



def triangles_to_edges(faces):
  """Computes mesh edges from triangles.
     Note that this triangles_to_edges method was provided as part of the
     code release for the MeshGraphNets paper by DeepMind, available here:
     https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
  """

  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  #print(edges.shape)
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  #print(packed_edges.shape)
  # remove duplicates and unpack
  df = pd.DataFrame(packed_edges)
  unique_edges = df.drop_duplicates()
  unique_edges = tf.convert_to_tensor(unique_edges)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))


def get_vtu_parametric_flow():
    base_dir = r"D:\parametric PMD\Flow around a cylinder"
    start = 0.2
    end = 10.1
    step = 0.1

    num_steps = int((end - start) / step) + 1

    velocity_list = []

    for idx in range(num_steps):
        param = round(start + idx * step, 1)  # 0.2, 0.3, ..., 10.1
        folder_name = f"flow{param}e-4"  # 注意：没有空格！
        folder_path = os.path.join(base_dir, folder_name)

        vtu_file = os.path.join(folder_path, "circle-2d-drag_150.vtu")  # 你如果要改成109，这里改
        if not os.path.exists(vtu_file):
            print(f"警告: {vtu_file} 文件不存在，跳过")
            continue

        mesh = meshio.read(vtu_file)
        velocity_ts = mesh.point_data["Velocity"][:, [0, 1]]
        velocity_ts = torch.tensor(velocity_ts).type(torch.float)

        velocity_list.append(velocity_ts)

    velocity = torch.hstack(velocity_list)
    print('矩阵规模为', velocity.shape)
    return velocity


def get_vtu_parametric_step():
    base_dir = r"D:\parametric PMD\backward facing step"
    start = 2
    end = 200
    step = 2

    num_steps = int((end - start) / step) + 1

    velocity_list = []

    for idx in range(num_steps):
        param = round(start + idx * step, 1)  
        folder_name = f"step{param}e-5"  # 注意：没有空格！
        folder_path = os.path.join(base_dir, folder_name)

        vtu_file = os.path.join(folder_path, "backward_facing_step_2d_25.vtu")  # 你如果要改成109，这里改
        if not os.path.exists(vtu_file):
            print(f"警告: {vtu_file} 文件不存在，跳过")
            continue

        mesh = meshio.read(vtu_file)
        velocity_ts = mesh.point_data["Velocity"][:, [0, 1]]
        velocity_ts = torch.tensor(velocity_ts).type(torch.float)

        velocity_list.append(velocity_ts)

    velocity = torch.hstack(velocity_list)
    print('矩阵规模为', velocity.shape)
    return velocity


def get_vtu_num(path):
	# count the number of vtu files
		f_list = os.listdir(path)
		vtu_num = 0
		for i in f_list:
			if os.path.splitext(i)[1] == '.vtu':
				vtu_num = vtu_num+1
		return vtu_num

def get_dataset(path):
  
    # path 是你的vtu所在文件的位置
    vtu_num = get_vtu_num(path)  # 查看有多少vtu文件
    print('文件数量为',vtu_num)
    
    for ts in range(vtu_num):
        if ts == 0:
            mesh = meshio.read(path + "/cylinder_" + str(ts) + ".vtu")
            velocity = mesh.point_data["Velocity"][:, [0,1]]  # u, v方向速度
            velocity = torch.tensor(velocity).type(torch.float)
        else:
            mesh = meshio.read(path + "/cylinder_" + str(ts) + ".vtu")
            velocity_ = mesh.point_data["Velocity"][:, [0,1]]  # u, v方向速度
            velocity_ = torch.tensor(velocity_).type(torch.float)
            
            # 使用 hstack 正确地合并张量
            velocity = torch.hstack([velocity, velocity_])

    print('矩阵规模为',velocity.shape)
    return velocity

def get_dataset1(path):
  
    # path 是你的vtu所在文件的位置
    vtu_num = get_vtu_num(path)  # 查看有多少vtu文件
    print('文件数量为',vtu_num)
    
    for ts in range(vtu_num):
        if ts == 0:
            mesh = meshio.read(path + "/lock_exchange_" + str(ts) + ".vtu")
            velocity = mesh.point_data["Temperature"].reshape(-1,1)
            velocity = torch.tensor(velocity).type(torch.float)
        else:
            mesh = meshio.read(path + "/lock_exchange_" + str(ts) + ".vtu")
            velocity_ = mesh.point_data["Temperature"].reshape(-1,1)
            velocity_ = torch.tensor(velocity_).type(torch.float)
            
            # 使用 hstack 正确地合并张量
            velocity = torch.hstack([velocity, velocity_])

    print('矩阵规模为',velocity.shape)
    return velocity

def make_animation(gs):
    '''
    
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.cla()
    ax.set_aspect('equal')
    ax.set_axis_off()
    pos = gs.mesh_pos
    velocity = gs.x
    faces = gs.cells
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    norm = colors.Normalize(vmin=velocity.min(), vmax=velocity.max())

    mesh_plot = ax.tripcolor(triang, velocity[:, 0], cmap='rainbow', norm=norm, shading='flat') # x-velocity
    #mesh_plot = ax.tripcolor(triang, velocity[:, 0], cmap='rainbow', vmin= 0, vmax=1,  shading='flat' ) # x-velocity
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #ax.set_title( fontsize = '20')
    divider = make_axes_locatable(ax)#在ax上创建一个可分离区域
    cax = divider.append_axes('right', size='5%', pad=0.05)
    clb = fig.colorbar(mesh_plot, ax=ax, cax=cax,orientation='vertical')

    #clb = fig.colorbar(mesh_plot,cax= ,orientation='vertical')

    clb.ax.tick_params(labelsize=20)
    clb.ax.set_title('x velocity  (m/s)',fontdict = {'fontsize': 20})
    
    plt.show()
    return fig,

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

def construct_transition_matrix_from_geodesic(geodesic_dist_matrix, sigma):
    """
    根据测地距离矩阵构建转移矩阵
    :param geodesic_dist_matrix: 测地距离矩阵
    :param sigma: 高斯核尺度参数
    :return: 转移矩阵
    """
    transition_matrix = np.exp(-geodesic_dist_matrix**2 / (2 * sigma**2))
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)  # 行归一化
    return transition_matrix