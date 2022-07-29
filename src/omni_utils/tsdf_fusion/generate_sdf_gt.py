import sys
sys.path.append('.')

import time
import os
from src.omni_utils.tsdf_fusion.fusion import *
import pickle
import argparse as ap
from tqdm import tqdm
import torch.multiprocessing

from src.omni_utils.array import *
from src.omni_utils.camera import *
from src.omni_utils.common import *
from src.omni_utils.geometry import *
from src.omni_utils.image import *

torch.multiprocessing.set_sharing_strategy('file_system')



def parse_args():
    parser = ap.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='under_parking_seq')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to raw dataset", default='/data/chsung/under_parking_seq')
    parser.add_argument("--pose_file", default='trajectory.txt')
    parser.add_argument("--save_name", metavar="DIR",
                        help="file name", default='all_tsdf_200_radius_10')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.04, type=float)
    
    parser.add_argument("--img_fmt", default='cam_center_square_640/%05d.png')
    parser.add_argument("--depth_fmt", default='cam_center_square_640_depth/%05d.tiff')
    
    parser.add_argument('--phi_deg', default=-90.0, type=float)
    parser.add_argument('--phi_max_deg', default=90.0, type=float)

    parser.add_argument('--window_size', default=200, type=int)

    # ray multi processes
    return parser.parse_args()


args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)


def save_tsdf_full(args, depth_list, cam_pose_list, color_list, save_mesh=False):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    phi_deg = args.phi_deg
    phi_max_deg = args.phi_max_deg
    
    
    
    vol_bnds = np.zeros((3, 2))

    n_imgs = len(depth_list.keys())
    image_id = depth_list.keys()
    
    for id in tqdm(image_id):
        h, w = color_list[id].shape[0:2]
        
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        # Compute camera view frustum and extend convex hull        
        xs, ys = np.meshgrid(range(w), range(h)) # row major
        w_2, h_2 = w / 2.0, (h - 1) / 2.0
        xs = (xs - w_2) / w_2 * np.pi + (np.pi / 2.0)
        if phi_max_deg > 0.0:
            med = np.deg2rad((phi_max_deg - phi_deg) / 2.0)
            med2 = np.deg2rad((phi_max_deg + phi_deg) / 2.0)
            ys = (ys - h_2) / h_2 * med - med2
        else:
            ys = (ys - h_2) / h_2 * np.deg2rad(phi_deg)
            
        X = -np.cos(ys) * np.cos(xs)
        Y = np.sin(ys) # sphere
        # Y = np.sin(ys) / np.cos(ys) # cylinder
        # Y = ys / np.deg2rad(phi_deg) # perspective cylinder
        Z = np.cos(ys) * np.sin(xs)
        dirs = np.concatenate((np.reshape(X, [1, -1]),
            np.reshape(Y, [1,-1]), np.reshape(Z, [1,-1]))).astype(np.float64)

        pc = dirs * depth_im.reshape(1, -1)
        pc = applyTransform(cam_pose, pc)
        pc = pc.reshape(3, -1)
        
        
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(pc, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(pc, axis=1))
        
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin, use_gpu=True))

    # Loop through RGB-D images and fuse them together
    print("Integrate voxel volume...")
    t0_elapse = time.time()
    for id in tqdm(depth_list.keys()):
        # if id % 2 == 0:
        #     print("Fusing frame {}/{}".format(str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]

        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_pose, obs_weight=1., phi_deg=phi_deg, phi_max_deg=phi_max_deg)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
        np.savez_compressed(os.path.join(args.save_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    if save_mesh:
        for l in range(args.num_layers):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()

            meshwrite(os.path.join(args.save_path, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms,
                      colors)

            # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            # print("Saving point cloud to pc.ply...")
            # point_cloud = tsdf_vol_list[l].get_point_cloud()
            # pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)


def save_fragment_pkl(args, depth_list, cam_pose_list):
    phi_deg = args.phi_deg
    phi_max_deg = args.phi_max_deg
    
    fragments = []
    print('segment: process scene {}'.format(args.dataset))

    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    print("Get Fragment Bound...")
    for id in tqdm(depth_list.keys()):
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum_pano(depth_im, cam_pose, phi_deg, phi_max_deg)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            ids.append(id)
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum_pano(depth_im, cam_pose, phi_deg, phi_max_deg)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
            if count == args.window_size:
                all_ids.append(ids)
                all_bnds.append(vol_bnds)
                ids = []
                count = 0

    with open(os.path.join(args.save_path, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    
    print("Save Fragment...")
    # save fragments
    for i, bnds in tqdm(enumerate(all_bnds)):
        if not os.path.exists(os.path.join(args.save_path, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, 'fragments', str(i)))
        fragments.append({
            'scene': args.dataset,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'bnds': bnds,
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })

    with open(os.path.join(args.save_path, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)
        
def splitTrajectoryResult(trajectory: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = trajectory.shape[0]
    if rows != 8:
        sys.exit(
            'Trajectory must has 8 rows '
            '(fidx, rx, ry, rz, tx ty, tz, timestamp)')
    fidxs = trajectory[0, :].astype(np.int32)
    poses = trajectory[1:7, :].astype(np.float32)
    timestamps = trajectory[-1, :].astype(np.int64)
    return fidxs, poses, timestamps


def process(args):
    img_fmt = args.img_fmt
    depth_fmt = args.depth_fmt
    
    # config_path = os.path.join(args.input_folder, args.config_file)
    pose_path = os.path.join(args.data_path, args.pose_file)
    fidxs, cam_poses, _ = splitTrajectoryResult(np.loadtxt(pose_path).T)
    cam_poses = cam_poses.T
    
    depth_all = {}
    color_all = {}
    cam_pose_all = {}
    
    for i, fidx in tqdm(enumerate(fidxs)):
        color_path = os.path.join(args.data_path, img_fmt % (fidx))
        depth_path = os.path.join(args.data_path, depth_fmt % (fidx))
        
        color_img = np.array(Image.open(color_path), dtype=np.float32)
        depth_img = 1.0 / readImageFloat(depth_path)
        depth_img[depth_img >= 1e6] = 0
        
        R, tr = getRot(cam_poses[i]), getTr(cam_poses[i])
        cam_pose = np.eye(4)
        cam_pose[:3, :] = np.concatenate((R, tr), axis=-1)
        
        depth_all.update({fidx: depth_img})
        color_all.update({fidx: color_img})
        cam_pose_all.update({fidx: cam_pose})
   
    save_tsdf_full(args, depth_all, cam_pose_all, color_all, save_mesh=True)
    save_fragment_pkl(args, depth_all, cam_pose_all)



def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret

if __name__ == "__main__":
    # all_proc = args.n_proc * args.n_gpu

    # ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    # if args.dataset == 'scannet':
    #     if not args.test:
    #         args.data_path = os.path.join(args.data_path, 'scans')
    #     else:
    #         args.data_path = os.path.join(args.data_path, 'scans_test')
    #     files = sorted(os.listdir(args.data_path))
    # else:
    #     raise NameError('error!')

    # files = split_list(files, all_proc)

    # ray_worker_ids = []
    # for w_idx in range(all_proc):
    #     ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))

    # results = ray.get(ray_worker_ids)

    # if args.dataset == 'scannet':
    #     generate_pkl(args)

    process(args)