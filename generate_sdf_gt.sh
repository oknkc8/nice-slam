python src/omni_utils/tsdf_fusion/generate_sdf_gt.py \
--dataset under_parking_seq \
--data_path /data/multipleye/under_parking_seq \
--pose_file trajectory_200.txt \
--save_name 200_tsdf_radius_50_omnimvs \
--voxel_size 0.04 \
--img_fmt cam_center_square_640/%05d.png \
--depth_fmt cam_center_square_640_depth/%05d.tiff \
# --img_fmt rgb_omni_160_640/sigma_3/%05d.png \
# --depth_fmt depth_omni_160_640/%05d.tiff \
--phi_deg -45.0 \
--phi_max_deg 45.0