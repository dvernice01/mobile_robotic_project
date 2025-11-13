docker run --gpus all -it --privileged --rm \
    --network host --ipc host --pid host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v /usr/local/zed/settings:/usr/local/zed/settings \
    -v /usr/local/zed/resources:/usr/local/zed/resources \
    -v /dev:/dev \
    -v /tmp:/tmp \
    -v /var/nvidia/nvcam/settings/A:/var/nvidia/nvcam/settings/ \
    -v /etc/systemd/system/zed_x_daemon.service:/etc/systemd/system/zed_x_daemon.service \
    -v /dev/shm:/dev/shm \
    -v /home/studenti01/dvernice/project/ros2_ws:/root/ros2_ws \
    --name zed_ros2_project \
    zed_ros2_desktop_u22.04_sdk_5.0.0_cuda_12.6.3
