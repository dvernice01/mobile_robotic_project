
Instructions for reproducing the implemented control:

1) First of all, it is necessary to be able to access IsaacLab. To do this, simply follow the instructions at this link: https://github.com/isaac-sim/IsaacLab/tree/main. Then, from the folder containing the material, type the following commands:

  -python3 docker/container.py start ros2

  -python3 docker/container.py enter ros2

  -./isaaclab.sh -s

  -python3 docker/container.py stop ros2 (to stop the container once the simulation is finished)

2) Check that you have enabled the ROS2 Bridge Extension and the ZED Extension (link for ZED Extension: https://www.stereolabs.com/docs/isaac-sim). Once this is done, simply import the file turtlebot_waffle.usd and the IsaacLab environment should be functional.

3) The next step is to clone the repository with: git clone https://github.com/dvernice01/mobile_robotic_project.git

4) After cloning it, the image must be built by launching the command ./desktop_build_dockerfile_from_sdk_ubuntu_and_cuda_version.sh ubuntu-22.04 cuda-12.6.3 zedsdk-5.0.0 and it will be possible to access the container with the command ./run.sh

5) Once the container is open, launch the command ./run_zed_ros2_interfaces.sh and the environment is ready to be used.

6) Finally, start the simulation from IsaacLab and launch the commands:

  -ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true

  -ros2 run apf_teleop apf_teleop The robot can be teleoperated like a TurtleBot with the keys: W (forward) - A (rotate left) - D (rotate right) - X (backward) - S (reset velocities).
