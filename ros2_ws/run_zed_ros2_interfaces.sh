apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
rm -rf build install log
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
echo source $(pwd)/install/local_setup.bash >> ~/.bashrc
source ~/.bashrc
colcon build
source install/setup.bash