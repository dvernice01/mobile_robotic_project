import os
import select
import sys
import math
import struct
import numpy as np
import rclpy
import tf2_geometry_msgs 

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

from geometry_msgs.msg import Twist, TwistStamped, PointStamped
from sensor_msgs.msg import PointCloud2

from rclpy.clock import Clock
from rclpy.qos import QoSProfile
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)

from pyclustering.cluster.dbscan import dbscan
from pyclustering.utils import timedcall

from tf2_ros import Buffer, TransformListener


BURGER_MAX_LIN_VEL = 2.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 2.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.01

TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')

msg = """
Control Your TurtleBot3!
---------------------------
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity (Burger : ~ 0.22, Waffle and Waffle Pi : ~ 0.26)
a/d : increase/decrease angular velocity (Burger : ~ 2.84, Waffle and Waffle Pi : ~ 1.82)

space key, s : force stop

CTRL-C to quit
"""

e = """
Communications Failed
"""

# vettore dei guadagni 
APF_CONFIG = {
    'repulsive_gain': 0.1,      # Guadagno forza repulsiva
    'attractive_gain': 5.0,     # Guadagno forza attrattiva  
    'LIN_VELOCITY_GAIN': 1.0,
    'ANG_VELOCITY_GAIN': 0.1,
}

def get_key(settings):
    if os.name == 'nt': #definisce quale sistema operativo sta usando  
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))

#introduce una rampa per aumentare la vel in maniera incrementale
def make_simple_profile(output_vel, input_vel, slop):
    if input_vel > output_vel:
        output_vel = min(input_vel, output_vel + slop)
    elif input_vel < output_vel:
        output_vel = max(input_vel, output_vel - slop)
    else:
        output_vel = input_vel

    return output_vel
    

# make_simple_profile serve a saturare l'uscita nel caso in cui chiediamo un input troppo alto 
def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel


def check_linear_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)


def check_angular_limit_velocity(velocity):
    if TURTLEBOT3_MODEL == 'burger':
        return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    else:
        return constrain(velocity, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)
    
def update_velocity(pub,ROS_DISTRO,lin_velocity, ang_velocity):
    if ROS_DISTRO == 'humble':
        twist = Twist()
        twist.linear.x = lin_velocity
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = ang_velocity
        pub.publish(twist)

    else:
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = Clock().now().to_msg()
        twist_stamped.header.frame_id = ''
        twist_stamped.twist.linear.x = lin_velocity
        twist_stamped.twist.linear.y = 0.0
        twist_stamped.twist.linear.z = 0.0

        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = 0.0
        twist_stamped.twist.angular.z = ang_velocity
        pub.publish(twist)


class PerceptionNode(Node):

    def __init__(self):
        
        super().__init__('perception_node')
        self.object_pub = PointStamped ()
        self.cb_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(PointCloud2, '/filtered_pc', 10)
        self.sub = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.perception_callback,
            10,
            callback_group=self.cb_group
        )
        
        
    def perception_callback(self, msg):
        
        objects = self.get_obstacle_positions(msg)
        self.object_pub.point.x = objects[0]
        self.object_pub.point.y = objects[1]
        self.object_pub.point.z = objects[2]
        self.pub.publish(self.object_pub)
        
        
    def get_obstacle_positions(self, msg):
        
        obstacles = []
        objects_detected = []
        x_offset = 0
        y_offset = 4
        z_offset = 8
        
        #now = self.node.get_clock().now()
        msg_time = Time.from_msg(msg.header.stamp)
        
        # Verifica che il TF sia disponibile PRIMA di processare i punti
        try:
            # Cerca la trasformazione con timeout più lungo
            transform = self.tf_buffer.lookup_transform(
                'zed_camera_link',    # target frame
                msg.header.frame_id,  # source
                msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            
            
        except Exception as e:
            self.node.get_logger().warn(f"TF non disponibile: {e}")
            return []  # Ritorna lista vuota se non c'è TF
        
        # Ora processa i punti
        for i in range(0, len(msg.data), msg.point_step):
            point_data = msg.data[i:i + msg.point_step]
            
            x = struct.unpack('f', point_data[x_offset:x_offset+4])[0] 
            y = struct.unpack('f', point_data[y_offset:y_offset+4])[0]  
            z = struct.unpack('f', point_data[z_offset:z_offset+4])[0]

            
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
                continue

            # Crea il punto nel frame della camera
            p = PointStamped()
            p.header.frame_id = msg.header.frame_id
            p.header.stamp = msg.header.stamp
            p.point.x = x
            p.point.y = y
            p.point.z = z

            # Trasforma nel frame del robot (usa il transform già ottenuto)
            try:
                p_chassis = tf2_geometry_msgs.do_transform_point(p, transform)
                
                x_robot = p_chassis.point.x
                y_robot = p_chassis.point.y
                z_robot = p_chassis.point.z
                
                # Filtra punti validi
                if (0.1 < x_robot <= 1.0 and
                    abs(y_robot) < 3.0 and
                    -0.5 < z_robot < 2.5):
                    
                    #print(f"Trasformato: camera[{x:.2f},{y:.2f},{z:.2f}] -> chassis[{x_robot:.2f},{y_robot:.2f},{z_robot:.2f}]")
                    objects_detected.append([x_robot, y_robot, z_robot])
                    
            except Exception as e:
                self.node.get_logger().error(f"Errore trasformazione punto: {e}")
                continue  
            
        return [x_robot, y_robot, z_robot]  
            
#L'OBIETTIVO E' FARE UN CLAUSTER AL FINE DI RICONOSCERE GLI OSTACOLI COSÌ DA FARE IL CONTROLLO RISPETTO AD UN 
# OSTACOLO PIUTTOSTO CHE DA CIASCUN PUNTO
           
class DBSCANNode(Node):
    
    def __init__(self):
        
        self.cb_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(PointStamped, '/obstacles', 10)
        self.sub = self.create_subscription(
            PointCloud2,
            '/filtered_pc',
            self.dbscan_callback,
            10,
            callback_group=self.cb_group
        )
        
    def cluster_obstacles(self, pointcloud_data):
        
        if not pointcloud_data:
            return []
        
        valid_points_3d = []
        for points in pointcloud_data:
            valid_points_3d.append(points)        
            
            if len(valid_points_3d) < self.min_points:
                return []
            
            # 2. Applica DBSCAN
            dbscan_instance = dbscan(valid_points_3d, self.eps, self.min_points)
            
            # Misura il tempo di esecuzione
            time_execution, _ = timedcall(dbscan_instance.process)
            
            # 3. Estrai i cluster
            clusters = dbscan_instance.get_clusters()
            
            print(f"DBSCAN trovato {len(clusters)} ostacoli in {time_execution:.3f}s")
            
            # 4. Converti i cluster in ostacoli 3D
            obstacles_3d = []
            
            for cluster_indices in clusters:
                cluster_points = []
                        
                # Calcola il centroide dell'ostacolo
                if cluster_points:
                    centroid = self.calculate_centroid(cluster_points)
                    obstacles_3d.append({
                        'centroid': centroid,  # [x, y, z] del centro
                        'points': cluster_points,  # Tutti i punti del cluster
                        'size': len(cluster_points),
                        'distance': math.sqrt(centroid[0]**2 + centroid[1]**2 + centroid[2]**2)
                    })
            
        return obstacles_3d
    
    def calculate_centroid(self, points):

        if not points:
            return None
        
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points) 
        sum_z = sum(p[2] for p in points)
        
        n = len(points)
        return [sum_x/n, sum_y/n, sum_z/n]
    
    def get_largest_obstacle(self, obstacles):
        """Restituisce l'ostacolo più grande (più punti)"""
        if not obstacles:
            return None
        return max(obstacles, key=lambda x: x['size'])
    
    def get_closest_obstacle(self, obstacles):
        """Restituisce l'ostacolo più vicino"""
        if not obstacles:
            return None
        return min(obstacles, key=lambda x: x['distance'])

    def dbscan_callback(self, msg):
        
        obstacles = self.cluster_obstacles(msg)

        for o in obstacles:
            p = PointStamped()
            p.point.x, p.point.y, p.point.z = o
            self.pub.publish(p)
            
class TeleopNode(Node):

    def __init__(self):

        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.pub = self.create_publisher(Twist, '/cmd_vel_teleop', 10)
        self.create_subscription(
            PointStamped, '/obstacles', self.teleop_callback, 10, callback_group=self.cb_group
        )

        self.target_lin = 0.0
        self.target_ang = 0.0
        self.ctrl_lin = 0.0
        self.ctrl_ang = 0.0

        self.settings = termios.tcgetattr(sys.stdin)
        self.timer = self.create_timer(0.05, self.loop, callback_group=self.cb_group)


    def teleop_callback(self,obstacles):
        
        if len(obstacles) > 0:
            return
        else:
            key = get_key(self.settings)

            if key == 'w':
                self.target_lin = check_linear_limit_velocity(self.target_lin + LIN_VEL_STEP_SIZE)
            elif key == 'x':
                self.target_lin = check_linear_limit_velocity(self.target_lin - LIN_VEL_STEP_SIZE)
            elif key == 'a':
                self.target_ang = check_angular_limit_velocity(self.target_ang + ANG_VEL_STEP_SIZE)
            elif key == 'd':
                self.target_ang = check_angular_limit_velocity(self.target_ang - ANG_VEL_STEP_SIZE)
            elif key == ' ':
                self.target_lin = self.target_ang = 0.0

            self.ctrl_lin = make_simple_profile(self.ctrl_lin, self.target_lin, LIN_VEL_STEP_SIZE / 2)
            self.ctrl_ang = make_simple_profile(self.ctrl_ang, self.target_ang, ANG_VEL_STEP_SIZE / 2)

            update_velocity(pub,ROS_DISTRO,self.ctrl_lin, self.ctrl_ang)
            
            
class APFNode(Node):
    
    def __init__(self):
        
        self.teleop_vel = 0
        self.obstacle = []
        self.cb_group = MutuallyExclusiveCallbackGroup()
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            PointStamped, '/obstacles', self.apf_callback, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Twist, '/cmd_vel_teleop', self.apfTeleop_callback, 10, callback_group=self.cb_group
        )
        
    def compute_attractive_forces(self, v_lin, angle):
        
        #attractive_force = self.config['attractive_gain'] * (v_lin * np.sin (np.arctan2(y_dist, x_dist)))
        new_lin_vel = v_lin * np.cos (angle)
        attractive_force = v_lin * np.sin(angle) * self.config['attractive_gain'] # angolo verde = atan di (y/x) e si semplifica con tan
        
        return [ new_lin_vel, attractive_force ]
    
    def compute_repulsive_forces(self, x_dist, angle):

        gamma = 1
        eta_0 = 1
        eta_i = x_dist * np.sin(angle)
        eta_i = np.sign(eta_i) * max(abs(eta_i), 0.1)
        repulsive_force = (self.config['repulsive_gain']/eta_i**2) * ((1/eta_i - 1/eta_0)**(gamma - 1))
        
        return repulsive_force

    def compute_total_force(self, control_linear_velocity, distance):
        
        angle = np.atan2 (distance[1] , distance[0])
        print(f"[OSTACOLO] x={distance[0]:.2f}, y={distance[1]:.2f}, z={distance[2]:.2f}, angle={np.degrees(angle):.1f}°")
        [ new_lin_vel, attractive_force ] = self.compute_attractive_forces (control_linear_velocity, angle)
        repulsive_force = self.compute_repulsive_forces(distance[0], angle )
        total_force = attractive_force - repulsive_force
         
        return [ new_lin_vel, total_force, angle ]
    
    def apfTeleop_callback(self, TeleoperationVel):
        
        self.teleop_vel = TeleoperationVel
    
    def apf_callback(self,obstacles):
        
        for o in obstacles:
            
            self.obstacle = o
            distance_x = o[0]
            distance_y = o[1]
            distance_z = o[2]
        
        if len(obstacles < 1):
            
            self.pub.publish(self.teleop_vel)
        else:
            
            [apf_lin_vel, apf_ang_vel, angle] = compute_total_force(self, self.teleop_vel, self.obstacle)
            self.ctrl_lin = make_simple_profile(self.ctrl_lin, apf_lin_vel, LIN_VEL_STEP_SIZE / 2)
            self.ctrl_ang = make_simple_profile(self.ctrl_ang, apf_ang_vel, ANG_VEL_STEP_SIZE / 2)

            update_velocity(self.pub,ROS_DISTRO,self.ctrl_lin, self.ctrl_ang)
            
            
def main():
    rclpy.init()

    executor = MultiThreadedExecutor(num_threads=4)

    nodes = [
        PerceptionNode(),
        DBSCANNode(),
        TeleopNode(),
        APFNode()
    ]

    for n in nodes:
        executor.add_node(n)

    executor.spin()


if __name__ == '__main__':
    main()
        
 