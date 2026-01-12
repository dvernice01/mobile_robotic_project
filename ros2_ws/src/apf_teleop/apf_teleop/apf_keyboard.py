#debug
import os
import select
import sys

#librerie importate da me 
import math
import struct
import numpy as np

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PointStamped

from sensor_msgs.msg import PointCloud2
import rclpy
from rclpy.clock import Clock
from rclpy.qos import QoSProfile
from rclpy.time import Time
from rclpy.duration import Duration

from pyclustering.cluster.dbscan import dbscan
from pyclustering.utils import timedcall
import tf2_geometry_msgs 
from tf2_ros import Buffer, TransformListener


if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

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
    'repulsive_gain': 2.0,      # Guadagno forza repulsiva
    'attractive_gain': 2.0,     # Guadagno forza attrattiva  
    'LIN_VELOCITY_GAIN': 1.0,
    'ANG_VELOCITY_GAIN': 0.2,
}

# get_key serve a prendere il singolo tasto premuto.
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
        
        
        
class ObstacleDetector:
    def __init__(self, eps=0.5, min_points=3):
        """
        DBSCAN parameters:
        - eps: distance threshold for neighborhood (in meters)
        - min_points: minimum points to form a cluster
        """
        self.eps = eps
        self.min_points = min_points
        
    def cluster_obstacles(self, pointcloud_data):
        """Clusterizza i punti per identificare ostacoli separati"""
        if not pointcloud_data:
            return []
        
        # 1. Prepara i dati per DBSCAN (solo coordinate x,y per clustering 2D)
        points_2d = []
        valid_points_3d = []
        
        for point in pointcloud_data:
            x, y, z = point
            # Usa solo punti validi per il clustering
            if 0.3 < x <= 3.0 and z > -0.5:  # e z >= 0.4 se vuoi
                points_2d.append([x, y])  # DBSCAN su coordinate 2D
                valid_points_3d.append(point)
        
        if len(points_2d) < self.min_points:
            return []
        
        # 2. Applica DBSCAN
        dbscan_instance = dbscan(points_2d, self.eps, self.min_points)
        
        # Misura il tempo di esecuzione
        time_execution, _ = timedcall(dbscan_instance.process)
        
        # 3. Estrai i cluster
        clusters = dbscan_instance.get_clusters()
        noise = dbscan_instance.get_noise()
        
        print(f"DBSCAN trovato {len(clusters)} ostacoli, {len(noise)} punti rumore in {time_execution:.3f}s")
        
        # 4. Converti i cluster in ostacoli 3D
        obstacles_3d = []
        
        for cluster_indices in clusters:
            cluster_points = []
            for idx in cluster_indices:
                cluster_points.append(valid_points_3d[idx])
            
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
        """Calcola il centroide di un cluster di punti"""
        if not points:
            return [0, 0, 0]
        
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


class ArtificialPotentialField:
    #definisco le varie funzioni della classe 
    # nella funzione init definisco la configurazione e il vettore degli ostacoli 
    def __init__(self, config):
        self.config = config
        self.obstacles = []  
    
    def compute_attractive_forces(self, v_lin, angle):
        
        #attractive_force = self.config['attractive_gain'] * (v_lin * np.sin (np.arctan2(y_dist, x_dist)))
        new_lin_vel = v_lin * np.cos (angle)
        attractive_force = v_lin * np.sin(angle) * self.config['attractive_gain'] # angolo verde = atan di (y/x) e si semplifica con tan
        
        return [ new_lin_vel, attractive_force ]
    
    def compute_repulsive_forces(self, x_dist, angle):

        gamma = 1
        eta_0 = 3
        eta_i = x_dist * np.sin(angle)
        print(eta_i)
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


class APFController:

    def __init__(self, node, ros_distro, control_linear_velocity, control_angular_velocity, obstacle_detector_instance):
        self.node = node
        self.ros_distro = ros_distro
        self.control_linear_velocity = control_linear_velocity
        self.control_angular_velocity = control_angular_velocity
        self.apf = ArtificialPotentialField(APF_CONFIG)
        self.obstacle_detector = obstacle_detector_instance
        
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, node)

        #self.obstacle_centroids = [] -0.05 -0.04 0.02    x=2.83, y=1.17, z=0.32

    def point_cloud_callback(self, msg):
        #self.node.get_logger().info(f"PointCloud callback chiamata, frame={msg.header.frame_id}, punti={len(msg.data)} bytes")
        try:
            obstacles = self.get_obstacle_positions(msg)
            self.apf.obstacles = obstacles

        except Exception as e:
            self.node.get_logger().error(f"Errore pointcloud: {e}")

    def get_obstacle_positions(self, msg):
        obstacles = []
        objects_detected = []
        
        x_offset = 0
        y_offset = 4
        z_offset = 8
        
        now = self.node.get_clock().now()
        msg_time = Time.from_msg(msg.header.stamp)

        # 🔴 SCARTA POINTCLOUD VECCHIE (>0.5s)
        if (now - msg_time).nanoseconds > 5e8:
            self.node.get_logger().warn("PointCloud troppo vecchia, scarto")
            return []
        
        # Verifica che il TF sia disponibile PRIMA di processare i punti
        try:
            # Cerca la trasformazione con timeout più lungo
            transform = self.tf_buffer.lookup_transform(
                'zed_camera_link',  # target frame
                msg.header.frame_id,              # source
                msg_time,                         
                timeout=Duration(seconds=0.5)
            )
            self.node.get_logger().info(f"TF trovato: {msg.header.frame_id} -> chassis_link")
            
        except Exception as e:
            self.node.get_logger().warn(f"TF non disponibile: {e}")
            return []  # Ritorna lista vuota se non c'è TF
        
        # Ora processa i punti
        for i in range(0, len(msg.data), msg.point_step):
            point_data = msg.data[i:i + msg.point_step]
            
            x = struct.unpack('f', point_data[x_offset:x_offset+4])[0] 
            y = struct.unpack('f', point_data[y_offset:y_offset+4])[0]  
            z = struct.unpack('f', point_data[z_offset:z_offset+4])[0]
            
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
                #print(f"Robot frame:  x={x_robot:.2f}, y={y_robot:.2f}, z={z_robot:.2f}")
                
                # Filtra punti validi
                if (0.3 < x_robot <= 3.0 and
                    abs(y_robot) < 3.0 and
                    -0.5 < z_robot < 2.5):
                    
                    #print(f"Trasformato: camera[{x:.2f},{y:.2f},{z:.2f}] -> chassis[{x_robot:.2f},{y_robot:.2f},{z_robot:.2f}]")
                    objects_detected.append([x_robot, y_robot, z_robot])
                    
            except Exception as e:
                self.node.get_logger().error(f"Errore trasformazione punto: {e}")
                continue
        
        # Clusterizza gli ostacoli
        claustered_objects = self.obstacle_detector.cluster_obstacles(objects_detected)
        for obstacle in claustered_objects:
            obstacles.append(obstacle['centroid'])
        
        return obstacles



def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    ROS_DISTRO = os.environ.get('ROS_DISTRO')
    qos = QoSProfile(depth=10)
    node = rclpy.create_node('apt_keyboard')
    node.set_parameters([
        rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )
    ])

    status = 0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0

    # Crea il controller APF
   #apf_controller = APFController(node, ROS_DISTRO, control_linear_velocity, control_angular_velocity,pub)
    obstacle_detector_instance = ObstacleDetector(eps=0.2, min_points=5)
   
    if ROS_DISTRO == 'humble':
        pub = node.create_publisher(Twist, 'cmd_vel', qos)
            # Crea il controller APF
        apf_controller = APFController(node, ROS_DISTRO, control_linear_velocity, control_angular_velocity, obstacle_detector_instance)
        sub_pointcloud = node.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            apf_controller.point_cloud_callback,
            qos)
    else:
        pub = node.create_publisher(TwistStamped, 'cmd_vel', qos)
            # Crea il controller APF
        apf_controller = APFController(node, ROS_DISTRO, control_linear_velocity, control_angular_velocity, obstacle_detector_instance)
        sub_pointcloud = node.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            apf_controller.point_cloud_callback,
            qos)

    try:
        print(msg)

        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
           
            if len(apf_controller.apf.obstacles) > 0 and control_linear_velocity >= 0.3:
                print("APF attivo, numero ostacoli:", len(apf_controller.apf.obstacles))

                # Modalità APF
                for obstacle in apf_controller.apf.obstacles:
                    x, y, z = obstacle
                    distance = [x, y, z]

                    [ velocity, force, angle ] = apf_controller.apf.compute_total_force(
                        control_linear_velocity, 
                        distance
                    )
                    """print('velocities:', velocities[0], velocities[1])
                    MAX_VEL = 2.0
                    
                    if abs(velocities[0]) > MAX_VEL:
                        velocities[0] = np.sign(velocities[0]) * MAX_VEL
                       
                    if abs(velocities[1]) > MAX_VEL:
                        velocities[1] = np.sign(velocities[1]) * MAX_VEL"""
                    
                    #final_lin_vel = control_linear_velocity - (np.sign (control_linear_velocity) * velocities[0] * APF_CONFIG['LIN_VELOCITY_GAIN'])
                    final_lin_vel = velocity * APF_CONFIG['LIN_VELOCITY_GAIN']
                    #if velocities[1] == 1:
                    #    final_ang_vel = 0
                    #else: 
                    if force < 0:
                        sign = np.sign(distance[1])
                        final_ang_vel = -(sign*abs(force))
                    else:
                        final_ang_vel = control_angular_velocity
                        
                    final_ang_vel = check_angular_limit_velocity (final_ang_vel)
                    #final_ang_vel = control_angular_velocity - (np.sign (control_linear_velocity) * velocities[1] * APF_CONFIG['ANG_VELOCITY_GAIN'])
                    #final_ang_vel = - velocities[1] * APF_CONFIG['ANG_VELOCITY_GAIN']
                    
                update_velocity(pub,ROS_DISTRO, final_lin_vel, final_ang_vel)
                print('velocities: ', final_lin_vel, final_ang_vel)

            else:

                key = get_key(settings)
                if key == 'w':
                    target_linear_velocity =\
                        check_linear_limit_velocity(target_linear_velocity + LIN_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == 'x':
                    target_linear_velocity =\
                        check_linear_limit_velocity(target_linear_velocity - LIN_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == 'a':
                    target_angular_velocity =\
                        check_angular_limit_velocity(target_angular_velocity + ANG_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == 'd':
                    target_angular_velocity =\
                        check_angular_limit_velocity(target_angular_velocity - ANG_VEL_STEP_SIZE)
                    status = status + 1
                    print_vels(target_linear_velocity, target_angular_velocity)
                elif key == ' ' or key == 's':
                    target_linear_velocity = 0.0
                    control_linear_velocity = 0.0
                    target_angular_velocity = 0.0
                    control_angular_velocity = 0.0
                    print_vels(target_linear_velocity, target_angular_velocity)
                else:
                    if (key == '\x03'):
                        break

                if status == 20:
                    print(msg)
                    status = 0

                control_linear_velocity = make_simple_profile(
                    control_linear_velocity,
                    target_linear_velocity,
                    (LIN_VEL_STEP_SIZE / 2.0))
  
                control_angular_velocity = make_simple_profile(
                    control_angular_velocity,
                    target_angular_velocity,
                    (ANG_VEL_STEP_SIZE / 2.0))
                
                update_velocity(pub, ROS_DISTRO, control_linear_velocity, control_angular_velocity)
            
            


    except Exception as e:
        print(e)

    finally:
        if ROS_DISTRO == 'humble':
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            pub.publish(twist)
        else:
            twist_stamped = TwistStamped()
            twist_stamped.header.stamp = Clock().now().to_msg()
            twist_stamped.header.frame_id = ''
            twist_stamped.twist.linear.x = control_linear_velocity
            twist_stamped.twist.linear.y = 0.0
            twist_stamped.twist.linear.z = 0.0
            twist_stamped.twist.angular.x = 0.0
            twist_stamped.twist.angular.y = 0.0
            twist_stamped.twist.angular.z = control_angular_velocity
            pub.publish(twist_stamped)
            

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == '__main__': 
    main()