import os
import select
import sys
import math
import struct
import numpy as np
import rclpy
import tf2_geometry_msgs 
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')

import csv

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

from geometry_msgs.msg import Twist, TwistStamped, PointStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from rclpy.clock import Clock
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
ROS_DISTRO = os.environ.get('ROS_DISTRO')

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
    'repulsive_gain': 10.0,      # Guadagno forza repulsiva
    'attractive_gain': 0.5,     # Guadagno forza attrattiva  
    'LIN_VELOCITY_GAIN': 10.0,   # Guadagno velocità lineare 
    'ANG_VELOCITY_GAIN': 0.5,   # Guadagno velocità angolare
}


def get_key(settings):
    
    if os.name == 'nt':   
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
    
def update_velocity(pub,ros_distro,lin_velocity, ang_velocity):
    
    if ros_distro == 'humble':
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
        pub.publish(twist_stamped)


class PerceptionNode(Node):

    def __init__(self):
        
        super().__init__(
            'perception_node',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]    
        )

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

        raw_points = []
        # Prendo i punti della PointCloud e li filtro. Nel mio caso filtro solo il pavimento
        for p_data in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True): 
            if p_data[2] > 0.1: 
                raw_points.append([p_data[0], p_data[1], p_data[2]])
        
        if not raw_points:
            return

        # Pubblico la nuvola filtrata 
        cloud_msg = point_cloud2.create_cloud_xyz32(msg.header, raw_points)
        self.pub.publish(cloud_msg)
        
#L'OBIETTIVO E' FARE UN CLAUSTER AL FINE DI RICONOSCERE GLI OSTACOLI COSÌ DA FARE IL CONTROLLO RISPETTO AD UN 
# OSTACOLO PIUTTOSTO CHE DA CIASCUN PUNTO
           
class DBSCANNode(Node):
    
    def __init__(self):
        
        super().__init__(
            'dbscannode',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]    
        )

        self.eps = 0.5        # Distanza minima tra due punti
        self.min_points = 5   # Numero di punti minimo per definire un ostacolo
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0)) 
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cb_group = ReentrantCallbackGroup()
        self.pubGeometry = self.create_publisher(PointStamped, '/obstacles', 10)
        self.pubFlag = self.create_publisher(Bool, '/obstacle_detected',10)
        self.sub = self.create_subscription(
            PointCloud2,
            '/filtered_pc',
            self.dbscan_callback,
            10,
            callback_group=self.cb_group
        )
        
    def cluster_obstacles(self, valid_points_3d):
        
        if not valid_points_3d:
            return []        
            
        # Applico DBSCAN
        dbscan_instance = dbscan(valid_points_3d, self.eps, self.min_points)
        
        # Misuro il tempo di esecuzione
        time_execution, _ = timedcall(dbscan_instance.process)
        
        # 3. Mi restiruisce una lista di cui ciascun elemento contiene una lista di indici che indicano il raggruppamento di punti
        cluster_points = dbscan_instance.get_clusters()
        
        print(f"DBSCAN trovato {len(cluster_points)} ostacoli in {time_execution:.3f}s")
        
        # 4. Converto i cluster in ostacoli 3D
        obstacles_3d = []
        valid_obstacle = []
                            
        # Calcolo il centroide dell'ostacolo
        for indices in cluster_points:
            
            valid_cluster = [valid_points_3d[i] for i in indices]
            centroid = self.calculate_centroid(valid_cluster)
            obstacles_3d.append({
                'centroid': centroid,  
                'points': valid_cluster,  
                'size': len(valid_cluster),
                'distance': math.sqrt(centroid[0]**2 + centroid[1]**2)
            })
            
        for o in obstacles_3d:
            valid_obstacle.append(o['centroid'])
            
        return valid_obstacle
    
    def calculate_centroid(self, points):

        if not points:
            return None       
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points) 
        n = len(points)
        return [sum_x/n, sum_y/n]
    
    def get_largest_obstacle(self, obstacles):

        if not obstacles:
            return None
        return max(obstacles, key=lambda x: x['size'])
    
    def get_closest_obstacle(self, obstacles):

        if not obstacles:
            return None
        return min(obstacles, key=lambda x: x['distance'])

    def dbscan_callback(self, msg):
        
        # Prendo i punti della PointCloud filtrata e li sottocampiono per poi elaborarli
        points = [
            [float(p[0]), float(p[1])] 
            for p in point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        ]
        downsampled_points = [list(p) for p in points[::3]]
        
        if not points:
            return
        
        obstacles = self.cluster_obstacles(downsampled_points)
        detected = len(obstacles) > 0
        self.pubFlag.publish(Bool(data=detected))
        
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',       # target frame
                msg.header.frame_id,  # source frame
                rclpy.time.Time(seconds=0),
                #Time.from_msg(msg.header.stamp),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"TF non disponibile: {e}")
            return

        for o in obstacles:
            p = PointStamped()
            p.header = msg.header
            p.header.stamp = self.get_clock().now().to_msg()
            p.point.x = o[0]
            p.point.y = o[1]
            # Presi gli ostacoli calcolati nel frame della zed faccio il cambio di coordinate rispetto al telaio del robot e li pubblico
            p_chassis = tf2_geometry_msgs.do_transform_point(p, transform) 
            self.pubGeometry.publish(p_chassis)
            
class TeleopNode(Node):

    def __init__(self):
        
        super().__init__(
            'teleopnode',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        self.settings = termios.tcgetattr(sys.stdin) 
        self.cb_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Twist, '/cmd_vel_teleop', 10)
        self.target_lin = 0.0
        self.target_ang = 0.0
        self.ctrl_lin = 0.0
        self.ctrl_ang = 0.0
        self.timer = self.create_timer(0.01, self.teleop_callback, callback_group=self.cb_group)

    def destroy_node(self):
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        super().destroy_node()
        
    def teleop_callback(self):
        
        key = get_key(self.settings)

        if key == 'w':
            self.target_lin = check_linear_limit_velocity(self.target_lin + LIN_VEL_STEP_SIZE)
            print_vels(self.target_lin, self.target_ang)
        elif key == 'x':
            self.target_lin = check_linear_limit_velocity(self.target_lin - LIN_VEL_STEP_SIZE)
            print_vels(self.target_lin, self.target_ang)
        elif key == 'a':
            self.target_ang = check_angular_limit_velocity(self.target_ang + ANG_VEL_STEP_SIZE)
            print_vels(self.target_lin, self.target_ang)
        elif key == 'd':
            self.target_ang = check_angular_limit_velocity(self.target_ang - ANG_VEL_STEP_SIZE)
            print_vels(self.target_lin, self.target_ang)
        elif key == ' ' or key == 's':
            self.target_lin = self.target_ang = 0.0

        self.ctrl_lin = make_simple_profile(self.ctrl_lin, self.target_lin, LIN_VEL_STEP_SIZE / 2)
        self.ctrl_ang = make_simple_profile(self.ctrl_ang, self.target_ang, ANG_VEL_STEP_SIZE / 2)
        update_velocity(self.pub,ROS_DISTRO,self.ctrl_lin, self.ctrl_ang)
    
            
class APFNode(Node):
    
    def __init__(self):
        super().__init__(
            'apfnode',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )
        self.config = APF_CONFIG
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.ctrl_lin = 0.0
        self.ctrl_ang = 0.0
        self.obstacle = []
        self.history_total_force_x = []
        self.history_total_force_y = []
        self.cb_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Twist, '/cmd_vel_apf', 10)
        self.create_subscription(
            PointStamped, '/obstacles', self.apf_callback, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Twist, '/cmd_vel', self.apfVel_callback, 10, callback_group=self.cb_group
        )
        
    def compute_attractive_forces(self, v_lin):
        # ci sono 2 forze attratie quella diretta lungo x e quella lungo y, il robot si muove solo lungo x
        # quindi avremo una forza attrattiva lungo x proporzionale a v_lin e una lungo y sempre nulla
        # questo perchè stimo proiettanfdo le forze sugli assi del robot
        attractive_force = np.abs(v_lin * self.config['attractive_gain'])
        
        return attractive_force 
    
    def compute_repulsive_forces(self, x_dist, y_dist):

        gamma = 4
        eta_0 = 20
        eta_i = max(abs(x_dist), 0.1)
        #eta_i = max(abs( np.sqrt(x_dist**2 + y_dist**2) ), 0.1)
        #dist_min = abs(eta_0 - eta_i)
        repulsive_force_x = (self.config['repulsive_gain']/eta_i**2) * ((1/eta_i - 1/eta_0)**(gamma - 1))
        repulsive_force_y = y_dist
        repulsive_force = [repulsive_force_x, repulsive_force_y]
        return repulsive_force

    def compute_total_force(self, control_linear_velocity, distance):
        
        if len(distance) < 3:
            self.get_logger().error("Dati ostacolo incompleti (possibile buffer overflow in Sim)")
            return [0.0, 0.0, 0.0]
        
        #angle = np.arctan2 (distance[1] , distance[0])
        print(f"[OSTACOLO] x={distance[0]:.2f}, y={distance[1]:.2f}, z={distance[2]:.2f}°")
        attractive_force = self.compute_attractive_forces (control_linear_velocity)
        repulsive_force = self.compute_repulsive_forces(distance[0], distance[1] )
        total_force_x =  float(attractive_force) - float(repulsive_force[0])
        total_force_y = 0.0 - repulsive_force[1] # 0 è la forza attrattiva lungo y
        self.history_total_force_x.append(float(total_force_x))
        self.history_total_force_y.append(float(total_force_y))

        return [ total_force_x, total_force_y ]
    
    def apfVel_callback(self, msg):
        
        self.lin_vel = msg.linear.x 
        self.ang_vel = msg.angular.z
    
    def apf_callback(self,msg):
        
        self.obstacle = [msg.point.x, msg.point.y, msg.point.z]
        [total_force_x, total_force_y] = self.compute_total_force( self.lin_vel, self.obstacle)
        if total_force_x < 0.0:
            apf_lin_vel = self.lin_vel + (total_force_x * self.config['LIN_VELOCITY_GAIN'])
        else:
            apf_lin_vel = self.lin_vel
        apf_ang_vel = (total_force_y* self.config['ANG_VELOCITY_GAIN'])
        #np.sign(msg.point.y)*
        update_velocity(self.pub,ROS_DISTRO,apf_lin_vel, apf_ang_vel)

        
class VelocityMuxNode(Node):
    
    def __init__(self):
        
        super().__init__('velocitymuxnode')
        self.LinearTeleop = 0.0
        self.AngularTeleop = 0.0
        self.LinearApf = 0.0
        self.AngularApf = 0.0
        self.obstacleDetected = False
        self.history_lin = [0.0]
        self.history_ang = [0.0]
        #self.history_rep = []
        self.last_obstacle_time = self.get_clock().now()
        self.cb_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            Twist, '/cmd_vel_teleop', self.mux_teleop_callback, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Twist, '/cmd_vel_apf', self.mux_apf_callback, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Bool, '/obstacle_detected', self.obstacle_flag_callback, 10, callback_group=self.cb_group
        )
        
        self.create_timer(0.1, self.vel_publish)
        
    def obstacle_flag_callback(self, msg):
        
        self.obstacleDetected = msg.data
        self.last_obstacle_time = self.get_clock().now()
        
    def vel_publish(self):

        now = self.get_clock().now()
        time_diff = (now - self.last_obstacle_time).nanoseconds / 1e9
        twist = Twist()
        twist_published =  Twist()
        
        if self.obstacleDetected and time_diff < 0.5:
            twist.linear.x = self.LinearApf
            twist.angular.z = self.AngularApf
            print_vels(self.LinearApf, self.AngularApf)
        else:
            twist.linear.x = self.LinearTeleop
            twist.angular.z = self.AngularTeleop

        twist_published.linear.x = make_simple_profile(self.history_lin[-1], twist.linear.x, LIN_VEL_STEP_SIZE)
        twist_published.angular.z = make_simple_profile(self.history_ang[-1], twist.angular.z, ANG_VEL_STEP_SIZE)
        #update_velocity(self.pub,ROS_DISTRO,self.ctrl_lin, self.ctrl_ang)
        self.history_lin.append(float(twist_published.linear.x))
        self.history_ang.append(float(twist_published.angular.z))
        #self.history_rep.append(APFNode().history_rep)
        self.pub.publish(twist_published)
            
    def mux_teleop_callback (self,msg):
        
        self.LinearTeleop = msg.linear.x
        self.AngularTeleop = msg.angular.z
        
    def mux_apf_callback (self,msg):
        
        self.LinearApf = msg.linear.x
        self.AngularApf = msg.angular.z
            
def main(args=None):
    
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=5)
    p_node = PerceptionNode ()
    p_node.tf_buffer = Buffer()
    p_node.tf_listener = TransformListener(p_node.tf_buffer, p_node)
    mux_node = VelocityMuxNode()
    apf_node = APFNode()
    teleop_node = TeleopNode()
    dbscan_node = DBSCANNode()

    nodes = [
        p_node,
        dbscan_node,
        teleop_node,
        apf_node,
        mux_node
    ]

    for n in nodes:
        executor.add_node(n)

    try:
        executor.spin()
        
    except KeyboardInterrupt:
        
        # SALVATAGGIO VELOCITÀ 
        try:
            file_vel = 'velocita_simulazione.csv'
            h_lin = mux_node.history_lin
            h_ang = mux_node.history_ang
            
            with open(file_vel, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'linear_x', 'angular_z'])
                for i in range(len(h_lin)):
                    writer.writerow([i, float(h_lin[i]), float(h_ang[i])])
            print(f"File velocità salvato: {file_vel} ({len(h_lin)} campioni)")
            print(f"Ultimo valore angolare: {h_ang[-1]}") # Per tua verifica
        except Exception as e:
            print(f"Errore salvataggio velocità: {e}")

        # SALVATAGGIO FORZE 
        try:
            file_force = 'forza_simulazione.csv'
            h_force_x = apf_node.history_total_force_x
            h_force_y = apf_node.history_total_force_y
   
            with open(file_force, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'total_force', 'total_force_y'])
                for i in range(len(h_force_x)):
                    writer.writerow([i, float(h_force_x[i]), float(h_force_y[i])])
            print(f"File forza salvato: {file_force} ({len(h_force_x)} campioni)")
        except Exception as e:
            print(f"Errore salvataggio forza: {e}")
        executor.shutdown()
        for n in nodes:
            executor.remove_node(n)
            teleop_node.destroy_node()
            
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
        
 