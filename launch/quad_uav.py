import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Common boundary args
    boundary_args = [
        '--xmin', '0.0',
        '--ymin', '0.0',
        '--xmax', '50.0',
        '--ymax', '40.0'
    ]

    # Four agents with distinct IDs and start positions
    agents = [
        {'id': 1, 'x':  0.0, 'y':  0.0},
        {'id': 2, 'x': 10.0, 'y':  0.0},
        {'id': 3, 'x':  0.0, 'y': 10.0},
        {'id': 4, 'x': 5.0, 'y': 3.0},
    ]

    ld = LaunchDescription()

    for ag in agents:
        ld.add_action(
            Node(
                package='drone',
                executable='uav_agent',
                name=f'uav_agent_{ag["id"]}',
                output='screen',
                arguments=[
                    '--agent_id', str(ag['id']),
                    '--posx',    str(ag['x']),
                    '--posy',    str(ag['y']),
                ] + boundary_args
            )
        )

    return ld
