#
# File: collect.py
# Desc: Manages the data collection process by launching and watching the carla server and collection client.
# Auth: Serwan Jassim
#
# Copyright © 2022-present Saivvy. All rights reserved.
#
##########################################
import subprocess
import os
import time
import psutil
import argparse
import multiprocessing
import re
from main import _multi_agent_main as ppo_multi_agent_main
import hydra
import sys


class CarlaManager:
    """
    This manager launches the carla server and the data collection client and collects as many scenes as specified.
    If either the server or the client crash, this manager will detect it after some time, tear down all processes down
    and launch them again to collect the remaining number of scenes.
    """
    def __init__(self, args: argparse.Namespace):
        self.carla_location = args.carla_location
        self.render = args.render
        self.carla_instance = args.carla_instance
        self.carla_port = 2100 + 10 * self.carla_instance
        self.client_process = None

    def launch_server(self):
        """
        Launches the server process.
        """
        subprocess.Popen(
            f'{self.carla_location} -RenderOffScreen -carla-port={self.carla_port}', # -graphicsadapter=0
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w")
        )

    def launch_client(self):
        """
        Launches the client process.
        """
        multiprocessing.set_start_method('spawn', force=True)
        self.client_process = multiprocessing.Process(target=ppo_multi_agent_main, args=([self.carla_instance]))
        self.client_process.start()

    def server_is_live(self):
        try:
            txt = subprocess.check_output("lsof -nP -iTCP -sTCP:LISTEN | grep CarlaUE4", shell=True)
            ports_info = txt.decode("utf-8").splitlines()
            for port_info in ports_info:
                port = re.findall(r"TCP \*:(\d+) ", port_info)[0]
                if port == str(self.carla_port):
                    return True
                
            return False
        except subprocess.CalledProcessError:
            return False

    def get_pid_from_port(self):
        try:
            txt = subprocess.check_output("lsof -nP -iTCP -sTCP:LISTEN | grep CarlaUE4", shell=True)
            ports_info = txt.decode("utf-8").splitlines()
            for port_info in ports_info:
                if str(self.carla_port) in port_info:
                    return re.findall(r"CarlaUE4-\s* (\d+) ", port_info)[0]
        except subprocess.CalledProcessError:
            return None

    def launch(self):
        """
        Launch both the server and the client. Include a short delay between server and client launch to make sure
        server is running.
        """
        print(f"Launching server on port {self.carla_port}")
        self.launch_server()

        while not self.server_is_live():
            time.sleep(5)

        print("Launching client")
        self.launch_client()

    def tear_down(self):
        """
        Tear down the server and client processes.
        """
        # process the client process that is stored in instance variable
        if self.client_process is not None:
            print("Killed client process")
            self.client_process.terminate()

        # server starts multiple processes so it requires iteration
        server_pid = self.get_pid_from_port()
        if server_pid is not None:
            print("Killed server process")
            psutil.Process(int(server_pid)).kill()

        time.sleep(2)

    def has_crashed(self, data_dir: str) -> bool:
        return self.server_is_live()

    def manage(self):
        self.launch()
        while True:
            time.sleep(30)
            if not self.server_is_live():
                print("Restarting server ...")	
                self.tear_down()
                self.launch_server()
                while not self.server_is_live():
                    time.sleep(5)
            if not self.client_process.is_alive():
                print("Client terminated! Exiting...")
                self.tear_down()
                return

def main():
    #os.environ["HYDRA_FULL_ERROR"] = "1"
    
    # p = argparse.ArgumentParser(description="Carla server parameters.")
    # p.add_argument("--carla_instance", type=int, default=0,
    #                     help="Carla instance index. Decides which ports are used")
    # p.add_argument('--carla_location', type=str, default="/opt/carla-simulator/CarlaUE4.sh",
    #                     help="Location of carla startup bash script")
    # p.add_argument("--render", action='store_true',
    #                     help="Whether to render the server on screen (False by default).")

    # args = p.parse_args()
    
    args = argparse.Namespace()
    args.carla_instance = 0
    # print("Command line arguments:")
    # for arg in sys.argv[1:]:
    #     print("  " + arg)
    #     param, value = arg.split('=')
    #     if param == "env.train.id":
    #         args.carla_instance = int(value)
    args.carla_location = "/opt/carla-simulator/CarlaUE4.sh"
    args.render = False
    manager = CarlaManager(args)
    
    try:
        manager.manage()
    except KeyboardInterrupt:
        manager.tear_down()
    except Exception as e:
        manager.tear_down()
        import traceback
        print(traceback.format_exc())
        print(e)

if __name__ == '__main__':
    main()
    
