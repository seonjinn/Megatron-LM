import argparse
from datetime import datetime
import socket
import subprocess


def run_cmd(cmd_str):
    proc = subprocess.Popen(cmd_str, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.read().decode()
    return out


def main(args):
    times_cmd = f"sacct -X -n --format=start,end --parsable -j {args.job_id}"
    times = run_cmd(times_cmd)

    start, end, _ = times.split("|")

    if start == "Unknown":
        start = None
    else:
        start = datetime.fromisoformat(start).timestamp()

    if end == "Unknown":
        end = None
    else:
        end = datetime.fromisoformat(end).timestamp()

    nodes_cmd = f'scontrol show hostnames `sacct -X -n --format=nodelist --parsable2 -j {args.job_id}`'
    nodelist = run_cmd(nodes_cmd)
    nodelist = nodelist.split("\n")

    cluster = "cw-dfw-cs-001"
    hostname = socket.gethostname()
    if "draco-oci" in hostname:
        cluster = "draco-oci-iad"
    elif "oci-nrt" in hostname:
        cluster = "oci-nrt-cs-001"

    base_url = f"https://dashboards.telemetry.dgxc.ngc.nvidia.com/d/rYdddlPWk_cw_dfw_05_15_2024/node-health?orgId=8var-gpu=$__all"

    base_url += f"&var-cluster={cluster}"

    node_url_str = ""
    for n in nodelist:
        if len(n) > 0:
            node_url_str += f"&var-node={n}"

    time_str = ""
    if start is not None:
        time_str += f"&from={int(start) * 1000}"
    if end is not None:
        time_str += f"&to={int(end) * 1000}"

    url = base_url + node_url_str + time_str
    print(url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gets grafana url from job id")
    parser.add_argument('job_id')

    args = parser.parse_args()
    main(args)
