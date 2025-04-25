import os
import json
import socket

from sparse_frontier.utils.globals import GlobalSettings


def get_free_ports(n: int) -> list[int]:
    """Find N free ports on the local machine."""
    free_ports = []
    sockets = []
    
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', 0))  # Bind to an available port
            free_ports.append(s.getsockname()[1])  # Get the assigned port number
            sockets.append(s)  # Keep the socket open to reserve the port
    finally:
        # Close all sockets to release the ports
        for s in sockets:
            s.close()
    
    return free_ports


def get_latest_commit_id():
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return None


def save_config(dir_path: str):
    from omegaconf import OmegaConf
    from datetime import datetime
    
    cfg = GlobalSettings.get('cfg')

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict['commit_id'] = get_latest_commit_id()
    
    # Add timestamp
    config_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path = os.path.join(dir_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
