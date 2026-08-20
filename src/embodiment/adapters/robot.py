import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class RobotAdapterError(RuntimeError):
    pass


class RobotAdapter(ABC):
    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def move_joint(self, joint: str, position: float, velocity: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_gripper(
        self,
        state: str,
        width: Optional[float] = None,
        force: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RosCliRobotAdapter(RobotAdapter):
    def __init__(
        self,
        ros_version: Optional[str] = None,
        command_topic: Optional[str] = None,
        state_topic: Optional[str] = None,
        ros2_bin: Optional[str] = None,
        rostopic_bin: Optional[str] = None,
        runner=None,
    ):
        self._ros_version = str(ros_version or os.environ.get("ROS_VERSION") or "").strip()
        self._command_topic = (
            str(command_topic or os.environ.get("EMBODIMENT_ROBOT_COMMAND_TOPIC") or "").strip()
            or "/embodiment/command"
        )
        self._state_topic = (
            str(state_topic or os.environ.get("EMBODIMENT_ROBOT_STATE_TOPIC") or "").strip()
            or "/embodiment/state"
        )
        self._ros2_bin = ros2_bin or shutil.which("ros2")
        self._rostopic_bin = rostopic_bin or shutil.which("rostopic")
        self._runner = runner or subprocess.run

    @staticmethod
    def is_detected() -> bool:
        return bool(
            os.environ.get("ROS_VERSION")
            or os.environ.get("ROS_MASTER_URI")
            or os.environ.get("ROS_DOMAIN_ID")
            or shutil.which("ros2")
            or shutil.which("rostopic")
        )

    def _cli(self) -> Optional[str]:
        if self._ros_version == "2" and self._ros2_bin:
            return "ros2"
        if self._ros_version == "1" and self._rostopic_bin:
            return "rostopic"
        if self._ros2_bin:
            return "ros2"
        if self._rostopic_bin:
            return "rostopic"
        return None

    def describe(self) -> Dict[str, Any]:
        cli = self._cli()
        ros_version = self._ros_version or ("2" if cli == "ros2" else "1" if cli == "rostopic" else None)
        detected = bool(ros_version or os.environ.get("ROS_MASTER_URI") or os.environ.get("ROS_DOMAIN_ID"))
        if not detected and not cli:
            return {"detected": False}
        return {
            "detected": True,
            "transport": "ros",
            "ros_version": ros_version,
            "ros_master_uri": os.environ.get("ROS_MASTER_URI"),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "command_topic": self._command_topic,
            "state_topic": self._state_topic,
            "command_bridge_available": bool(cli),
            "cli": cli,
        }

    def get_state(self) -> Dict[str, Any]:
        state = self.describe()
        if not state.get("detected"):
            return state
        state["available_commands"] = ["move_joint", "set_gripper"] if state.get("command_bridge_available") else []
        return state

    def move_joint(self, joint: str, position: float, velocity: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "op": "move_joint",
            "joint": joint,
            "position": float(position),
        }
        if velocity is not None:
            payload["velocity"] = float(velocity)
        self._publish_command(payload)
        return {
            "queued": True,
            "joint": joint,
            "position": float(position),
            "velocity": float(velocity) if velocity is not None else None,
        }

    def set_gripper(
        self,
        state: str,
        width: Optional[float] = None,
        force: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"op": "set_gripper", "state": state}
        if width is not None:
            payload["width"] = float(width)
        if force is not None:
            payload["force"] = float(force)
        self._publish_command(payload)
        result = {"queued": True, "state": state}
        if width is not None:
            result["width"] = float(width)
        if force is not None:
            result["force"] = float(force)
        return result

    def _publish_command(self, payload: Dict[str, Any]) -> None:
        cli = self._cli()
        if not cli:
            raise RobotAdapterError("ROS command bridge is not available")
        payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        try:
            if cli == "ros2":
                message = "{data: " + json.dumps(payload_json) + "}"
                self._runner(
                    [self._ros2_bin, "topic", "pub", "--once", self._command_topic, "std_msgs/msg/String", message],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=True,
                )
            else:
                message = "data: " + json.dumps(payload_json)
                self._runner(
                    [self._rostopic_bin, "pub", "-1", self._command_topic, "std_msgs/String", message],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=True,
                )
        except FileNotFoundError as e:
            raise RobotAdapterError(f"ROS CLI not available: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RobotAdapterError("ROS command publish timed out") from e
        except subprocess.CalledProcessError as e:
            stderr = str(e.stderr or "").strip() or str(e)
            raise RobotAdapterError(stderr) from e
