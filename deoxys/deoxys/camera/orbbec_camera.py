from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2
from pyorbbecsdk import *

# Camera intrinsics - you may need to update these values for your Orbbec camera
serial_number_to_cam_intr = {
    # Add your Orbbec camera serial numbers and intrinsics here
    # Example format:
    # "SERIAL_NUMBER": {"fx": 000.0, "fy": 000.0, "px": 000.0, "py": 000.0},
}

def frame_to_bgr_image(frame):
    """Convert frame data to BGR image."""
    data = frame.get_data()
    width = frame.get_width()
    height = frame.get_height()
    format = frame.get_format()
    
    # Print debug information
    # print(f"{format}, {width}x{height}, Data size: {len(data)}")
    
    if format == OBFormat.BGR:
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    elif format == OBFormat.RGB:
        rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif format == OBFormat.YUYV:
        yuyv = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 2))
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        return bgr
    elif format == OBFormat.MJPG:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unsupported format: {format}")

class Camera:
    def __init__(self, serial_number: Optional[str]=None, camera_width: int=640, camera_height: int=480, camera_fps: int=30,
                 auto_white_balance: bool=False, white_balance_temp: int=4500) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        
        # Initialize Orbbec camera
        self.context = Context()
        device_list = self.context.query_devices()
        
        if device_list.get_count() == 0:
            raise RuntimeError("No device connected")
            
        # Find device with matching serial number or use the first one
        self.device = None
        if serial_number is not None:
            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                device_info = device.get_device_info()
                if device_info.get_serial_number() == serial_number:
                    self.device = device
                    break
        if self.device is None:
            self.device = device_list.get_device_by_index(0)
            
        self.pipeline = Pipeline(self.device)
        self.config = Config()
        
        # Configure streams using profile lists
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            
            # Try to get the profile with our desired resolution
            try:
                # Try MJPG format first as it's commonly supported
                matching_profile = profile_list.get_video_stream_profile(
                    self.camera_width,
                    self.camera_height,
                    OBFormat.MJPG,
                    self.camera_fps
                )
                print(f"Found matching profile: MJPG, {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
            except Exception as e:
                print(f"Could not get MJPG profile: {e}")
                matching_profile = None
            
            if matching_profile is not None:
                color_profile = matching_profile
            else:
                print(f"Warning: Could not find profile with resolution {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
                print("Using default profile instead")
                color_profile = profile_list.get_default_video_stream_profile()
            
            # Print selected profile information
            print(f"Selected color profile: {color_profile.get_format()}, {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")
            self.config.enable_stream(color_profile)
            self.has_color = True
        except OBError as e:
            print(f"Color sensor not available: {e}")
            self.has_color = False
            
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        # Print depth profile information
        print(f"Depth profile: {depth_profile.get_format()}, {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")
        self.config.enable_stream(depth_profile)
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Configure camera properties
        try:
            # Try to set camera properties
            try:
                # Try to set white balance properties
                try:
                    # Try to set white balance
                    if not auto_white_balance:
                        # First try to disable auto white balance
                        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, False)
                        print("Auto white balance disabled")
                        
                        # Then try to set manual white balance
                        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, white_balance_temp)
                        print(f"White balance temperature set to {white_balance_temp}K")
                    else:
                        # Enable auto white balance
                        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, True)
                        print("Auto white balance enabled")
                    
                    # Print current white balance settings
                    try:
                        auto_wb = self.device.get_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL)
                        wb_temp = self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT)
                        print(f"Current white balance settings: auto={auto_wb}, temp={wb_temp}K")
                    except Exception as e:
                        print(f"Failed to get current white balance settings: {e}")
                except Exception as e:
                    print(f"Failed to set white balance: {e}")
                    
            except Exception as e:
                print(f"Failed to set white balance: {e}")
        except Exception as e:
            print(f"Warning: Could not configure camera properties: {e}")
        
        # Get first frames to get depth scale
        frames = self.pipeline.wait_for_frames(1000)  # 1000ms timeout
        if frames is None:
            print("Warning: Could not get initial frames, using default depth scale")
            self.depth_scale = 0.001  # default value in meters
        else:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                self.depth_scale = depth_frame.get_depth_scale()
                print("Depth Scale is: ", self.depth_scale)
            else:
                self.depth_scale = 0.001  # default value in meters
                print("Warning: Could not get depth scale, using default:", self.depth_scale)

    def close(self):
        self.pipeline.stop()

    def get_frame(self, filter=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            filter bool optional Whether to apply filters to depth frames. Defaults to True.
        Returns:
            color_image np.ndarray shape=(H, W, 3) Color image(BGR)
            depth_image np.ndarray shape=(H, W) Depth image in meters
        """
        frames = self.pipeline.wait_for_frames(1000)  # 1000ms timeout
        if frames is None:
            print("Warning: Could not get frames")
            return None, None
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        color_image = None
        if color_frame is not None:
            try:
                color_image = frame_to_bgr_image(color_frame)
            except Exception as e:
                print(f"Error converting color frame: {e}")
                color_image = None
        
        depth_image = None
        if depth_frame is not None:
            try:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                depth_image = depth_image.astype(np.float32) * self.depth_scale
                
                if filter:
                    # Apply basic filtering - you can customize this based on your needs
                    depth_image = cv2.medianBlur(depth_image.astype(np.float32), 5)
            except Exception as e:
                print(f"Error converting depth frame: {e}")
                depth_image = None
        
        return color_image, depth_image

    def get_camera_intrinsics(self, use_raw=False, serial_number: Optional[str]=None):
        """
        Args:
            use_raw bool optional Whether to use the camera intrinsics from the raw stream. Defaults to False.
        Returns:
            {
                "fx": Focal length x,
                "fy": Focal length y,
                "px": Principal point x,
                "py": Principal point y
            }
        """
        if use_raw:
            try:
                profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                intrinsics = color_profile.get_intrinsic_parameters()
                return {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "px": intrinsics.px,
                    "py": intrinsics.py
                }
            except Exception as e:
                print(f"Warning: Could not get intrinsics from camera: {e}")
                
        if serial_number in serial_number_to_cam_intr:
            return serial_number_to_cam_intr[serial_number]
        else:
            raise ValueError(f"No intrinsics available for camera {serial_number}")

class MultiCamera:
    def __init__(self, serial_numbers: Optional[List[str]]=None, camera_width: int=640, camera_height: int=480, camera_fps: int=30,
                 auto_white_balance: bool=False, white_balance_temp: int=4500) -> None:
        # Initialize context and get all available devices
        context = Context()
        device_list = context.query_devices()
        
        if device_list.get_count() == 0:
            raise RuntimeError("No devices connected")
            
        # Get all available serial numbers
        all_serial_numbers = []
        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            device_info = device.get_device_info()
            serial_number = device_info.get_serial_number()
            print('Found device:', device_info.get_name(), serial_number)
            all_serial_numbers.append(serial_number)

        if serial_numbers is None:
            serial_numbers = all_serial_numbers
        else:
            serial_numbers = [sn for sn in serial_numbers if sn in all_serial_numbers]

        if not serial_numbers:
            raise RuntimeError("No matching devices found")

        print("Using cameras with serial numbers: ", serial_numbers)
            
        self.cameras = {
            serial_number: Camera(serial_number, camera_width, camera_height, camera_fps,
                                     auto_white_balance=auto_white_balance, white_balance_temp=white_balance_temp)
            for serial_number in serial_numbers
        }

        # Warm up cameras
        for _ in range(20):
            print("Warming up cameras...")
            self.get_frame()

    def close(self):
        for camera in self.cameras.values():
            camera.close()

    def get_frame(self, serial_numbers: Optional[List[str]]=None, filter=True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if serial_numbers is None:
            serial_numbers = list(self.cameras.keys())
        return {
            serial_number: self.cameras[serial_number].get_frame(filter)
            for serial_number in serial_numbers
        }
    
    def get_camera_intrinsics(self, serial_numbers: Optional[List[str]]=None, use_raw=False) -> Dict[str, Dict[str, float]]:
        if serial_numbers is None:
            serial_numbers = self.cameras.keys()
        return {
            serial_number: self.cameras[serial_number].get_camera_intrinsics(use_raw, serial_number)
            for serial_number in serial_numbers
        }