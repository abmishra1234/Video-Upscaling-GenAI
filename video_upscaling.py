import cv2
import numpy as np
import logging
import json
import os
import gc
import psutil  # For system memory monitoring
from datetime import datetime
from moviepy.editor import VideoFileClip
import sys

# Load configuration from JSON
def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise Exception(f"Configuration file {config_file} not found.")
    except json.JSONDecodeError:
        raise Exception("Error parsing the JSON configuration file.")

# Configure Logging
def configure_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{current_time}_upscaling.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized")

# Memory check function to prevent overload
def check_memory_limit():
    mem = psutil.virtual_memory()
    if mem.available < (0.1 * mem.total):  # If available memory is less than 10% of total memory
        logging.warning("Low memory detected. Available memory is below 10%. Consider halting the process.")
        raise MemoryError("System is running out of memory. Aborting process.")

# Map interpolation method from config
def get_interpolation_method(method_name):
    if method_name == "INTER_LINEAR":
        return cv2.INTER_LINEAR
    elif method_name == "INTER_NEAREST":
        return cv2.INTER_NEAREST
    elif method_name == "INTER_CUBIC":
        return cv2.INTER_CUBIC
    elif method_name == "INTER_LANCZOS4":
        return cv2.INTER_LANCZOS4
    else:
        raise ValueError(f"Invalid interpolation method: {method_name}")

# Video Upscaling Function with Chunking for Memory Management
def upscale_video(input_path, temp_output_path, target_resolution, interpolation_method, chunk_size=100):
    try:
        video_capture = cv2.VideoCapture(input_path)
        if not video_capture.isOpened():
            raise Exception(f"Unable to open video file: {input_path}")
        
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Original Video Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, tuple(target_resolution))

        logging.info(f"Upscaling video to resolution: {target_resolution}")

        frame_counter = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            upscaled_frame = cv2.resize(frame, tuple(target_resolution), interpolation=interpolation_method)
            video_writer.write(upscaled_frame)

            frame_counter += 1
            if frame_counter % chunk_size == 0:
                logging.info(f"Processed {frame_counter} frames out of {total_frames}.")
                check_memory_limit()  # Check memory after processing a chunk
                gc.collect()  # Trigger garbage collection to free memory

        video_capture.release()
        video_writer.release()
        logging.info("Video upscaling completed successfully.")
    
    except MemoryError as e:
        logging.exception(f"Memory error encountered: {str(e)}. Consider lowering resolution or chunk size.")
        raise e
    except Exception as e:
        logging.exception(f"Error occurred while processing video: {str(e)}")
        raise e

# Function to extract and add audio to the upscaled video
def extract_and_add_audio(input_path, temp_upscaled_video, final_output_video):
    try:
        original_clip = VideoFileClip(input_path)
        audio_clip = original_clip.audio
        upscaled_clip = VideoFileClip(temp_upscaled_video)
        final_clip = upscaled_clip.set_audio(audio_clip)

        final_clip.write_videofile(final_output_video, codec='libx264')
        logging.info("Audio extracted and added to the upscaled video successfully.")
    except Exception as e:
        logging.exception(f"Error occurred while adding audio: {str(e)}")
        raise e

# Function to clean up temporary files
def cleanup_temp_file(temp_file_path):
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Temporary file {temp_file_path} removed successfully.")
        else:
            logging.warning(f"Temporary file {temp_file_path} does not exist.")
    except Exception as e:
        logging.exception(f"Error occurred during cleanup: {str(e)}")

# Main function
def main():
    try:
        config = load_config('config.json')
        
        configure_logging(config['log_dir'])

        interpolation_method = get_interpolation_method(config['interpolation_method'])

        temp_output_path = os.path.join(config['output_folder'], 'temp_upscaled_video.mp4')
        upscale_video(config['input_video'], temp_output_path, config['target_resolution'], interpolation_method)

        extract_and_add_audio(config['input_video'], temp_output_path, config['output_video'])

        cleanup_temp_file(temp_output_path)
        
    except MemoryError as e:
        logging.error(f"Memory error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Failed to complete video upscaling: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
