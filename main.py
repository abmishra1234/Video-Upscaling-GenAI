import cv2
import moviepy.editor as mp

def upscale_video_with_audio(input_path, output_path, scale_factor):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (int(width * scale_factor), int(height * scale_factor)))

    # Read and upscale frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Upscale the frame
        upscaled_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)

        # Write the upscaled frame to the output video
        out.write(upscaled_frame)

    # Release video objects
    cap.release()
    out.release()

    # Extract audio using moviepy
    video_clip = mp.VideoFileClip(input_path)
    audio_clip = video_clip.audio

    # Combine upscaled video with original audio
    upscaled_video = mp.VideoFileClip("temp_video.mp4")
    final_clip = upscaled_video.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264")

    print(f"Upscaled video with audio saved at {output_path}")

# Usage example
input_video_path = 'input_video.mp4'
output_video_path = 'upscaled_video_with_audio.mp4'
scale_factor = 2  # Scale by 2x

upscale_video_with_audio(input_video_path, output_video_path, scale_factor)
