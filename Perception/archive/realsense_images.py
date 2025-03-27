import pyrealsense2 as rs
import cv2
import numpy as np


def capture_video(output_filename='output.mp4', codec='mp4v', fps=20.0, frame_size=(848, 480)):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.array(color_frame.get_data())

            out.write(color_image)

            cv2.imshow('Video Capture', color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release everything when job is finished
        cv2.destroyAllWindows()
        pipeline.stop()
        out.release()

    

# Example usage
capture_video()