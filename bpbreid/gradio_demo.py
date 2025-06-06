import gradio as gr
from multi_object_tracking.app_tracking import track_objects_init, track_objects


with gr.Blocks() as demo:
    gr.Markdown(f"# Multi-Camera Person Re-Identification")
    gr.Markdown(f"# Fist step, relize multi-object tracking in each camera")
    with gr.Row():
        with gr.Column():
            gr.Markdown("model and parameters")
            detect_model  = gr.Dropdown(label="目标检测模型", choices=["yolov8n", "yolov8l", "yolov11l"], value="yolov8l")
            tracking_model  = gr.Dropdown(label="目标跟踪模型", choices=["BotSort", "StrongSort",  "ByteTrack"], value="BotSort")
            conf = gr.Slider(label="置信度阈值", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
            iou = gr.Slider(label="iou阈值", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
            draw_tracks_line = gr.Checkbox(label="是否绘制轨迹线", value=False)

        with gr.Column():
            gr.Markdown("video input")
            video_input = gr.Video(sources=["upload", "webcam"], label="上传MP4视频或调取摄像头")
            run_button = gr.Button("Run")
        
        with gr.Column():
            gr.Markdown("video output")
            video_output = gr.Video(label="输出视频", show_download_button=True, format="mp4")
    

    run_button.click(track_objects, inputs=[detect_model, video_input, tracking_model, draw_tracks_line, conf, iou], outputs=[video_output])

    gr.Markdown(f"# Second step, relize multi-camera person re-identification")


# start Gradio Inference
demo.launch(debug=True,
            share=True)