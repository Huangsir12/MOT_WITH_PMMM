import argparse
from multi_object_tracking.app_tracking import  TrackObjects


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default="yolov8n", help='select from yolov8n, yolov8l, yolov11l')
    parser.add_argument('--video_input', type=str, default="video_datasets/door1.mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--tracking_model', type=str, default="BotSort", help='select from BotSort, StrongSort, ByteTrack')
    parser.add_argument('--output_dir', type=str, default="track_results", help='output folder')  # output folder
    parser.add_argument('--draw_tracks_line', type=bool, default=False, help='if draw tracks line')
    parser.add_argument('--trackor_conf', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--trackor_iou', type=float, default=0.5, help='iou threshold')
    parser.add_argument(
        '--reid_config_file', type=str, default='configs/bpbreid/bpbreid_inference.yaml', help='path to config file'
    )

    opt = parser.parse_args()
    trackor = TrackObjects(opt.detect_model, opt.tracking_model, opt.reid_config_file, opt.output_dir)
    trackor.track_objects_with_reid(opt.video_input, 
                               opt.draw_tracks_line, 
                               opt.trackor_conf, 
                               opt.trackor_iou)