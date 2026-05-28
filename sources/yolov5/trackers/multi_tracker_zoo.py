from ...yolov5.trackers.strong_sort.utils.parser import get_config
# from gsan.yolov5.trackers.bytetrack.byte_tracker import BYTETracker
from ...yolov5.trackers.ocsort.ocsort import OCSort


def create_tracker(tracker_type, appearance_descriptor_weights, device, half):
    # if tracker_type == 'bytetrack':
    #     bytetracker = BYTETracker(
    #         track_thresh=0.6,
    #         track_buffer=30,
    #         match_thresh=0.8,
    #         frame_rate=30
    #     )
    #     return bytetracker
    # elif tracker_type == 'ocsort':
    #     ocsort = OCSort(
    #         det_thresh=0.45,
    #         iou_threshold=0.2,
    #         use_byte=False
    #     )
    #     return ocsort
    # else:
    #     print('No such tracker')
    #     exit()
        
    if tracker_type == 'ocsort':
        ocsort = OCSort(
            det_thresh=0.45,
            iou_threshold=0.3,
            use_byte=False
        )
        return ocsort
    else:
        print('No such tracker')
        exit()