import numpy as np
from dataset.data import Sequence, SequenceList
from tools.args_temp import args


def VOTRGBTDataset():
    return VOTRGBTDatasetClass().get_sequence_list()


class VOTRGBTDatasetClass():
    """VOT2019 RGB-T dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = args.votrgbt_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 5
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames_ir = ['{base_path}/{sequence_path}/ir/{frame:0{nz}}i.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        frames_rgb = ['{base_path}/{sequence_path}/color/{frame:0{nz}}v.{ext}'.format(base_path=self.base_path,
                    sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                     for frame_num in range(start_frame, end_frame + 1)]

        # test = [f.replace('ir', 'color') for f in frames]
        # test = [f.replace('i.jpg', 'v.jpg') for f in test]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence(sequence_name, frames_ir, frames_rgb, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_path = self.base_path + 'list.txt'
        sequence_list = np.loadtxt(str(list_path), dtype=np.str)
        sequence_list = sequence_list.tolist()

        return sequence_list
