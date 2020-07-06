
class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames_infrared, frames_color, ground_truth_rect, object_class=None):
        self.name = name
        self.frames_infrared = frames_infrared
        self.frames_color = frames_color
        self.ground_truth_rect = ground_truth_rect
        self.init_state = list(self.ground_truth_rect[0,:])
        self.object_class = object_class


class SequenceList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Sequence name not in the dataset.')
        elif isinstance(item, int):
            return super(SequenceList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return SequenceList([super(SequenceList, self).__getitem__(i) for i in item])
        else:
            return SequenceList(super(SequenceList, self).__getitem__(item))

    def __add__(self, other):
        return SequenceList(super(SequenceList, self).__add__(other))

    def copy(self):
        return SequenceList(super(SequenceList, self).copy())