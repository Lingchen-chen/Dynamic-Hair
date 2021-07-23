from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--test_video_dir', type=str, default='G:/MyPaper/TestData/video302')
        self.parser.add_argument('--test_save_dir', type=str, default='results', help='save dir in each case dir')
        self.parser.add_argument('--test_start_frame', type=int, default=102, help='the start frame to test')
        self.parser.add_argument('--test_refine_iters', type=int, default=2, help='the refine iters for enforcing temporal coherence')
        self.parser.add_argument('--test_frames', type=int, default=50, help='# of frames to test')
        self.parser.add_argument('--build_tc_wsz', type=int, default=5, help='window size for building initial temporal coherence')
        self.parser.add_argument('--save_model', action='store_true', help='whether or not to save hair model')
        self.parser.add_argument('--test_thresh', type=float, default=0.2, help='test threshold for occ')
        self.isTrain = False