class PuzzleConfigs:
    def __init__(self, num_rows, num_cols, solution_consideration_cutoff, apply_gaussian_blur, gaussian_blur_radius,
                 show_solution, log_to_screen):
        super().__init__()
        self.NUM_ROWS = num_rows
        self.NUM_COLS = num_cols
        self.SOLUTION_CONSIDERATION_CUTOFF = solution_consideration_cutoff
        self.GAUSSIAN_BLUR = apply_gaussian_blur
        self.GAUSSIAN_BLUR_RADIUS = gaussian_blur_radius
        self.SHOW_SOLUTION = show_solution
        self.LOG_TO_SCREEN = log_to_screen

        self.IMG_WIDTH = None
        self.IMG_HEIGHT = None
        self.ROW_HEIGHT = None
        self.COL_WIDTH = None

    def fit_on_image(self, input_image_size):
        self.IMG_WIDTH = input_image_size[0]
        self.IMG_HEIGHT = input_image_size[1]

        self.ROW_HEIGHT = self.IMG_HEIGHT / self.NUM_ROWS
        self.COL_WIDTH = self.IMG_WIDTH / self.NUM_COLS

